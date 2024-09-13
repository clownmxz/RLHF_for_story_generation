import random

import numpy as np
import torch
import time
from transformers import DataCollatorForLanguageModeling
from utils import (logprobs_from_logits,
                   whiten, clip_by_value,
                   entropy_from_logits,
                   flatten_dict,
                   stack_dicts,
                   WANDB_PADDING,
                   stats_to_np)


class AdaptiveKLController:
    # 这个类的主要目的是根据当前KL散度和目标KL散度之间的差异，动态调整KL散度的系数，以实现某种优化目标。
    def __init__(self, init_kl_coef, target, horizon):
        # 初始化KL系数、目标KL系数和预测时间范围
        self.init_kl_coef = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        # 获取目标KL系数
        target = self.target
        # 计算比例误差
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        # 计算乘数
        mult = 1 + proportional_error * n_steps / self.horizon
        # 更新KL系数
        self.init_kl_coef *= mult


class PPOTrainer:
    default_params = {
        "lr": 1.41e-6,  # Adam优化器的学习率
        "init_kl_coef": 0.2,  # 初始的KL散度
        "target": 6,  # 目标KL散度
        "horizon": 10000,  # 比例范围因子，控制更新KL的幅度
        "gamma": 1,  # 优势函数计算时的gamma参数
        "lam": 0.95,  # 优势函数计算时的lambda参数
        "cliprange": .2,  # PPO算法中的裁剪范围
        "cliprange_value": .2,  # 裁剪值函数的范围
        "vf_coef": .1,  # 价值函数的比例因子
        "batch_size": 256,  # 每一个优化步的样本数量
        "forward_batch_size": 16,  # 这个参数表示每次前向传播的样本数量
        "ppo_epochs": 4,  # 这个参数表示PPO算法的迭代次数
    }

    def __init__(self, model, ref_model, tokenizer, **ppo_params):

        self.ppo_params = self.default_params
        self.ppo_params.update(ppo_params)

        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        self.kl_controller = AdaptiveKLController(
            self.ppo_params["init_kl_coef"],
            self.ppo_params["target"],
            self.ppo_params["horizon"]
        )

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.ppo_params["lr"])

    def step(self, queries, responses, scores):
        """
        确定PPO优化器每一步应做的事情
        :param queries:  模型的prompts，List[tensor]，len(queries) = batch_size，tensor的shape为[query_len]
        :param responses:  模型的response，List[tensor]，len(responses) = batch_size，tensor的shape为[response_len]
        :param scores:  上一步得到的评分, List[tensor]，len(scores) = batch_size
        :return: 以字典形式返回所有的训练信息
        """

        batch_size = self.ppo_params["batch_size"]
        assert batch_size == len(
            queries), f"Batch size ({batch_size}) does not match number of examples ({len(queries)})"

        timing = dict()  # 用来记录每一步优化所需时间信息
        t0 = time.time()

        t = time.time()
        # 获取模型的生成部分的log_probs, ref_log_probs, 模型评分value
        logprobs, ref_logprobs, values = self.batch_forward_pass(queries, responses)
        timing['time/ppo/forward_pass'] = time.time() - t

        t = time.time()
        # 计算折扣奖励分数
        rewards, non_scores_rewards = self.compute_rewards(logprobs, ref_logprobs, scores)
        timing['time/ppo/compute_rewards'] = time.time() - t

        t = time.time()
        all_statistics = []
        idxs = list(range(batch_size))
        for _ in range(self.ppo_params["ppo_epochs"]):
            random.shuffle(idxs)  # 打乱样本顺序
            for i in range(batch_size):
                idx = idxs[i]
                # logprobs: [batch_size, response_len]
                # values: [batch_size, response_len]
                # rewards: [batch_size, response_len]
                # queries: [batch_size, query_len]
                # responses: [batch_size, response_len]
                # model_inputs: [batch_size, query_len + response_len]
                train_statistics = self.train_one_minibatch(logprobs[idx].unsqueeze(0),
                                                            values[idx].unsqueeze(0),
                                                            rewards[idx].unsqueeze(0),
                                                            queries[idx].unsqueeze(0),
                                                            responses[idx].unsqueeze(0),
                                                            torch.cat([queries[idx], responses[idx]], dim=-1).unsqueeze(
                                                                0))
                all_statistics.append(train_statistics)
        timing['time/ppo/optimize_step'] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_statistics)

        train_stats['policy/advantages'] = torch.flatten(train_stats['policy/advantages']).unsqueeze(0)
        train_stats['policy/advantages'] = torch.nan_to_num(train_stats['policy/advantages'], WANDB_PADDING)
        train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)

        stats = self.record_step_stats(scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs,
                                       non_score_reward=non_scores_rewards, train_stats=train_stats,
                                       kl_coef=self.kl_controller.init_kl_coef)
        stats = stats_to_np(stats)

        timing['time/ppo/calc_stats'] = time.time() - t

        self.kl_controller.update(stats['objective/kl'], self.ppo_params['batch_size'])

        timing['time/ppo/total'] = time.time() - t0

        stats.update(timing)

        return stats

    def batch_forward_pass(self, queries, responses):

        batch_size = self.ppo_params["batch_size"]
        forward_batch_size = self.ppo_params["forward_batch_size"]

        all_logprobs = []
        all_ref_logprobs = []
        all_values = []

        # batch_size是PPO优化器每一步所传入的样本数，forward_batch_size是传入给模型的样本数，因此在优化的每一步，模型需要执行的次数为
        # batch_size // forward_batch_size次
        for i in range(int(batch_size // forward_batch_size)):
            cur_queries_batch = queries[i * forward_batch_size: (i + 1) * forward_batch_size]
            cur_responses_batch = responses[i * forward_batch_size: (i + 1) * forward_batch_size]
            # 这一步是为了将queries和responses拼接起来，因为对于GPT2而言，上一步产生的responses要和queries拼接在一起作为这一步的输入
            input_ids = \
                self.data_collator([torch.cat([q, r], dim=-1) for q, r in zip(cur_queries_batch, cur_responses_batch)])[
                    "input_ids"]
            with torch.no_grad():
                logits, value = self.model(input_ids)  # value的值来自fine tune的model
                ref_logits, _ = self.ref_model(input_ids)  # 参考模型，不返回评分值
                # logits和ref_logits的形状为[forward_batch_size, seq_len, vocab_size], seq_len = query_len + response_len
                # value的形状为[forward_batch_size, seq_len], seq_len = query_len + response_len

            # logits[:, :-1, :]和input_ids[:, 1:]是为了将每个时间步下的输入序列和输出序列进行对齐
            loglogits = logprobs_from_logits(logits[:, :-1, :],
                                             input_ids[:, 1:])  # loglogits的形状为[forward_batch_size, seq_len - 1]
            ref_loglogits = logprobs_from_logits(ref_logits[:, :-1, :],
                                                 input_ids[:, 1:])  # ref_loglogits的形状为[forward_batch_size, seq_len - 1]

            for j in range(forward_batch_size):
                # 这里的start和end就代表了模型生成的部分的信息，去掉了prompt部分，只将responses的部分的信息加入
                start = len(cur_queries_batch[j]) - 1
                end = start + len(cur_responses_batch[j])
                all_logprobs.append(loglogits[j, start:end])
                all_ref_logprobs.append(ref_loglogits[j, start:end])
                all_values.append(value[j, start:end])

        return all_logprobs, all_ref_logprobs, all_values

    def compute_rewards(self, logprobs, ref_logprobs, scores):
        """
        依据KL散度和奖励分数计算每一个token的奖励

        :param logprobs: 优化模型的生成部分的概率分布
        :param ref_logprobs: 参考模型的生成部分的概率分布
        :param scores: 奖励模型生成的奖励分数
        :return:
        """
        rewards, non_scores_rewards = [], []
        for score, logprob, ref_logprob in zip(scores, logprobs, ref_logprobs):
            kl = logprob - ref_logprob  # 计算KL散度，(response_len, )
            # 这里为什么是负的kl_controller.init_kl_coef和kl相乘，还没有搞懂
            non_scores_reward = - self.kl_controller.init_kl_coef * kl  # 计算非奖励部分的奖励，(response_len, )
            non_scores_rewards.append(non_scores_reward)
            reward = non_scores_reward.clone()  # 除了最后一位的所有token的奖励都来自KL散度
            reward[-1] += score  # 最后一位token的奖励有来自人工评分的部分
            rewards.append(reward)
        return rewards, non_scores_rewards  # (batch_size, response_len)

    def train_one_minibatch(self, logprob, value, reward, query, response, model_input):
        """
        训练一个batch的数据
        :param logprob: [1, response_len, vocab_size]
        :param value: [1, response_len]
        :param reward: [1, response_len]
        :param query: [1, query_len]
        :param response: [1, response_len]
        :param model_input: [1, query_len + response_len]
        :return:
        """
        loss_p, loss_v, train_statistics = self.loss(logprob, value, reward, query, response, model_input)
        # 用两个损失函数来约束，目的是：
        # 1.希望RLHF能最大限度反馈生成模型的奖励值
        # 2.希望优化模型与参考模型的输出不要距离太远，否则偏离太严重，影响原本表达
        loss = loss_p + loss_v
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return train_statistics

    def loss(self, logprob, value, reward, query, response, model_input):
        """
        计算PPO的损失函数
        :param logprob: [1, response_len, vocab_size]
        :param value: [1, response_len]
        :param reward: [1, response_len]
        :param query: [1, query_len]
        :param response: [1, response_len]
        :param model_input: [1, query_len + response_len]
        :return:
        """
        lastgaelam = 0
        advantages_reversed = []
        gen_len = response.shape[1]
        advantages = torch.zeros(size=(1, gen_len))
        # 对于强化学习的逆序，可以理解为当前的结果决定了未来的结果，如果未来的结果很差，需要对模型进行修正，使其远离此结果，因此从未来逆序处理
        for t in reversed(range(gen_len)):
            nextvalue = value[:, t + 1] if t < gen_len - 1 else 0.0
            # 这就是优势函数的计算公式，由奖励函数和价值函数计算得到
            delta = reward[:, t] + self.ppo_params["gamma"] * nextvalue - value[:, t]
            lastgaelam = delta + self.ppo_params["gamma"] * self.ppo_params["lam"] * lastgaelam
            advantages_reversed.append(lastgaelam)
            # 一开始是逆序，真正的优势函数要再逆转一遍
            advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)   # [1, response_len]

        # 计算价值函数的损失
        returns = advantages + value  # 回报由之前的价值函数和优势函数得到， [1, response_len]
        pred_logits, pred_value = self.model(model_input)  # [1, seq_len, vocab_size], [1, seq_len]
        pred_logprob = logprobs_from_logits(pred_logits[:, :-1, :], model_input[:, 1:])  # [1, seq_len - 1]
        pred_logprob, pred_value = pred_logprob[:, -gen_len:], pred_value[:, -gen_len:]  # 只保留生成内容部分的信息
        # 对当前价值函数进行裁剪，限定范围，保持稳定性
        vpredclipped = clip_by_value(pred_value, value - self.ppo_params["cliprange_value"], value + self.ppo_params["cliprange_value"])
        vf_loss1 = (pred_value - returns) ** 2  # 均方误差，未裁剪
        vf_loss2 = (vpredclipped - returns) ** 2  # 均方误差，裁剪
        vf_loss = 0.5 * torch.mean(torch.max(vf_loss1, vf_loss2))  # 取最大值，防止经过裁剪后损失过大
        vf_clipfrac = torch.mean(torch.gt(vf_loss2, vf_loss1).double()) # 计算裁剪比例

        # 计算策略函数的损失
        advantages = whiten(advantages)     # 标准化
        advantages = advantages.detach()    # 优势函数不需要梯度传播，优势函数主要用于评估当前动作相对于平均动作的相对性能，而不是直接用于更新策略参数。
        ratio = torch.exp(pred_logprob - logprob)  # 计算新旧策略的比率
        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.ppo_params["cliprange"], 1.0 + self.ppo_params["cliprange"])
        pg_loss = torch.mean(torch.max(pg_losses1, pg_losses2))  # 取最大值，防止经过裁剪后损失过大
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses1).double())  # 计算裁剪比例

        loss = pg_loss + self.ppo_params["vf_coef"] * vf_loss   # 总损失

        entropy = torch.mean(entropy_from_logits(pred_logits))  # 计算熵，用于衡量模型的随机性，熵越大，模型的随机性越大，多样性越好
        approxkl = 0.5 * torch.mean((pred_logprob - logprob) ** 2) # 计算KL散度，用于衡量新旧策略的差异，KL散度越小，策略越稳定
        policykl = torch.mean(pred_logprob - logprob)  # 计算策略的KL散度，用于衡量新旧策略的差异，KL散度越小，策略越稳定
        return_mean, return_var = torch.mean(returns), torch.var(returns)
        value_mean, value_var = torch.mean(value), torch.var(value)

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, total=loss),
            policy=dict(entropy=entropy, approxkl=approxkl, policykl=policykl, clipfrac=pg_clipfrac,
                        advantages=advantages, advantages_mean=torch.mean(advantages), ratio=ratio),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(vpred=torch.mean(pred_value), error=torch.mean((pred_value - returns) ** 2),
                     clipfrac=vf_clipfrac, mean=value_mean, var=value_var),
        )

        return pg_loss, self.ppo_params["vf_coef"] * vf_loss, flatten_dict(stats)

    def record_step_stats(self, kl_coef, **data):
        kl_list = [logprobs - ref_logprobs for logprobs, ref_logprobs in zip(data['logprobs'], data['ref_logprobs'])]
        mean_kl = torch.mean(torch.stack([torch.sum(kl) for kl in kl_list]))
        mean_entropy = torch.mean(torch.stack([torch.sum(-log_probs) for log_probs in data['logprobs']]))
        mean_non_score_reward = torch.mean(
            torch.stack([torch.sum(non_score_reward) for non_score_reward in data['non_score_reward']]))
        stats = {
            'objective/kl': mean_kl,
            'objective/kl_dist': kl_list,
            'objective/logprobs': data['logprobs'],
            'objective/ref_logprobs': data['ref_logprobs'],
            'objective/kl_coef': kl_coef,
            'objective/entropy': mean_entropy,
            'ppo/mean_non_score_reward': mean_non_score_reward,
        }

        for k, v in data['train_stats'].items():
            stats[f'ppo/{k}'] = torch.mean(v, dim=0)
        stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
        return stats

