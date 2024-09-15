import torch
import torch.nn as nn
from transformers import BertTokenizer, GPT2Model
from torch.nn.parameter import Parameter

class GPT_2(nn.Module):
    def __init__(self, use_RLHF=True):
        super(GPT_2, self).__init__()
        self.use_RLHF = use_RLHF

        # 这一块内容可以构成Actor网络，机器人的行为可以定义为回复，具体到每个token的概率分布
        self.model = GPT2Model.from_pretrained("../pretrain_model/gpt2")
        self.lm_head = nn.Linear(768, 21128, bias=False)
        weight = torch.load("../model_save/gpt2_lm_head_weight.pth")
        self.lm_head.weight = Parameter(weight)

        # 这一块内容可以构成Critic网络，用于评估回复的质量，用一个简单的全连接层
        self.value_layer = nn.Sequential(
            nn.Linear(768, 1),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

    def forward(self, input_tokens):
        embedding = self.model(input_tokens)["last_hidden_state"]
        embedding = nn.Dropout(p=0.1)(embedding)
        logits = self.lm_head(embedding)

        if not self.use_RLHF:
            return logits
        else:
            value = self.value_layer(embedding)
            value = torch.squeeze(value, dim=-1)
            return logits, value

    @ torch.no_grad()
    def generate(self, max_length=50, prompt_token=None, temperature=1., top_p=0.9):

        prompt_token_list = list(prompt_token)
        for i in range(max_length):
            token_input = torch.tensor([prompt_token_list]).to("cuda")
            if self.use_RLHF:
                result, _ = self.forward(token_input)
            else:
                result = self.forward(token_input)
            logits = result[:, -1, :]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = self.sample_top_p(probs, top_p)
            next_token = next_token.reshape(-1)
            prompt_token_list.append(next_token.item())
        return prompt_token_list

    def sample_top_p(self, probs, top_p):
        # 对probs进行降序排序，得到排序后的概率和对应的索引
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # 累积概率
        cum_probs = torch.cumsum(probs_sort, dim=-1)
        # 计算mask，如果概率减去累积概率大于top_p，则mask为True
        mask = probs_sort - cum_probs > top_p
        # 将mask为True的概率置为0
        probs_sort[mask] = 0
        # 将概率归一化
        probs_sort = probs_sort / torch.sum(probs_sort, dim=-1, keepdim=True)
        # 从归一化后的概率中采样一个概率
        new_token = torch.multinomial(probs_sort, num_samples=1)
        # 根据采样得到的概率，从排序后的索引中找到对应的token
        new_token = probs_idx.gather(dim=-1, index=new_token)

        return new_token








