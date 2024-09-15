import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from transformers import AutoTokenizer, GPT2ForSequenceClassification
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

from ppo import PPOTrainer
from model import GPT_2

config = {
    "steps": 25000,
    "batch_size": 128,
    "forward_batch_size": 16,
    "ppo_epochs": 4,
    "lr": 2e-6,
    "init_kl_coef": 0.2,
    "target": 6,
    "horizon": 10000,
    "gamma": 1,
    "lam": 0.95,
    "cliprange": .2,
    "cliprange_value": .2,
    "vf_coef": .1,
    "gen_len": 16,
    "save_freq": 5,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe_device = 0 if torch.cuda.is_available() else -1

# prompt池
prompts = ["<Sp>", "<ROC>", "<Fairy>"]

# 加载风格分类模型，作为reward model，需要提前训练出相应模型
class_model = GPT2ForSequenceClassification.from_pretrained("../pretrain_model/classifier_model.pth")

# 加载GPT-2模型，作为参考模型和优化模型
# 从预训练模型中加载GPT2分词器
gpt_tokenizer = AutoTokenizer.from_pretrained("../pretrain_model/gpt2")
# 将GPT2的分词器的结束符设置为填充符
gpt_tokenizer.eos_token = gpt_tokenizer.pad_token
# 创建GPT2微调语言模型，并设置使用RLHF
gpt_model = GPT_2(use_RLHF=True)
# 创建参考GPT2微调语言模型，并设置使用RLHF
ref_gpt_model = GPT_2(use_RLHF=True)
gpt_model.load_state_dict(torch.load("../pretrain_model/llm_model.pth"))
ref_gpt_model.load_state_dict(torch.load("../pretrain_model/llm_model.pth"))

# 将GPT2模型移动到指定设备上
gpt_model.to(device)
# 将参考GPT2模型移动到指定设备上
ref_gpt_model.to(device)


# 加载PPO优化训练器
ppo_trainer = PPOTrainer(gpt_model, ref_gpt_model, gpt_tokenizer, **config)
total_ppo_epochs = int(config["steps"] / config["batch_size"])


# 训练开始
image_list = []
for epoch in tqdm(range(total_ppo_epochs)):
    logs, timing = dict(), dict()
    t0 = time.time()

    batch = {"tokens": [], "query": []}

    # 初始化输入数据，prompts和其对应的token
    for _ in range(config["batch_size"]):
        # 随机选择一个prompt
        prompt = random.choice(prompts)
        # 将自然语言的prompt编码成token，会自动在前面加上bos，结尾加上eos
        tokens = gpt_tokenizer.encode(prompt)
        batch["tokens"].append(tokens)
        batch["query"].append(prompt)
    # 错位输入就是gpt2系列模型的特征，t[:-1]
    query_tensors = [torch.tensor(t[:-1]).long().to(device) for t in batch["tokens"]]

    t = time.time()
    response_tensors = []
    # 初始化生成数据，作为响应数据
    for i in range(config["batch_size"]):
        gen_len = config["gen_len"]
        prompt_token = query_tensors[i].detach().cpu().numpy()
        response = gpt_model.generate(max_length=gen_len, prompt_token=prompt_token)
        # 获取对应的模型的生成部分作为响应数据，去掉prompt的部分
        response_tensors.append(torch.tensor(response[-gen_len:]).long().to(device))
    batch["response"] = [gpt_tokenizer.decode(r) for r in response_tensors]
    timing["time/get_response"] = time.time() - t

    # 通过奖励模型计算当前状态得分
    t = time.time()
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]  # 完整的句子输入
    cur_style = texts[0]
    pipe_outputs = class_model.classfier(texts[1:])  # 判断当前内容的风格特征
    reward = []

    for output in pipe_outputs:
        if output["label"] == cur_style:
            reward.append(output["score"])
        else:
            reward.append(1 - output["score"])
    reward_tensors = torch.tensor(reward).to(device)
    timing["time/get_reward"] = time.time() - t

    t = time.time()
    # 训练PPO模型
    states = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

    mean_reward = torch.mean(reward_tensors).cpu().numpy()
    image_list.append(mean_reward)
    print()
    print(f"epoch {epoch + 1} mean-reward: {mean_reward}", 'Random Sample 5 text(s) of model output:')
    for i in range(5):  # 随机打5个生成的结果
        print(f'{i + 1}. {random.choice(texts)}')

torch.save(gpt_model.state_dict(),"model_save/gpt2_model.pth")
torch.save(ref_gpt_model.state_dict(),"model_save/ref_gpt_model.pth")

print(image_list)
plt.plot(image_list)
plt.show()
















