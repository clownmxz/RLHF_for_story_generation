import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import GPT_2

def get_data(filepath):
    token_list = []
    tokenizer = BertTokenizer.from_pretrained("../pretrain_model/gpt2")
    with open(filepath, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            line = eval(line)   # 将字符串转换为字典
            text = line["text"]
            input = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
            input_ids = input["input_ids"]
            attention_mask = input["attention_mask"]
            for id in input_ids[0]:
                token_list.append(id.item())
    f.close()
    token_list = torch.tensor(token_list * 5)
    return token_list

class TextSamplerDataset(Dataset):
    # 初始化函数，传入数据data和序列长度seq_len
    def __init__(self, data, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.data = data

    # 返回数据集的长度
    def __len__(self):
        return self.data.size(0) // self.seq_len

    # 根据索引返回数据
    def __getitem__(self, index):
        # 随机生成一个起始位置
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,)).item()
        # 获取从起始位置开始的序列
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        # 返回序列的前self.seq_len个元素和后self.seq_len个元素
        return full_seq[:-1], full_seq[1:]

if __name__ == '__main__':
    max_length = 128 + 1
    batch_size = 12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = "../pretrain_model/llm_model.pth"
    glm_model = GPT_2(use_RLHF=False)
    glm_model.to(device)

    optimizer = torch.optim.AdamW(glm_model.parameters(), lr=2e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1200, eta_min=2e-6, last_epoch=-1)
    criterion = torch.nn.CrossEntropyLoss()

    token_list = get_data("data_sets/train.txt")
    train_dataset = TextSamplerDataset(token_list, max_length)

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    for epoch in range(30):
        pbar = tqdm(loader, total=len(loader))
        for token_inp, token_tgt in pbar:
            token_inp = token_inp.to(device)
            token_tgt = token_tgt.to(device)

            logits = glm_model(token_inp)
            loss = criterion(logits.view(-1, logits.size(-1)), token_tgt.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()  # 执行优化器
            pbar.set_description(
                f"epoch:{epoch + 1}, train_loss:{loss.item():.5f}, lr:{lr_scheduler.get_last_lr()[0] * 100:.5f}")
        if (epoch + 1) % 2 == 0:
            torch.save(glm_model.state_dict(), save_path)