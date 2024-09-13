import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

def get_data(filepath):
    token_list = []
    label_list = []
    tokenizer = BertTokenizer.from_pretrained("../pretrain_model/gpt2")
    with open(filepath, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            line = dict(line)
            line = line["texts"]
            label = line["style"]
            if label == "<Sp>":
                token_list.append(0)
            elif label == '<St>':
                token_list.append(1)
            else:
                token_list.append(2)
            text = "".join(line[1:])
            input = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
            input_ids = input["input_ids"]
            attention_mask = input["attention_mask"]
            for id in input_ids[0]:
                token_list.append(id.item())
    f.close()

    token_list = torch.tensor(token_list * 5)
    label_list = torch.tensor(label_list * 5)
    return token_list, label_list

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