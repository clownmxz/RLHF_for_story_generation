from transformers import GPT2ForSequenceClassification, AutoTokenizer
import numpy as np
import torch

def get_classifier_data(filepath):
    max_len = 80
    labels = []     # 用来存储分类的标签
    contexts = []    # 用来存储文本内容
    token_list = [] # 用来存储token的id
    tokenizer = AutoTokenizer.from_pretrained("../pretrain_model/gpt-2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = eval(line)   # 将字符串转换为字典
            label = line["style"]
            if label == "<Sp>":
                labels.append(0)
            elif label == "<ROC>":
                labels.append(1)
            elif label == "<Fairy>":
                labels.append(2)
            else:
                raise ValueError(f"错误的标签：{label}")

            context = line["text"]
            context = " ".join(context)
            contexts.append(context)
            token = tokenizer.encode(context, max_length=max_len, truncation=True, padding='max_length')
            token_list.append(token)

    return labels, token_list

if __name__ == '__main__':

    labels, token_list = get_classifier_data("../data_sets/train.txt")

    # 打乱数据集
    random_seed = 32
    np.random.seed(random_seed)
    np.random.shuffle(labels)
    np.random.seed(random_seed)
    np.random.shuffle(token_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    batch_size = 32

    # 定义一些常用参数
    model = GPT2ForSequenceClassification.from_pretrained("../pretrain_model/gpt-2", num_labels=3)
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    train_length = len(labels)
    for epoch in range(epochs):
        train_num = train_length // batch_size
        train_loss = 0.
        train_correct = 0.
        for i in range(train_num):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch_input = torch.tensor(token_list[start:end]).to(device)
            batch_label = torch.tensor(labels[start:end]).to(device)

            pred = model(batch_input)["logits"]
            loss = loss_fn(pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += ((torch.argmax(pred, dim=-1) == batch_label).sum().item()) / len(batch_label)

        train_loss /= train_num
        train_acc = train_correct / train_num
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    torch.save(model.state_dict(), "../pretrain_model/classifier_model.pth")


