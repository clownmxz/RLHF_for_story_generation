import torch.nn.functional as F
import torch
import collections
from torch.nn.utils.rnn import pad_sequence
import numpy as np

WANDB_PADDING = -1

def flatten_dict(nested, sep='/'):
    # 定义一个递归函数，用于将嵌套字典展开
    def rec(nest, prefix, into):
        # 遍历嵌套字典的键值对
        for k, v in nest.items():
            # 如果键中包含分隔符，则抛出异常
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            # 如果值是字典，则递归调用rec函数
            if isinstance(v, collections.Mapping):
                rec(v, prefix + k + sep, into)
            # 否则，将键值对添加到展开后的字典中
            else:
                into[prefix + k] = v
    # 创建一个空字典，用于存储展开后的结果
    flat = {}
    # 调用rec函数，将嵌套字典展开
    rec(nested, '', flat)
    # 返回展开后的字典
    return flat

def stack_dicts(stats_dicts):
    # 创建一个空字典，用于存储结果
    results = dict()
    # 遍历第一个字典的键
    for k in stats_dicts[0]:
        # 将每个字典中对应键的值展平，并存储在一个列表中
        stats_list = [torch.flatten(d[k]) for d in stats_dicts]
        # 将展平后的列表进行填充，并存储在结果字典中
        results[k] = pad_sequence(stats_list, batch_first=True, padding_value=WANDB_PADDING)
    # 返回结果字典
    return results

# 定义一个函数，用于从logits中计算log概率
def logprobs_from_logits(logits, labels):
    # 使用log_softmax函数计算log概率
    logprobs = F.log_softmax(logits, dim=-1)
    # 使用gather函数从log概率中提取对应label的log概率
    logprobs = logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    # 返回log概率
    return logprobs

def whiten(values, shift_mean=True):
    # 计算values的均值和方差
    mean, var = torch.mean(values), torch.var(values)
    # 将values标准化
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    # 如果shift_mean为False，则将标准化后的值加上均值
    if not shift_mean:
        whitened += mean
    # 返回标准化后的值
    return whitened

def clip_by_value(x, tensor_min, tensor_max):
    # 使用torch.max和torch.min函数将x裁剪到tensor_min和tensor_max之间
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped

def entropy_from_logits(logits):
    # 计算logits的softmax值
    pd = torch.nn.functional.softmax(logits, dim=-1)
    # 计算entropy
    entropy = -torch.sum(pd * torch.log(pd), dim=-1)
    # 返回熵值
    return entropy

def stats_to_np(stats_dict):
    # 创建一个新的字典
    new_dict = dict()
    # 遍历stats_dict中的键值对
    for k, v in stats_dict.items():
        # 如果值是torch.Tensor类型
        if isinstance(v, torch.Tensor):
            # 将值从GPU移动到CPU，并转换为numpy数组
            new_dict[k] = v.detach().cpu().numpy()
        else:
            # 否则，将值直接赋给新的字典
            new_dict[k] = v
        # 如果新的字典中的值是标量
        if np.isscalar(new_dict[k]):
            # 将标量转换为浮点数
            new_dict[k] = float(new_dict[k])
    # 返回新的字典
    return new_dict

