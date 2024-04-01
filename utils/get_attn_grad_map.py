import torch
import datasets

attn_grads = torch.load("")
history_ids = []
response_ids = []
grads = []
attns = []
grad_attns = []

for ag in attn_grads:
    history_id = ag['input_ids'][ag['input_ids'] != 1]
    src_len = history_id.shape[0]
    history_ids.append(history_id[1:-1])

    response_id = ag['label_ids'][ag['label_ids'] != 1]
    response_id = response_id[response_id != -100]
    tgt_len = response_id.shape[0]
    response_ids.append(response_id[1:-1])

    grad = ag['grad'][:tgt_len, :src_len]
    # importance_score = importance_score.mean(0)
    # importance_score = torch.nn.functional.softmax(importance_score)
    grads.append(grad[1:-1, 1:-1])

    attn = ag['attn'][:tgt_len, :src_len]
    # importance_score = importance_score.mean(0)
    # importance_score = torch.nn.functional.softmax(importance_score)
    attns.append(attn[1:-1, 1:-1])

    attn_grad = ag['attn_grad'][:tgt_len, :src_len]
    # importance_score = importance_score.mean(0)
    # importance_score = torch.nn.functional.softmax(importance_score)
    grad_attns.append(attn_grad[1:-1, 1:-1])

dataset_dict = {
        'history_ids': history_ids,
        'response_ids': response_ids,
        'grad_map': grads,
        'attn_map': attns,
        'grad_attn_map': grad_attns,
        }

dataset_dict = datasets.Dataset.from_dict(dataset_dict)
dataset_dict.save_to_disk("dataset/ami_map")
