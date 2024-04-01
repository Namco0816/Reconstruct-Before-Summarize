import torch
import datasets

attn_grads = torch.load("")
history_ids = []
response_ids = []
importance_scores = []

for ag in attn_grads:
    history_id = ag['input_ids'][ag['input_ids'] != 1]
    src_len = history_id.shape[0]
    history_ids.append(history_id[1:-1])

    response_id = ag['label_ids'][ag['label_ids'] != 1]
    response_id = response_id[response_id != -100]
    tgt_len = response_id.shape[0]
    response_ids.append(response_id[1:-1])

    importance_score = ag['attn_grad'][:tgt_len, :src_len]
    importance_score = importance_score.mean(0)
    # importance_score = torch.nn.functional.softmax(importance_score)
    importance_scores.append(importance_score[1:-1])

dataset_dict = {
        'history_ids': history_ids,
        'response_ids': response_ids,
        'importance_scores': importance_scores
        }

dataset_dict = datasets.Dataset.from_dict(dataset_dict)
dataset_dict.save_to_disk("dataset/icsi_importance_merge_orig_bart")
