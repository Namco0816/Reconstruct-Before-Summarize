import datasets
import torch
import math
import random
from collections import Counter

dataset_id = 'ami'
strategy = 'random'
sub_ratio = 0.25

ami = datasets.load_from_disk(f"dataset/{dataset_id}_with_attn_grad")
train = ami['train']
val = ami['validation']
test = ami['test']
channel_ks =  [0.008] # [0.025, 0.02, 0.015, 0.01, 0.008, 0.006]
avg_ks = [0.064] #[0.2, 0.16, 0.12, 0.08, 0.064, 0.048]
for channel_k, avg_k in zip(channel_ks, avg_ks):
    def get_segments_id(example):
        input_ids = example['input_ids']
        importance_score = torch.tensor(example['importance_scores'])
        avg_importance = importance_score.mean(0)
        channel_wise_importance = torch.topk(
            importance_score,
            dim = 1,
            k = int(channel_k * importance_score.shape[-1])
        ).indices
        avg_importance = torch.topk(avg_importance, k = int(avg_k * avg_importance.shape[0])).indices

        channel_wise_importance, avg_importance = channel_wise_importance.tolist(), avg_importance.tolist()
        channel_wise_importance = [item for sublist in channel_wise_importance for item in sublist]
        channel_wise_importance = sorted(list(set(channel_wise_importance)))

        sub_len = int(sub_ratio * len(channel_wise_importance))
        index_needed_to_be_substitute = random.choices(channel_wise_importance, k = sub_len)
        if strategy == 'random':
            sub_token = random.choices(example['input_ids'], k = sub_len)

        else:
            high_freq_token = Counter(example['input_ids']).most_common(sub_len)
            sub_token = [item[0] for item in high_freq_token]

        for index, new_tok in zip(index_needed_to_be_substitute, sub_token):
            example['input_ids'][index] = new_tok


        avg_importance = sorted(avg_importance)

        channel_seg_start = torch.tensor(channel_wise_importance[:-1])
        channel_seg_end = torch.tensor(channel_wise_importance[1:])

        channel_segment_id = ((channel_seg_end - channel_seg_start) // 2 + channel_seg_start).tolist()
        channel_segments = []
        for i in range(len(channel_segment_id)):
            if i == 0:
                start = 0
            else:
                start = channel_segment_id[i - 1] + 1
            end = channel_segment_id[i] + 1
            if start < len(input_ids):
                channel_segments.append([start, end])

            if i == len(channel_segment_id) - 1:
                if channel_segment_id[i] + 1 < len(input_ids):
                    channel_segments.append([channel_segment_id[i] + 1, -1])

        avg_seg_start = torch.tensor(avg_importance[:-1])
        avg_seg_end = torch.tensor(avg_importance[1:])
        avg_segment_id = ((avg_seg_end - avg_seg_start) // 2 + avg_seg_start).tolist()

        avg_segments = []
        for i in range(len(avg_segment_id)):
            if i == 0:
                start = 0
            else:
                start = avg_segment_id[i - 1] + 1
            end = avg_segment_id[i] + 1
            if start < len(input_ids):
                avg_segments.append([start, end])

            if i == len(avg_segment_id) - 1:
                if avg_segment_id[i] + 1 < len(input_ids):
                    avg_segments.append([avg_segment_id[i] + 1, -1])

        example['channel_seg_ids'] = channel_segments
        example['avg_seg_ids'] = avg_segments
        example['avg_importance'] = avg_importance
        example['channel_wise_importance'] = channel_wise_importance

        return example
    ami = ami.map(get_segments_id)
    ami.save_to_disk(f"dataset/{dataset_id}_with_segments_c{channel_k}_a{avg_k}_attn_grad_subration{sub_ratio}_{strategy}")