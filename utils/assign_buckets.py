import datasets
import torch
import math

dataset_id = 'ami'

channel_ks = [0.008] #[0.025, 0.02, 0.015, 0.01, 0.008, 0.006]
avg_ks = [0.064] #[0.2, 0.16, 0.12, 0.08, 0.064, 0.048]
for channel_k, avg_k in zip(channel_ks, avg_ks):
    ami_with_segments = datasets.load_from_disk(f"dataset/{dataset_id}_with_segments_c{channel_k}_a{avg_k}_attn_attn_grad_deletemoderandom_0.75")

    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
            """
            Adapted from Mesh Tensorflow:
            https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
            Translate relative position to a bucket number for relative attention. The relative position is defined as
            memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
            position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
            small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
            positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
            This should allow for more graceful generalization to longer sequences than the model has been trained on
            Args:
                relative_position: an int32 Tensor
                bidirectional: a boolean - whether the attention is bidirectional
                num_buckets: an integer
                max_distance: an integer
            Returns:
                a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
            """
            relative_buckets = 0
            if bidirectional:
                num_buckets //= 2
                relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
                relative_position = torch.abs(relative_position)
            else:
                relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
            # now relative_position is in the range [0, inf)

            # half of the buckets are for exact increments in positions
            max_exact = num_buckets // 2
            is_small = relative_position < max_exact

            # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
            relative_position_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).to(torch.long)
            relative_position_if_large = torch.min(
                relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
            )

            relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)

            current_buckets_num = relative_buckets[0]
            buckets_num = []
            bucket_order = 0
            for i in range(relative_buckets.shape[0]):
                if relative_buckets[i] != current_buckets_num:
                    bucket_order += 1
                    current_buckets_num = relative_buckets[i]

                buckets_num.append(bucket_order)

            return relative_buckets, buckets_num

    def get_buckets_order(example):
        channel_importance = example['channel_wise_importance']
        channel_segments = example['channel_seg_ids']

        avg_importance = example['avg_importance']
        avg_segments = example['avg_seg_ids']

        input_ids = example['input_ids']
        buckets_mask = torch.arange(len(input_ids))

        global_channel_buckets = 0
        global_channel_order = []
        for i, (seg_index, seg_anchor_index) in enumerate(zip(channel_segments, channel_importance)):
            seg = input_ids[seg_index[0]: seg_index[1]] if seg_index[1] != -1 else input_ids[seg_index[0]: ]
            seg_relative_position = buckets_mask[seg_index[0]: seg_index[1]] if seg_index[1] != -1 else buckets_mask[seg_index[0]: ]
            seg_relative_position = torch.tensor(seg_relative_position) - seg_anchor_index
            seg_num_buckets = len(seg) / len(input_ids) * 512 * 2
            relative_buckets, buckets_order = _relative_position_bucket(seg_relative_position, num_buckets=max(seg_num_buckets, 4), max_distance= max(4 * seg_num_buckets, 16))
            buckets_order = [b_o + global_channel_buckets for b_o in buckets_order]
            global_channel_buckets = max(buckets_order) + 1
            global_channel_order = global_channel_order + buckets_order
        example['channel_buckets_order'] = global_channel_order
        assert len(example['channel_buckets_order']) == len(example['input_ids'])

        global_avg_buckets = 0
        global_avg_order = []
        for i, (seg_index, seg_anchor_index) in enumerate(zip(avg_segments, avg_importance)):
            seg = input_ids[seg_index[0]: seg_index[1]] if seg_index[1] != -1 else input_ids[seg_index[0]: ]
            seg_relative_position = buckets_mask[seg_index[0]: seg_index[1]] if seg_index[1] != -1 else buckets_mask[seg_index[0]: ]
            seg_relative_position = torch.tensor(seg_relative_position) - seg_anchor_index
            seg_num_buckets = len(seg) / len(input_ids) * 512 * 2
            relative_buckets, buckets_order = _relative_position_bucket(seg_relative_position, num_buckets=max(seg_num_buckets, 4), max_distance= max(4 * seg_num_buckets, 16))
            buckets_order = [b_o + global_avg_buckets for b_o in buckets_order]
            global_avg_buckets = max(buckets_order) + 1
            global_avg_order = global_avg_order + buckets_order
        example['avg_buckets_order'] = global_avg_order
        assert len(example['avg_buckets_order']) == len(example['input_ids'])
        return example

    ami_with_segments = ami_with_segments.map(get_buckets_order)
    ami_with_segments.save_to_disk(f"dataset/{dataset_id}_with_assigned_buckets_c{channel_k}_a{avg_k}_attn_grad_deletemoderandom0.75")
