import torch
import math

def get_buckets(seq_length, num_buckets = 20, max_distance = 128):
    context_position = torch.arange(seq_length, dtype=torch.long)[:, None]
    memory_position = torch.arange(seq_length, dtype=torch.long)[None, :]
    relative_position = memory_position - context_position
    relative_buckets = 0
    num_buckets //= 2
    relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
    relative_position = torch.abs(relative_position)
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
    return relative_buckets

r = get_buckets(256)
print(r[128])
