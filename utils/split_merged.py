import datasets
dataset_id = 'ami'

dataset_merged_with_attn_grad = datasets.load_from_disk(f"dataset/{dataset_id}_with_importance_attn")
dataset_orig = datasets.load_from_disk(f"dataset/{dataset_id}_en")

train_orig = dataset_orig['train']
val_orig = dataset_orig['validation']
test_orig = dataset_orig['test']

train = dataset_merged_with_attn_grad[:len(train_orig)]
validation = dataset_merged_with_attn_grad[len(train_orig) : len(train_orig) + len(val_orig)]
test = dataset_merged_with_attn_grad[len(train_orig) + len(val_orig) : ]

train['summary'] = train_orig['sub_summary']
validation['summary'] = val_orig['sub_summary']
test['summary'] = test_orig['sub_summary']

train = datasets.Dataset.from_dict(train)
validation = datasets.Dataset.from_dict(validation)
test = datasets.Dataset.from_dict(test)

dataset = datasets.DatasetDict(
    {
        "train": train,
        "validation": validation,
        "test": test
    }
)
dataset.save_to_disk(f"dataset/{dataset_id}_with_attn")