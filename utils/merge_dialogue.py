import datasets
icsi = datasets.load_from_disk("dataset/icsi_dialogue")

train = icsi['train']
val = icsi['validation']
test = icsi['test']

history = train['history'] + val['history'] + test['history']
response = train['response'] + val['response'] + test['response']
filename = train['file_name'] + val['file_name'] + test['file_name']

dataset = {
    'history': history,
    'response': response,
    'filename': filename
}

dataset = datasets.Dataset.from_dict(dataset)
dataset = datasets.DatasetDict(
    {
        'train': dataset,
        'validation': dataset,
        'test': dataset
    }
)
dataset.save_to_disk("dataset/icsi_dialogue_merge")