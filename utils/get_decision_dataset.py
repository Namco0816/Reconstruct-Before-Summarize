import datasets

ami = datasets.load_from_disk("dataset/ami_zh_with_idx")

train = ami['train']
val = ami['validation']
test = ami['test']

data_samples = []
for instance in train:
    utter = train['utterances']
    utter_list = utter.split("\n\n")
    utter_count = len(utter_list)

    j = utter_count / 5
    k = ls_len%5

    ls_return = []
    for i in range(0,(5-1)*j,j):
        ls_return.append(ls[i:i+j])
    ls_return.append(ls[(5-1)*j:])
