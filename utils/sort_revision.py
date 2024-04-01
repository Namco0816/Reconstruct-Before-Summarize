import datasets

merged_revision = datasets.load_from_disk("dataset/merged_revision_ver1")

def map_fun(example):
    utter = example['utterances']
    utter = utter.split("Speaker")
    utter = "Speaker" + utter[0] + "\r\nSpeaker".join(utter[1:])
    example['utterances'] = utter
    return example

merged_revision = merged_revision.map(map_fun)
merged_revision.save_to_disk('dataset/merged_revision_ver1')
print(merged_revision)