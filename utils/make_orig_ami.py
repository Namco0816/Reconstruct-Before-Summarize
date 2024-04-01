import datasets
import make_meeting_list
import string
import glob

eval_list = make_meeting_list.icsi_eval_set
test_list = make_meeting_list.icsi_test_set
train_list = make_meeting_list.icsi_train_set

act_train_list = []
# for id in train_list:
#     sub_train_list = glob.glob(f'dataset/utterance/meeting/ami_with_action/{id}_*_utterances.txt')
#     sub_ids = [i.split("/")[-1].strip("_utterances.txt") for i in sub_train_list]
#     act_train_list += sub_ids

# act_eval_list = []
# for id in eval_list:
#     sub_eval_list = glob.glob(f'dataset/utterance/meeting/ami_with_action/{id}_*_utterances.txt')
#     sub_ids = [i.split("/")[-1].strip("_utterances.txt") for i in sub_eval_list]
#     act_eval_list += sub_ids

# act_test_list = []
# for id in test_list:
#     sub_test_list = glob.glob(f'dataset/utterance/meeting/ami_with_action/{id}_*_utterances.txt')
#     sub_ids = [i.split("/")[-1].strip("_utterances.txt") for i in sub_test_list]
#     act_test_list += sub_ids

speak_starts = (
    "Project Manager: ",
    "Industrial Designer: ",
    "User Interface Designer: ",
    'Marketing Expert: ',
)
icsi_starts = (
    "Grad: ",
    "Undergrad: ",
    "Professor: ",
    "PhD: ",
    "Postdoc: "
)

def merge_ref_utt(data_ids):
    data_samples = []
    for data_id in data_ids:
        utter = f'dataset/utterance/meeting/icsi/{data_id}_utterances.txt'
        ref = f'dataset/utterance/meeting/icsi/{data_id}_reference.txt'
        data = {'utterances': None, 'sub_summary': None, 'file_name': data_id}
        orig_utterances = open(utter).read()
        utt = orig_utterances.split('\n')[0]
        for orig_utt in orig_utterances.split("\n")[1:]:
            if orig_utt.startswith(speak_starts) or orig_utt.startswith(icsi_starts):
                utt = utt + '\n' + orig_utt
            else:
                if utt[-1] in string.punctuation:
                    utt = utt + " " + orig_utt
                else:
                    utt = utt + '. ' + orig_utt
        data['utterances'] = utt
        data['sub_summary'] = open(ref).read()
        data_samples.append(data)

    return data_samples

train = datasets.Dataset.from_list(merge_ref_utt(train_list))
val = datasets.Dataset.from_list(merge_ref_utt(eval_list))
test = datasets.Dataset.from_list(merge_ref_utt(test_list))


ami = datasets.DatasetDict({'train':train, 'validation': val, 'test': test})
ami.save_to_disk("dataset/icsi_en_new")