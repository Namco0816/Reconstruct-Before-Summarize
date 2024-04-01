import datasets
import re
import string

text_column = 'utterances'
summary_column = 'sub_summary'
ami = datasets.load_from_disk("dataset/icsi_en")

def process_text(text):
    text = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text, flags=re.IGNORECASE) # YOU you You -> you
    text = re.sub(r'\s(\')\s([a-z])', r'\1\2', text) # pattern: ( ' t) -> ('t)
    text = re.sub(r'\s(\')([a-z])', r'\1\2', text) # pattern: ( 's) -> ('s)
    text = re.sub(r'(\w)\s(n\'t)', r'\1\2', text) # pattern: (was n't) -> (wasn't)
    text = re.sub(r'(\wn)\s(na)', r'\1\2', text) # pattern: (gon na) -> (gonna)
    text = re.sub(r'\s(\-)', r'\1', text)
    text = re.sub(r'(\-)\s', r'\1', text)
    text = re.sub(r'\s([,?.!()"])', r'\1', text) # (s ') -> (s') 
    text = re.sub(r'\s([a-z]+?)(\s[a-z]:)[\W\D]', r' \1\2', text)
    text = re.sub(r'\s([a-z]+?)(\s[a-z]:)', r'\r\n\1\2', text)
    text = re.sub(r'\r\n([a-z]+?)(\s[a-z]:)(\S)', r'\r\n\1\2 \3', text)
    text = re.sub(r'([,?.!()":])\s([,?.!()":])+', r'\1', text)
    text = re.sub(r'([,?.!()":])([,?.!()":])+', r'\1', text)
    text = re.sub(r'\b([a-z])([,?.!()":])\b', r'. ', text)
    text = re.sub(r'([,?.!()":])(\w+\b)', r'\1 \2', text)
    text = text.replace("_", '')
    # text = text.lstrip()

    return text

inputs, targets, file_name = [],[], []
def preprocess_function(examples):
    # remove pairs where at least one record is None
    for i in range(len(examples[text_column])):
        if examples[text_column][i] is not None and examples[summary_column][i] is not None:
            # history_list = examples[text_column][i].split("\n")
            examples[text_column][i] = process_text(examples[text_column][i])
            examples[text_column][i] = examples[text_column][i].replace("\n", ". ")
            # history_list = [item.replace("\n", ". ") for item in examples[text_column][i]]
            history_list = examples[text_column][i].split(".")
            history_list = history_list + ['[END_DIALOGUE]']
            # history_list = [item.split(".") for item in history_list]
            # history_list = [item for sublist in history_list for item in sublist]
            history_list = [item for item in history_list if item != ""]                
            for j in range(1, len(history_list)):
                history = ".".join(history_list[max(0, j - 8): j])
                history = process_text(history)
                if history[-1] not in string.punctuation:
                    history = history + '.'

                response = history_list[j]
                response = response.lstrip()
                # if response.startswith("Project Manager:"):
                #     response = response.replace("Project Manager:", "")
                #     history = history + " Project Manager: "

                # elif response.startswith("Industrial Designer:"):
                #     response = response.replace("Industrial Designer:", "")
                #     history = history + " Industrial Designer: "

                # elif response.startswith("Marketing Expert:"):
                #     response = response.replace("Marketing Expert:", "")
                #     history = history + " Marketing Expert: "

                # elif response.startswith("User Interface Designer:"):
                #     response = response.replace("User Interface Designer:", "")
                #     history = history + " User Interface Designer: "
                
                history = process_text(history)
                
                inputs.append(history)     
                targets.append(response)
                file_name.append(examples['file_name'][i])

ami['train'].map(preprocess_function, batched=True)
        
ami_train_dialogue = {
    'history': inputs,
    'response': targets,
    'file_name': file_name
    }

ami_dialogue_train = datasets.Dataset.from_dict(ami_train_dialogue)

inputs, targets, file_name = [], [], []
ami['validation'].map(preprocess_function, batched=True)
        
ami_validation_dialogue = {
    'history': inputs,
    'response': targets,
    'file_name': file_name
    }

ami_dialogue_validation = datasets.Dataset.from_dict(ami_validation_dialogue)

inputs, targets, file_name = [], [], []
ami['test'].map(preprocess_function, batched=True)
        
ami_test_dialogue = {
    'history': inputs,
    'response': targets,
    'file_name': file_name
    }

ami_dialogue_test = datasets.Dataset.from_dict(ami_test_dialogue)

ami_dialogue = datasets.DatasetDict({
    'train': ami_dialogue_train,
    'validation': ami_dialogue_validation,
    'test': ami_dialogue_test
})

print(ami_dialogue)
ami_dialogue.save_to_disk("dataset/icsi_dialogue")