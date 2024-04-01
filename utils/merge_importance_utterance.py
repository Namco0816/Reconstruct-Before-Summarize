import datasets
dataset_id = 'icsi'

merged_dialogue = datasets.load_from_disk(f"dataset/{dataset_id}_dialogue_merge")
importance_score = datasets.load_from_disk(f"dataset/{dataset_id}_importance_merge_attn")

current_file = merged_dialogue['validation'][0]['filename']

input_ids = [[], [], [], [], [], [], [], []]
importance_scores = [[], [], [], [], [], [], [], []]

final_input = []
final_scores = []
final_filename = []

for i, (m_dialogue, m_score) in enumerate(zip(merged_dialogue['validation'], importance_score)):
    f_name = m_dialogue['filename']
    if f_name != current_file:
        len_vec = [len(input_id) for input_id in input_ids]
        max_len = max(len_vec)
        max_index = len_vec.index(max_len)
        for j in range(len(len_vec)):
            if j != max_index:
                if len(importance_scores[j]) < max_len:
                    importance_scores[j] = importance_scores[j] + importance_scores[max_index][-(max_len - len(importance_scores[j])):]

        assert [len(l) for l in importance_scores] == [max_len] * 8, f'{[len(l) for l in importance_scores]} not equal to {max_len}'
        final_input.append(input_ids[max_index])
        final_scores.append(importance_scores)
        final_filename.append(current_file)
        current_file = f_name
        input_ids = [[], [], [], [], [], [], [], []]
        importance_scores = [[], [], [], [], [], [], [], []]

    append_index = i % 8
    input_ids[append_index] = input_ids[append_index] + m_score['history_ids']
    importance_scores[append_index] = importance_scores[append_index] + m_score['importance_scores']

len_vec = [len(input_id) for input_id in input_ids]
max_len = max(len_vec)
max_index = len_vec.index(max_len)
for i in range(len(len_vec)):
    if i != max_index:
        if len(importance_scores[i]) < max_len:
            importance_scores[i] = importance_scores[i] + importance_scores[max_index][-(max_len - len(importance_scores[i])):]

assert [len(l) for l in importance_scores] == [max_len] * 8 
final_input.append(input_ids[max_index])
final_scores.append(importance_scores)
final_filename.append(current_file)

dataset_dict = {
    'input_ids': final_input,
    'importance_scores': final_scores,
    'filename': final_filename
}

ami_with_importance = datasets.Dataset.from_dict(dataset_dict)
ami_with_importance.save_to_disk(f'dataset/{dataset_id}_with_importance_attn')