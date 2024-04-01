import re
import make_meeting_list

ids = make_meeting_list.ami_meeting_list

for id in ids:
    utter_path = f'resources/data/meeting/ami_zh/{id}.da.zh'
    action_path = f'resources/data/meeting/ami_zh/{id}.actions.zh'
    decision_path = f'resources/data/meeting/ami_zh/{id}.decisions.zh'
    problem_path = f'resources/data/meeting/ami_zh/{id}.problems.zh'

    buffer = []

    act_dict = {}
    decision_dict = {}
    problem_dict = {}

    act_list = []
    decision_list = []
    problem_list = []

    sub_idx = 0
    with open(
        utter_path, 
        'r'
        ) as utter_r, open(
            action_path, 
            'r'
            ) as action_r, open(
                decision_path, 
                'r'
                ) as decision_r, open(
                    problem_path,
                    'r'
                ) as problem_r:

        for line in action_r:
            if line is not "\n":
                line_action = line.split('\t')        
                if len(line_action) == 3:
                    cur_action = line_action[2]
                else:
                    act_dict[line_action[1]] = cur_action

        for line in decision_r:
            if line is not "\n":
                line_decision = line.split('\t')        
                if len(line_decision) == 3:
                    cur_decision = line_decision[2]
                else:
                    decision_dict[line_decision[1]] = cur_decision
        
        for line in problem_r:
            if line is not "\n":
                line_problem = line.split('\t')        
                if len(line_problem) == 3:
                    cur_problem = line_problem[2]
                else:
                    problem_dict[line_problem[1]] = cur_problem

        for line in utter_r:
            if line is not "\n":
                if len(buffer) >= 50:
                    with open(
                        f'resources/data/meeting/ami_new/{id}_{sub_idx}.da.zh', 
                        'w'
                        ) as sub_utter, open(
                            f'resources/data/meeting/ami_new/{id}_{sub_idx}.sub_action.txt', 
                            'w'
                            ) as sub_action: 

                        sub_utter.write("\n".join(buffer))
                        if act_list != []:
                            sub_action.write("行动: " + "".join(act_list) + '\n')
                        else:
                            sub_action.write("行动: 不适用" + '\n')

                        if decision_list != []:
                            sub_action.write("决策: " + "".join(decision_list) + '\n')
                        else:
                            sub_action.write("决策: 不适用" + '\n')

                        if problem_list != []:
                            sub_action.write("问题: " + "".join(problem_list) + '\n')
                        else:
                            sub_action.write("问题: 不适用" + "\n")
                        
                    sub_idx += 1
                    buffer = []
                    act_list = []
                    decision_list = []
                    problem_list = []

                line_utter = line.split("\t")
                if line_utter[1] in act_dict.keys():
                    if act_dict[line_utter[1]] not in act_list:
                        act_list.append(act_dict[line_utter[1]])

                if line_utter[1] in decision_dict.keys():
                    if decision_dict[line_utter[1]] not in decision_list:
                        decision_list.append(decision_dict[line_utter[1]])
                
                if line_utter[1] in problem_dict.keys():
                    if problem_dict[line_utter[1]] not in problem_list:
                        problem_list.append(problem_dict[line_utter[1]])

                buffer.append(line)

        if buffer != []:
            with open(f'resources/data/meeting/ami_new/{id}_{sub_idx}.da.zh', 'w') as sub_utter, open(f'resources/data/meeting/ami_new/{id}_{sub_idx}.sub_action.txt', 'w') as sub_action:
                sub_utter.write("\n".join(buffer))

                if act_list != []:
                    sub_action.write("行动: " + "".join(act_list) + "\n")
                else:
                    sub_action.write("行动: 不适用"+ "\n")

                if decision_list != []:
                    sub_action.write("决策: " + "".join(decision_list)+ "\n")
                else:
                    sub_action.write("决策: 不适用"+ "\n")

                if problem_list != []:
                    sub_action.write("问题: " + "".join(problem_list)+ "\n")
                else:
                    sub_action.write("问题: 不适用"+ "\n")

            buffer = []
            act_list = []
            decision_list = []
            problem_list = []