"""
This gives a list of lists of data:

[image id, question id, question, answer]
"""


import utils
import config
import json

# questions
with open('train_14_questions.json' ,'r') as f:
    questions_json = json.load(f)

with open('train_14_answers.json', 'r') as f:
    answers_json = json.load(f)

questions = [(q['image_id'], q['question_id'], q['question']) for q in questions_json]

dataset_ques = []
for iid, qid, question in questions:
    question = question.lower()[:-1]
    dataset_ques.append((iid, qid, question.split(' ')))

lad = [ans_dict for ans_dict in answers_json]

dataset_answ = []
for answer_dict in lad:
    iid = answer_dict['image_id']
    qid = answer_dict['question_id']
    for answer in answer_dict['answers']:
        if answer['answer_confidence'] == 'yes':
            ans = answer['answer']
            break
    dataset_answ.append((iid, qid, ans))

dataset = []    
for idx, answ in enumerate(dataset_answ):
    iid, qid, ans = answ
    _, _, question_list = dataset_ques[idx]
    dataset.append((iid, qid, question_list, ans))

with open('train_qna.json', 'w') as f:
    json.dump(dataset, f)