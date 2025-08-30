import pandas as pd
behaviors = pd.read_csv('./raw_data/mind/MINDsmall_train/behaviors.tsv', sep='\t', header=None, names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
news = pd.read_csv('./raw_data/mind/MINDsmall_train/news.tsv', sep='\t', header=None, names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])


train_alpha = 10 # train set negative samples undersampling rate
val_alpha = 10 # valid set negative samples undersampling rate
test_alpha = 10 # test set negative samples undersampling rate


user_alpha = 3


tgt_fld_name = 'train_n'+str(train_alpha)+'_valid_n'+str(val_alpha)+'_test_n'+str(test_alpha)+'/'
tgt_folder_full = './data/MIND/' + tgt_fld_name + '/'

tgt_train_json = tgt_folder_full + 'train.json'
tgt_valid_json = tgt_folder_full + 'valid.json'
tgt_test_json = tgt_folder_full + 'test.json'


news_dict = {}

from tqdm import tqdm
user_dict = {}
news_id = {}

for index, row in tqdm(news.iterrows()):
    news_dict[row['news_id']] = index
for index, row in tqdm(behaviors.iterrows()):
    userid = row['user_id']
    if not user_dict.__contains__(userid):
        user_dict[userid] = {
            'history_news_ids': [],
            'history_titles': [],
            'history_catagories': [],
            'history_abstracts': [],
            'impression_news_ids': [],
            'impression_catagories': [],
            'impression_abstracts': [],
            'impression_titles': [],
            'impression_labels': [],
        }
    if row['history'] == 'NULL':
        continue
    histories = row['history'].split(' ')
    for history_news_id in histories:
        news_idx = news_dict[history_news_id]
        news_title = news.iloc[news_idx]['title']
        news_catagory = news.iloc[news_idx]['category']
        news_abstract = news.iloc[news_idx]['abstract']
        user_dict[userid]['history_news_ids'].append(news_idx)
        user_dict[userid]['history_catagories'].append(news_catagory)
        user_dict[userid]['history_titles'].append(news_title)
        user_dict[userid]['history_abstracts'].append(news_abstract)
    if row['impressions'] == 'NULL':
        continue
    impressions = row['impressions'].split(' ')
    for impression in impressions:
        impression_news_id, label = impression.split('-')
        news_idx = news_dict[impression_news_id]
        news_title = news.iloc[news_idx]['title']
        news_catagory = news.iloc[news_idx]['category']
        news_abstract = news.iloc[news_idx]['abstract']
        user_dict[userid]['impression_news_ids'].append(news_idx)
        user_dict[userid]['impression_titles'].append(news_title)
        user_dict[userid]['impression_labels'].append(int(label))
        user_dict[userid]['impression_catagories'].append(news_catagory)
        user_dict[userid]['impression_abstracts'].append(news_abstract)
        
        
new_user_dict = {}
for key in user_dict.keys():
    if len(user_dict[key]['historys'])  <= 5:
        pass
    else:
        new_user_dict[key] = user_dict[key]
        print(user_dict[key])

import random
import json

total_users = len(new_user_dict)
print('total users:', total_users)
used_users = int(total_users / user_alpha)
print('used users:', used_users)]

user_list = list(new_user_dict.keys())
random.shuffle(user_list)
user_list = user_list[:used_users]
train_user = user_list[:int(len(user_list) * 0.8)]
valid_user = user_list[int(len(user_list) * 0.8):int(len(user_list) * 0.9)]
test_user = user_list[int(len(user_list) * 0.9):]



def generate_json(user_list, output_json, split = 'train'):
    Prompt_json = []
    positive_list = []
    negative_list = []
    for user in user_list:
        history_news_ids = user_dict[user]['history_news_ids']
        history_titles = user_dict[user]['history_titles']
        history_catagories = user_dict[user]['history_catagories']
        history_abstracts = user_dict[user]['history_abstracts']
        impression_news_ids = user_dict[user]['impression_news_ids']
        impression_catagories = user_dict[user]['impression_catagories']
        impression_abstracts = user_dict[user]['impression_abstracts']
        impression_titles = user_dict[user]['impression_titles']
        impression_labels = user_dict[user]['impression_labels']

        random.seed(42)
        random.shuffle(history_news_ids)
        random.seed(42)
        random.shuffle(history_titles)
        random.seed(42)
        random.shuffle(history_catagories)
        random.seed(42)
        random.shuffle(impression_news_ids)
        random.seed(42)
        random.shuffle(impression_catagories)
        random.seed(42)
        random.shuffle(impression_titles)
        random.seed(42)
        random.shuffle(impression_labels)
        
        history_list = []
        for i in range(min(len(history_news_ids), 10)):
            history_list.append("\"" + history_news_ids[i] + "\"" + " in catagory " + history_catagories[i])

        history_str = ''
        for i in range(min(len(history_list),10)):
            if i == 0:
                history_str += history_list[i]
            else:
                history_str += ", " + history_list[i]
        print('user id:', user)
        print('history:', history_str)
                
        for i in range(len(impression_news_ids)):
            target_preference_str = "Yes." if impression_labels[i] == 1 else "No."
            target_news_str = "\"" + impression_titles[i] + "\"" + " in catagory " + impression_catagories[i]
            print('target news:', target_news_str)
            print('target preference:', target_preference_str)
            if impression_labels[i] == 1:
                positive_list.append({
                    "history_str": history_str,
                    "target_news_str": target_news_str,
                    "target_preference_str": target_preference_str,
                })
            else:
                print('target news:', target_news_str)
                print('target preference:', target_preference_str)
                negative_list.append({
                    "history_str": history_str,
                    "target_news_str": target_news_str,
                    "target_preference_str": target_preference_str,
                })

        for item in positive_list:
            Prompt_json.append({
                "instruction": "Given the user's history, identify whether the user will click the target news by answering \"Yes.\" or \"No.\".",
                "input": f"User History: {item['history_str']}\nWhether the user will click the target news {item['target_news_str']}?",
                "output": item['target_preference_str'],
            })
        
        random.seed(42)
        random.shuffle(negative_list)
            
        for item in negative_list:
            if split == 'train':
                prob = random.random()
                if prob > 1 / train_alpha:
                    continue
            elif split == 'valid':
                prob = random.random()
                if prob > 1 / val_alpha:
                    continue
            elif split == 'test':
                prob = random.random()
                if prob > 1 / test_alpha:
                    continue
            
            Prompt_json.append({
                "instruction": "Given the user's history, identify whether the user will click the target news by answering \"Yes.\" or \"No.\".",
                "input": f"User History: {item['history_str']}\nWhether the user will click the target news {item['target_news_str']}?",
                "output": item['target_preference_str'],
            })


    with open(output_json, 'w') as f:
        json.dump(Prompt_json, f, indent=4)


generate_json(train_user, tgt_train_json, split='train')
generate_json(valid_user, tgt_valid_json, split='valid')
generate_json(test_user, tgt_test_json, split='test')


"""
def generate_json(user_list, output_json):
    Prompt_json = []
    for user in user_list:
        history_news_ids = user_dict[user]['history_news_ids']
        history_titles = user_dict[user]['history_titles']
        history_catagories = user_dict[user]['history_catagories']
        history_abstracts = user_dict[user]['history_abstracts']
        impression_news_ids = user_dict[user]['impression_news_ids']
        impression_catagories = user_dict[user]['impression_catagories']
        impression_abstracts = user_dict[user]['impression_abstracts']
        impression_titles = user_dict[user]['impression_titles']
        impression_labels = user_dict[user]['impression_labels']

        random.seed(42)
        random.shuffle(history_news_ids)
        random.seed(42)
        random.shuffle(history_titles)
        random.seed(42)
        random.shuffle(history_catagories)
        random.seed(42)
        random.shuffle(impression_news_ids)
        random.seed(42)
        random.shuffle(impression_catagories)
        random.seed(42)
        random.shuffle(impression_titles)
        random.seed(42)
        random.shuffle(impression_labels)
        
        history_list = []
        for i in range(min(len(history_news_ids), 10)):
            history_list.append("\"" + history_news_ids[i] + "\"" + " in catagory " + history_catagories[i])

        history_str = ''
        for i in range(min(len(history_list),10)):
            if i == 0:
                history_str += history_list[i]
            else:
                history_str += ", " + history_list[i]

        for i in range(min(len(impression_news_ids), 10)):
            
            target_preference_str = "Yes." if impression_labels[i] == 1 else "No."
            target_news_str = "\"" + impression_titles[i] + "\"" + " in catagory " + impression_catagories[i]
            Prompt_json.append({
                "instruction": "Given the user's history, identify whether the user will click the target news by answering \"Yes.\" or \"No.\".",
                "input": f"User History: {history_str}\nWhether the user will click the target news {target_news_str}?",
                "output": target_preference_str,
            })
            
    with open(output_json, 'w') as f:
        json.dump(Prompt_json, f, indent=4)
        

generate_json(train_user, './data/MIND/train.json')
generate_json(valid_usser, './data/MIND/valid.json')
generate_json(test_user, './data/MIND/test.json')
"""