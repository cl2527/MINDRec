import pandas as pd


split = "test"

if split == "train":
    behaviors = pd.read_csv('./raw_data/mind/MINDsmall_train/behaviors.tsv', sep='\t', header=None, names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
    news = pd.read_csv('./raw_data/mind/MINDsmall_train/news.tsv', sep='\t', header=None, names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
else:
    behaviors = pd.read_csv('./raw_data/mind/MINDsmall_dev/behaviors.tsv', sep='\t', header=None, names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
    news = pd.read_csv('./raw_data/mind/MINDsmall_dev/news.tsv', sep='\t', header=None, names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])

train_alpha = 10 # train set negative samples undersampling rate
val_alpha = 1 # valid set negative samples undersampling rate
test_alpha = 1# test set negative samples undersampling rate0

num_history = 10

hist_len_cap = 30 # user undersampling rate

real_total_users = len(behaviors['user_id'].unique())

tgt_fld_name = 'tra_NU'+str(train_alpha)+'_val_NU'+str(val_alpha)+'_te_NU'+str(test_alpha)+'_histLen_'+str(hist_len_cap)
tgt_folder_full = './data/MIND_30cap/' + tgt_fld_name + '/'


if split == "train":
    tgt_json = tgt_folder_full + 'train.json'
elif split == "valid":
    tgt_json = tgt_folder_full + 'valid.json'
else:
    tgt_json = tgt_folder_full + 'test.json'

import os
if not os.path.exists(tgt_folder_full):
    os.makedirs(tgt_folder_full)

news_dict = {}

from tqdm import tqdm
user_dict = {}
news_id = {}

for index, row in tqdm(news.iterrows()):
    news_dict[row['news_id']] = index
#iterate only first 1/3 of the rows below

if split == "train":
    indices = list(range(0, int(len(behaviors)//10)))
elif split == "valid":
    indices = list(range(0, int(len(behaviors)//50)))
else:
    indices = list(range(int(len(behaviors)//50), int(len(behaviors)//50)*2))
    
for index, row in tqdm(behaviors.iloc[indices].iterrows()):
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
    hist = row.get("history", "")
    if not isinstance(hist, str):
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
    imp = row.get("impressions", "")
    if not isinstance(imp, str):
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
    if 5 < len(user_dict[key]['history_news_ids']) <= hist_len_cap and 5 < len(user_dict[key]['impression_news_ids']):
        new_user_dict[key] = user_dict[key]

import random
import json

total_users = len(new_user_dict)
print('total users:', total_users)



def generate_json(user_list, output_json, split = 'train'):
    Prompt_json = []

    for user in tqdm(user_list):
        positive_list = []
        negative_list = []
        history_news_ids = user_dict[user]['history_news_ids']
        history_titles = user_dict[user]['history_titles']
        history_catagories = user_dict[user]['history_catagories']
        history_abstracts = user_dict[user]['history_abstracts']
        impression_news_ids = user_dict[user]['impression_news_ids']
        impression_catagories = user_dict[user]['impression_catagories']
        impression_abstracts = user_dict[user]['impression_abstracts']
        impression_titles = user_dict[user]['impression_titles']
        impression_labels = user_dict[user]['impression_labels']

        rng = random.Random(42)

        # History
        """
        hist = list(zip(history_news_ids, history_titles, history_catagories, history_abstracts))
        rng.shuffle(hist)
        history_news_ids, history_titles, history_catagories, history_abstracts = map(list, zip(*hist))
        """
        
        # Impressions
        impr = list(zip(impression_news_ids, impression_catagories, impression_abstracts, impression_titles, impression_labels))
        rng.shuffle(impr)
        (impression_news_ids, impression_catagories, impression_abstracts,
        impression_titles, impression_labels) = map(list, zip(*impr))
        
        history_list = []
        for i in range(len(history_news_ids)):
            history_list.append("\"" + history_titles[i] + "\"" + " in " + history_catagories[i])

        history_str = ''
        for i in range(len(history_list)):
            if i == 0:
                history_str += history_list[i]
            else:
                history_str += ", " + history_list[i]
        #print('user id:', user)
        #print('history:', history_str)
        for i in range(len(impression_news_ids)):
            target_preference_str = "Yes." if impression_labels[i] == 1 else "No."
            target_news_str = "\"" + impression_titles[i] + "\"" + " in " + impression_catagories[i]

            if impression_labels[i] == 1:
                positive_list.append({
                    "history_str": history_str,
                    "target_news_str": target_news_str,
                    "target_preference_str": target_preference_str,
                })
            else:
                negative_list.append({
                    "history_str": history_str,
                    "target_news_str": target_news_str,
                    "target_preference_str": target_preference_str,
                })

        for item in positive_list:
            Prompt_json.append({
                "instruction": "Given the user's click history in chronical order, identify whether the user will click the target news by answering \"Yes.\" or \"No.\".",
                "input": f"User's click History: {item['history_str']}\nWhether the user will click the target news in next impression {item['target_news_str']}?",
                "output": item['target_preference_str'],
                "user_id": user,
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
                "instruction": "Given the user's click history in chronical order, identify whether the user will click the target news by answering \"Yes.\" or \"No.\".",
                "input": f"User's click History: {item['history_str']}\nWhether the user will click the target news in next impression {item['target_news_str']}?",
                "output": item['target_preference_str'],
                "user_id": user,
            })


    with open(output_json, 'w') as f:
        json.dump(Prompt_json, f, indent=4)

generate_json(new_user_dict, tgt_json, split=split)

