import pandas as pd


split = "val"

if split == "train":
    behaviors = pd.read_csv('./raw_data/mind/MINDsmall_train/behaviors.tsv', sep='\t', header=None, names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
    news = pd.read_csv('./raw_data/mind/MINDsmall_train/news.tsv', sep='\t', header=None, names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
else:
    behaviors = pd.read_csv('./raw_data/mind/MINDsmall_dev/behaviors.tsv', sep='\t', header=None, names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
    news = pd.read_csv('./raw_data/mind/MINDsmall_dev/news.tsv', sep='\t', header=None, names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])

real_total_users = len(behaviors['user_id'].unique())

tgt_folder_full = './data/MIND_multi/'

tgt_train_json = tgt_folder_full + 'train.json'
tgt_valid_json = tgt_folder_full + 'valid.json'

if split == "train":
    tgt_json = tgt_train_json
else:
    tgt_json = tgt_valid_json

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
for index, row in tqdm(behaviors.iloc[:int(len(behaviors)/3)].iterrows()):
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
    if 5 < len(user_dict[key]['history_news_ids']) <= 20 and 5 < len(user_dict[key]['impression_news_ids']) <= 20:
        new_user_dict[key] = user_dict[key]

import random
import json



def generate_json(user_list, output_json):
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

        history_list = []
        for i in range(len(history_news_ids)):
            #history_list.append("\"" + history_titles[i] + "\"" + " in catagory " + history_catagories[i])
            history_list.append("\"" + history_titles[i] + "\"")
        history_str = ''
        for i in range(len(history_list)):
            if i == 0:
                history_str += history_list[i]
            else:
                history_str += ", " + history_list[i]

        impression_list = []
        for i in range(len(impression_news_ids)):
            impression_list.append("\"" + impression_titles[i] + "\"" )
        impression_str = ''
        for i in range(len(impression_list)):
            if i == 0:
                impression_str += impression_list[i]
            else:
                impression_str += ", " + impression_list[i]
        
        label_list = []
        for i in range(len(impression_news_ids)):
            label_list.append('Yes' if impression_labels[i] == 1 else 'No')
        label_str = ''
        for i in range(len(label_list)):
            if i == 0:
                label_str += label_list[i]
            else:
                label_str += ", " + label_list[i]

        Prompt_json.append({
            "instruction": "Given the user's news click history in chronological order, identify whether the user will click each of the target news by answering \"Yes\" or \"No\".",
            "input": f"User History: {history_str}\nWhether the user will click the target news {impression_str}?",
            "output": label_str,
        })

    with open(output_json, 'w') as f:
        json.dump(Prompt_json, f, indent=4)

generate_json(new_user_dict, tgt_json)
