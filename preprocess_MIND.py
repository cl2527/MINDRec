import pandas as pd
behaviors = pd.read_csv('./raw_data/mind/MINDsmall_train/behaviors.tsv', sep='\t', header=None, names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
news = pd.read_csv('./raw_data/mind/MINDsmall_train/news.tsv', sep='\t', header=None, names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])

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
            'history_titles': [],
            'history_catagory': [],
            'history_abstract': [],
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
        user_dict[userid]['history_catagory'].append(news_catagory)
        user_dict[userid]['history_titles'].append(news_title)
        user_dict[userid]['history_abstract'].append(news_abstract)
    if row['impressions'] == 'NULL':
        continue
    impressions = row['impressions'].split(' ')
    for impression in impressions:
        impression_news_id, label = impression.split('-')
        news_idx = news_dict[impression_news_id]
        news_title = news.iloc[news_idx]['title']
        news_catagory = news.iloc[news_idx]['category']
        news_abstract = news.iloc[news_idx]['abstract']
        user_dict[userid]['impression_titles'].append(news_title)
        user_dict[userid]['impression_labels'].append(int(label))
        user_dict[userid]['impression_catagory'].append(news_catagory)
        user_dict[userid]['impression_abstract'].append(news_abstract)
        
    
new_user_dict = {}
for key in user_dict.keys():
    if len(user_dict[key]['historys'])  <= 5:
        pass
    else:
        new_user_dict[key] = user_dict[key]
        print(user_dict[key])

import random
import json
user_list = list(new_user_dict.keys())
random.shuffle(user_list)
train_user = user_list[:int(len(user_list) * 0.8)]
valid_usser = user_list[int(len(user_list) * 0.8):int(len(user_list) * 0.9)]
test_user = user_list[int(len(user_list) * 0.9):]

def generate_json(user_list, output_json):
    