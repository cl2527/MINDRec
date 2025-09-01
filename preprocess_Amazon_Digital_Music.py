import pandas as pd
import json
from tqdm import tqdm
import random
item_meta_file = './raw_data/Amazon_reviews/meta_Digital_Music.json'
ratings_file = './raw_data/Amazon_reviews/Digital_Music.csv'

item_meta = pd.read_json(item_meta_file, lines=True)
ratings = pd.read_csv(ratings_file)

user_dict = {}
item_dict = {}

for index, row in tqdm(item_meta.iterrows()):
    if not item_dict.__contains__(row['asin']):
        item_dict[row['asin']] = {}
        item_dict[row['asin']]['title'] = row['title']
        item_dict[row['asin']]['price'] = row['price']
        item_dict[row['asin']]['brand'] = row['brand']

#get the titles of each column

for index, row in tqdm(ratings.iterrows()):
    asin = row.iloc[0]
    user_id = row.iloc[1]
    rating = row.iloc[2]
    time_stamp = row.iloc[3]
    if asin not in item_dict.keys():
        continue
    if not user_dict.__contains__(user_id):
        user_dict[user_id] = {
            'titles': [],
            'ratings': [],
            'prices': [],
            'brands': [],
        }
    user_dict[user_id]['titles'].append(item_dict[asin]['title'])
    user_dict[user_id]['ratings'].append(rating)
    user_dict[user_id]['prices'].append(item_dict[asin]['price'])
    user_dict[user_id]['brands'].append(item_dict[asin]['brand'])


new_user_dict = {}
for key in user_dict.keys():
    if len(user_dict[key]['titles']) > 3:
        new_user_dict[key] = user_dict[key]

print('length of new user dict: ', len(new_user_dict))

#print the  first user in new_user_dict
print(new_user_dict)
print(new_user_dict.__len__())
print(user_dict.__len__())
"""
def generate_json(user_list, output_json):
    nrows = []
    for user in user_list:
        titles = user_dict[user]['titles']
        ratings = [int(_ > 3) for _ in user_dict[user]['ratings']]
        random.seed(42)
        random.shuffle(titles)
        random.seed(42)
        random.shuffle(ratings)
        nrows.append({
            'user': user,
            'history_titles': titles[:-1][:10],
            'history_ratings': ratings[:-1][:10],
            'title': titles[-1],
            'rating': ratings[-1],
        })
"""