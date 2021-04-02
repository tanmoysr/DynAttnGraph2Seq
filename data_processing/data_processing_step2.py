import json
import numpy as np
import itertools
from pprint import pprint
from datetime import datetime
import math
import os
from os import walk

stage_list = [
    'Dx',
    'TURBT',
    'BCG',
    'Chemotherapy',
    'Surgery'
]

subforums = ['3-newly-diagnosed', '8-non-invasive-superficial-bladder-cancer-questions-and-comments',
             '5-muscle-invasive-bladder-cancer-questions-and-comments',
             '7-metastatic-bladder-cancer', '10-women-and-bladder-cancer', '6-men-and-bladder-cancer',
             '20-caregivers-questions-and-comments',
             '31-articles-of-interest', '17-chit-chat']

dyngraph2seq = {}

sf2id = {}
id = 0;
for subforum in subforums:
    sf2id[subforum] = id
    id += 1

g_ids = {}
for i in range(0, id):
    g_ids[str(i)] = i

user_data_path = 'valid_users/'
_, _, filenames = next(walk(user_data_path))

for user_file in filenames:
    user = os.path.splitext(user_file)[0]
    print(user)

    with open(user_data_path + user_file) as f:
        user_info = json.load(f)
        posts = user_info["posts"]
        user_seq = user_info['signature']

        post_count = len(posts)

        info = {}
        g_adj = {}
        g_adjseq = []

    # start_date=datetime.strptime('2010-1-1', '%Y-%m-%d');
    # end_date=datetime.strptime('2018-1-1', '%Y-%m-%d');
    # 2007 1 - 2019 12
    for year in range(2007, 2020):
        for month in range(1, 13, 2):
            last_fid = 0;
            adj = np.zeros((id, id))

            forum_date = {}
            for post in posts:
                fid = sf2id[post['subforum']] + 1
                date = post['date'].split('T')
                date = datetime.strptime(date[0], '%Y-%m-%d')

                if date.year == year and (date.month == month or date.month == month+1):
                    ## gather visit time for each forum
                    if fid not in forum_date:
                        forum_date[fid] = []
                    forum_date[fid].append(date)

            ## compute median time of vist for each forum
            median_forum = {}
            for fid, tlist in forum_date.items():
                tlist = sorted(tlist)
                median = tlist[len(tlist) // 2]
                median_forum[median] = fid

            for date, fid in sorted(median_forum.items()):
                if last_fid > 0:
                    adj[last_fid - 1][fid - 1] = 1
                    print([str(last_fid-1) + '->' + str(fid-1)])
                last_fid = fid

            for i in range(0, id):
                if str(i) not in g_adj:
                    g_adj[str(i)] = []
                # accumulative
                g_adj[str(i)] = list(set(g_adj[str(i)] + (np.where(adj[i][:] > 0)[0]).tolist()))
            # raw
            # g_adj[str(i)]=(np.where(adj[i][:]>0)[0]).tolist()
            # print(g_adj[str(i)])

            g_adjseq.append(g_adj.copy())

    info['g_ids'] = g_ids
    info['g_ids_features'] = []  # g_ids_features
    info['g_adjseq'] = g_adjseq
    info['user'] = user
    info['seq'] = user_seq
    dyngraph2seq[user] = info

print('train size:' + str(len(dyngraph2seq) * 7 // 10))

train = dict(list(dyngraph2seq.items())[:len(dyngraph2seq) * 7 // 10])
dev = dict(list(dyngraph2seq.items())[len(dyngraph2seq) * 7 // 10:len(dyngraph2seq) * 8 // 10])
test = dict(list(dyngraph2seq.items())[len(dyngraph2seq) * 8 // 10:])

# save training
directory = 'dyngraph2seq/train.data'
with open(directory, 'w') as outfile:
    for user, seq in train.items():
        outfile.write(json.dumps(seq) + "\n")

# save dev
directory = 'dyngraph2seq/dev.data'
with open(directory, 'w') as outfile:
    for user, seq in dev.items():
        outfile.write(json.dumps(seq) + "\n")

# save dev
directory = 'dyngraph2seq/test.data'
with open(directory, 'w') as outfile:
    for user, seq in test.items():
        outfile.write(json.dumps(seq) + "\n")

pprint('done!')
