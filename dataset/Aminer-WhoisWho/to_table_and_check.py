import json
from itertools import groupby

import pandas as pd

train_author = json.load(open('./train_author.json'))
# print(train_author)
author_name_aid_pid = []
for author_name in train_author.keys():
    for aid, pids in train_author[author_name].items():
        # print(author_name, aid, pids)
        # author_name_aid_pids.append([author_name, aid, '|'.join(pids)])
        for pid in pids:
            assert len(pid) > 0
            author_name_aid_pid.append([author_name, aid, pid])

print(len(author_name_aid_pid))
train_pub = json.load(open('./train_pub.json'))
pubs = {}
for pid in train_pub.keys():
    paper_info = train_pub[pid]
    id = paper_info['id'] if paper_info.get('id') is not None else ''
    authors = paper_info['authors'] if paper_info.get('authors') is not None else ''
    title = paper_info['title'] if paper_info.get('title') is not None else ''
    abstract = paper_info['abstract'].replace('\t', ' ').replace('\n', ' ') if paper_info.get(
        'abstract') is not None else ''
    keywords = paper_info['keywords'] if paper_info.get('keywords') is not None else ''
    venue = paper_info['venue'] if paper_info.get('venue') is not None else ''
    year = paper_info['year'] if paper_info.get('year') is not None else ''
    assert pid == id
    # print([pid, id, authors, title, abstract, keywords, venue, year])
    assert pubs.get(pid) == None
    pubs[pid] = [pid, authors, title, abstract, keywords, venue, year]

author_name_aid_pid_pub = [item + pubs.get(item[-1])[1:] if pubs.get(item[-1]) is not None else [''] * 6 for item in
                           author_name_aid_pid]


def convert_to_author_list(n):
    res = [m['name'].lower().replace('-', '').replace('.', '').replace(' ', '') for m in n[3]]
    return res


num_variation = 0
num_all = 0
for k, g in groupby(author_name_aid_pid_pub, lambda s: s[1]):
    for n in g:
        num_all += 1
        author_list = convert_to_author_list(n)
        split = n[0].split('_')
        au_name0 = split[-1] + split[0]
        au_name = n[0].replace('_', '')
        # last_name = n[0].split('_')[-1].strip()
        # TODO
        # if au_name not in author_list:
        if au_name not in author_list and au_name0 not in author_list:
            print(n[0], author_list)
            num_variation += 1

print(num_variation, num_all, num_variation / num_all)
# last name frequency
from collections import Counter

print(Counter([n[0].split('_')[-1] for n in author_name_aid_pid_pub]))

pd.DataFrame(author_name_aid_pid_pub,
             columns=['author_name', 'aid', 'pid', 'authors', 'title', 'abstract', 'keywords', 'venue', 'year']).to_csv(
    'train_author_pub.tsv', sep='\t',
    index=None)
