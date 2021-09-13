import json

d1 = json.load(open('na-data-kdd18/data/global/name_to_pubs_train_500.json'))
print(len(d1), d1.keys())

d2 = json.load(open('na-data-kdd18/data/global/name_to_pubs_test_100.json'))
print(len(d2), d2.keys())

d = {}
d.update(d1)
d.update(d2)
print(len(d), d.keys())
assert len(d) == len(d1) + len(d2)

num_blocks = len(d)
# for k,v in d.items():
#     print(k, v)
num_author_group = sum([len(list(v.keys())) for k, v in d.items()])
citation_with_author_order = sum([sum([b for a, b in v.items()], []) for k, v in d.items()], [])
citations = [n[:n.index('-')] for n in citation_with_author_order]
num_citation = len(set(citations))

print('num_block: %d' % num_blocks)
print('num_author_group: %d' % num_author_group)
print('num_citation: %d' % num_citation)

block_author_papers = sum([[(k, n) for n in sum([b for _, b in v.items()], [])] for k, v in d.items()], [])
print(block_author_papers[:10])

# read bibliographic author name
pubs = json.load(open('na-data-kdd18/data/global/pubs_raw.json'))
pubs_author_name = []
for k, v in pubs.items():
    if 'authors' in v:
        for i, a in enumerate(v['authors']):
            pubs_author_name.append((k + '-' + str(i), a['name'].lower()))
print(len(pubs_author_name), pubs_author_name[:10])
pubs_author_name = dict(pubs_author_name)
print(len(pubs_author_name))

matched_names = []
from collections import Counter

lastnames = []
for a, b in block_author_papers:
    biblio_name = pubs_author_name.get(b)
    if biblio_name is None:
        continue
    lastname = a.split('_')[-1]
    lastnames.append(lastname)
    a = a.replace('_', ' ').replace(' ', '')
    biblio_name = biblio_name.replace('.', '').replace('-', ' ').replace(' ', '')
    # have verified there is no case of full name variation in this dataset
    if a != biblio_name:
        print(a, biblio_name)

    matched_names.append([a, biblio_name])

print(Counter(lastnames))
# convert json to csv
res = []
pubs_author_name = []
for k, v in pubs.items():
    authors, title, venue, year, abstract = v.get('authors'), v.get('title'), v.get('venue'), v.get('year'), v.get(
        'abstract')
    if abstract is not None:
        abstract = abstract.replace('\t', ' ').replace('\n', '')
    res.append([authors, title, venue, year, abstract])

import pandas as pd

pd.DataFrame(res, columns=['authors', 'title', 'venue', 'year', 'abstract']).to_csv('aminer-zhang-csv.csv', sep='\t',
                                                                                    index=None)
