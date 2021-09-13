from collections import Counter

import pandas as pd

names = pd.read_csv('Dataset/sigir/Gold Dataset/disambiguatedNames.csv', sep=';', encoding='iso8859-1')
print(names.head())
pubs = pd.read_csv('Dataset/sigir/Gold Dataset/goldstandardPublications.csv', sep=';', encoding='iso8859-1')
print(pubs.head())

author_names = names['name'].apply(lambda x: ' '.join(x.split(' ')[:-1])).values

counter = Counter(author_names)
print(counter)

print('num_block: %d' % len(set(author_names)))
num_citation = pubs.shape[0]
num_author_group = len(set(names['fk_authorid'].values))
print('num_author_group: %d' % num_author_group)
print('num_citation: %d' % num_citation)
