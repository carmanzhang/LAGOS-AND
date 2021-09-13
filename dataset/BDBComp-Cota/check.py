from collections import Counter

import pandas as pd

# font_bdbcomp.txt and title_bdbcomp.txt.
#
# The records of the file font_bdbcomp.txt are in the format of:
# citationId<>clusterId_sequential<>coauthor:coauthor:...:coauthor<>publicationVenueTitle<>author
#
# The records of the file title_bdbcomp.txt are in the format of:
# citationId<>workTitle

name = pd.read_csv('./bdbcomp/font_bdbcomp.txt', sep='<>',
                   names=['citationId', 'clusterId_sequential', 'authorList', 'publicationVenueTitle', 'author',
                          'null'])
print(name.head())
pub = pd.read_csv('./bdbcomp/title_bdbcomp.txt', sep='<>', names=['citationId', 'workTitle'], error_bad_lines=None)
print(pub.head())
print(pub.shape)

author_names = name['author'].apply(lambda x: '_'.join([
    x.split(' ')[-1],  # last name
    x.split(' ')[0][0]  # first initial
])).values

counter = Counter(author_names)
print(counter)


author_group_idx = [int(n) for n in set(name['clusterId_sequential'].apply(lambda x: str(x)[:str(x).index('_')]).values)]
for n in range(214):
    if n not in author_group_idx:
        print(n)
print(sorted(author_group_idx))
num_author_group = len(author_group_idx)
print('num_block: %d' % len(set(author_names)))
num_citation = len(set(name['citationId'].values))
print('num_author_group: %d' % num_author_group)
print('num_citation: %d' % num_citation)
