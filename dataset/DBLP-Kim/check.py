from collections import Counter

import pandas as pd

# author name: full name string extracted from DBLP
# unique author id: labels assigned manually by Dr. C. Lee Giles's team
# paper id: assigned by Dr. Jinseok Kim
# author list: names of authors in the byline of the paper
# year: publication year
# venue: conference or journal names
# title: stopwords removed and stemmed by the Porter's stemmer

df = pd.read_csv('./DBLP_labeled_data.txt', sep='\t',
                 names=['author name', 'unique author id', 'paper id', 'author list', 'year', 'venue', 'title', 'null'], index_col=None)
print(df.head())


author_names = df['unique author id'].apply(lambda x: ' '.join(x.split('-')[:-1])).values

counter = Counter(author_names)
print(counter)

print('num_block: %d' % len(set(author_names)))

num_citation = len(set(df['paper id'].values))
num_author_group = len(set(df['unique author id'].values))
print('num_author_group: %d' % num_author_group)
print('num_citation: %d' % num_citation)
