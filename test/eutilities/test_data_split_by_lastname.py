import pandas as pd
from sklearn.model_selection import KFold, GroupShuffleSplit

meta_columns = ['pm_ao1', 'pm_ao2', 'same_author', 'source', 'shared_lastname', 'lastname_hash_partition_for_split']

df = pd.read_csv('../../src/feature/pubmed_inner_outer_feature.tsv', sep='\t')
X = df[['pm_ao1', 'pm_ao2', 'same_author', 'source', 'shared_lastname', 'lastname_hash_partition_for_split']].values
Y = df['shared_lastname'].values
group = df['lastname_hash_partition_for_split'].values

kf = GroupShuffleSplit(n_splits=3)
# kf = KFold(n_splits=3, shuffle=True)
indx_split = kf.split(X, groups=group)

for train_index, test_index in indx_split:
    train_X, train_y = X[train_index], Y[train_index]
    test_X, test_y = X[test_index], Y[test_index]
    train_side = [n[4] for n in train_X]
    test_side = [n[4] for n in test_X]
    intersection = set(train_side).intersection(set(test_side))
    print(len(intersection), intersection)