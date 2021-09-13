import pandas as pd

df = pd.read_csv('./1900_pairs_train_test.csv', sep=';')
last_name_variation_cases = 0
for i, (ln1, ln2) in df[['Last_name1', 'Last_name2']].iterrows():
    if str(ln1).lower() != str(ln2).lower():
        last_name_variation_cases += 1
        print(ln1, ln2)

print('last_name_variation_cases: %d' % last_name_variation_cases)

num_paired_records = df.shape[0]
print('num_paired_records: %d' % num_paired_records)

pmid_arr = list(df['PMID1'].values) + list(df['PMID2'].values)
num_citation = len(set(pmid_arr))
# print('num_author_group: %d' % num_author_group)
print('num_citation: %d' % num_citation)

