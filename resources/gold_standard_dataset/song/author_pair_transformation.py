all = []
surnames = []
for line in open('and_corpus.txt', 'r'):
    if line.startswith('IDX'):
        continue
    IDX, Pmid, First_Author_Last_Name, First_Author_Initials = line.split('\t')
    all.append([IDX, Pmid, First_Author_Last_Name, First_Author_Initials])
    surnames.append(First_Author_Last_Name)
surnames = list(set(surnames))

print(len(surnames), surnames)

song_dataset_author_pair = []
for surname in surnames:
    same_surname_items = [item for item in all if item[2] == str(surname)]
    if same_surname_items is None or len(same_surname_items) == 0:
        continue

    l = len(same_surname_items)
    for i in range(l - 1):
        a1 = same_surname_items[i]
        for j in range(i + 1, l):
            a2 = same_surname_items[j]
            song_dataset_author_pair.append([a1[1], a2[1], str(1 if a1[0] == a2[0] else 0)])
            # if a1[0] == a2[0]:
            #     print('1', a1[1], a2[1])
            # else:
            #     print('0', a1[1], a2[1])
with open('song_dataset_author_pair.tsv', 'w') as fw:
    for item in song_dataset_author_pair:
        fw.write('\t'.join(item) + '\n')
