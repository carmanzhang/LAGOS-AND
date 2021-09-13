import os

base_path = './rexa_author_coref/rexa'
blocks = [n for n in os.listdir(base_path)]

num_blocks = 0
num_author_group = 0
num_citation = 0
for n in blocks:
    path = os.path.join(base_path, n)
    if os.path.isfile(path):
        continue
    num_blocks += 1
    block_authors = [n for n in os.listdir(path)]
    num_author_group += len(block_authors)
    for m in block_authors:
        path1 = os.path.join(path, m)
        if os.path.isfile(path1):
            continue
        citations = [n for n in os.listdir(path1)]
        num_citation += len(citations)

print('num_block: %d' % num_blocks)
print('num_author_group: %d' % num_author_group)
print('num_citation: %d' % num_citation)
