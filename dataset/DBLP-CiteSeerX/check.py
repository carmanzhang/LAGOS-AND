import os

ds_dir = './nameset_author-disamb'
ds = [n for n in os.listdir(ds_dir)]
print('num_block_size: %d' % len(ds))

num_citation = 0
num_author_group = 0
for n in ds:
    fn = os.path.join(ds_dir, n)
    author_idx_arr = []
    for line in open(fn, encoding='iso8859-1'):
        author_idx_citation_idx = line[:line.index(' ')]
        num_citation += 1
        author_idx, citation_idx = author_idx_citation_idx.split('_')
        author_idx_arr.append(author_idx)
    num_author_group += len(set(author_idx_arr))

print('num_author_group_size: %d' % num_author_group)
print('num_citation: %d' % num_citation)
