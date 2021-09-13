ds_fn = './DBLP name disambiguation dataset'
lines = [line for line in open(ds_fn)]
header = lines[0]
print('headers: %s' % header)
lines = ''.join(lines[1:])

blocks = lines.split('\n\n')

print('num_block: %d' % len(blocks))
num_citation = 0
num_author_group = 0
for n in blocks:
    if '\n' not in n:
        # print(n)
        continue
    block_name = n[:n.index('\n')]
    fields = n[n.index('\n') + 1:].split('\n')
    # assert len(fields) % 9 ==0
    # num_names = int(len(fields) / 9)
    author_idx_arr = []

    for m in fields:
        try:
            if '\t' in m:
                author_idx = int(m[:m.index('\t')])
                author_idx_arr.append(author_idx)
        except Exception as e:
            # print(e)
            pass
    num_author_group += len(set(author_idx_arr))
    num_citation += len(author_idx_arr)

#
# num_citation = 0
# num_author_group = 0
# for n in ds:
#     fn = os.path.join(ds_dir, n)
#     author_idx_arr = []
#     for line in open(fn, encoding='iso8859-1'):
#         author_idx_citation_idx = line[:line.index(' ')]
#         num_citation += 1
#         author_idx, citation_idx = author_idx_citation_idx.split('_')
#         author_idx_arr.append(author_idx)
#     num_author_group += len(set(author_idx_arr))

print('num_author_group: %d' % num_author_group)
print('num_citation: %d' % num_citation)
