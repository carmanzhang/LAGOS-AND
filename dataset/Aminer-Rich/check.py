import os

base_path = './rich-author-disambiguation-data/experimental-results'
blocks = [n for n in os.listdir(base_path)]

num_blocks = 0
num_author_group = 0
citations = []
for n in blocks:
    path = os.path.join(base_path, n)
    if not os.path.isfile(path) or 'classify' not in n:
        continue

    num_blocks += 1
    for line in open(path):
        if ':' not in line:
            continue
        id = line[:line.index(':')]
        papers = [m.strip() for m in line[line.index(':') + 1:].split(' ') if len(m.strip()) > 0]
        print(id, papers)
        num_author_group += 1
        citations.extend(papers)

num_citation = len(set(citations))
print('num_block: %d' % num_blocks)
print('num_author_group: %d' % num_author_group)
print('num_citation: %d' % num_citation)
