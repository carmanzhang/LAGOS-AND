import xmltodict

data = open('SCAD-zbMATH/scad-zbmath-01-open-access.xml').read()
data = xmltodict.parse(data)
data_instance = []
for n in data['publications']['publication']:
    title = n['title']
    authors = n['authors']['author']
    if type(authors) != list:
        authors = [authors]
    # print(authors)
    for au in authors:
        name, shortname, id = au['@name'], au['@shortname'], au['@id'],
        print(name, shortname, id, title)
        last_name_first_initial = shortname
        data_instance.append([name, last_name_first_initial, id, title])

num_blocks = len(set([n[1] for n in data_instance]))
num_author_group = len(set([n[2] for n in data_instance]))
num_citation = len(data_instance)

print('num_block: %d' % num_blocks)
print('num_author_group: %d' % num_author_group)
print('num_citation: %d' % num_citation)
