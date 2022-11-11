# download MAG KG from https://zenodo.org/record/3930398#.X9YvjnYzY5ll
import traceback

fos_names = set(map(lambda x: x.lower(),
                    ['Medicine', 'Biology', 'Chemistry', 'Computer Science', 'Engineering', 'Physics',
                     'Materials Science',
                     'Psychology', 'Mathematics', 'History', 'Sociology', 'Art', 'Political Science', 'Geography',
                     'Economics',
                     'Business', 'Geology', 'Philosophy', 'Environmental Science']))
file_name = 'FieldsOfStudy.nt'
fos_id_name_dict = {}
for line in open(file_name):
    splt = line.strip().split('>')
    assert len(splt) == 4
    if 'name' in splt[1]:
        temp = splt[2]
        fos = temp[:temp.index('^')].replace('\"', '').strip().lower()
        if fos in fos_names:
            fos_id = splt[0][splt[0].index('entity/') + 7:].strip()
            fos_id_name_dict[fos_id] = fos

# cat FieldsOfStudy.nt | grep 'level> "0"'
# <http://ma-graph.org/entity/95457728> <http://ma-graph.org/property/level> "0"^^<http://www.w3.org/2001/XMLSchema#integer> .
# <http://ma-graph.org/entity/127313418> <http://ma-graph.org/property/level> "0"^^<http://www.w3.org/2001/XMLSchema#integer> .
# <http://ma-graph.org/entity/162324750> <http://ma-graph.org/property/level> "0"^^<http://www.w3.org/2001/XMLSchema#integer> .
# <http://ma-graph.org/entity/205649164> <http://ma-graph.org/property/level> "0"^^<http://www.w3.org/2001/XMLSchema#integer> .
# <http://ma-graph.org/entity/185592680> <http://ma-graph.org/property/level> "0"^^<http://www.w3.org/2001/XMLSchema#integer> .
# <http://ma-graph.org/entity/138885662> <http://ma-graph.org/property/level> "0"^^<http://www.w3.org/2001/XMLSchema#integer> .
# <http://ma-graph.org/entity/144024400> <http://ma-graph.org/property/level> "0"^^<http://www.w3.org/2001/XMLSchema#integer> .
# <http://ma-graph.org/entity/192562407> <http://ma-graph.org/property/level> "0"^^<http://www.w3.org/2001/XMLSchema#integer> .
# <http://ma-graph.org/entity/33923547> <http://ma-graph.org/property/level> "0"^^<http://www.w3.org/2001/XMLSchema#integer> .
# <http://ma-graph.org/entity/86803240> <http://ma-graph.org/property/level> "0"^^<http://www.w3.org/2001/XMLSchema#integer> .
# <http://ma-graph.org/entity/41008148> <http://ma-graph.org/property/level> "0"^^<http://www.w3.org/2001/XMLSchema#integer> .
# <http://ma-graph.org/entity/17744445> <http://ma-graph.org/property/level> "0"^^<http://www.w3.org/2001/XMLSchema#integer> .
# <http://ma-graph.org/entity/127413603> <http://ma-graph.org/property/level> "0"^^<http://www.w3.org/2001/XMLSchema#integer> .
# <http://ma-graph.org/entity/15744967> <http://ma-graph.org/property/level> "0"^^<http://www.w3.org/2001/XMLSchema#integer> .
# <http://ma-graph.org/entity/39432304> <http://ma-graph.org/property/level> "0"^^<http://www.w3.org/2001/XMLSchema#integer> .
# <http://ma-graph.org/entity/144133560> <http://ma-graph.org/property/level> "0"^^<http://www.w3.org/2001/XMLSchema#integer> .
# <http://ma-graph.org/entity/121332964> <http://ma-graph.org/property/level> "0"^^<http://www.w3.org/2001/XMLSchema#integer> .
# <http://ma-graph.org/entity/71924100> <http://ma-graph.org/property/level> "0"^^<http://www.w3.org/2001/XMLSchema#integer> .
# <http://ma-graph.org/entity/142362112> <http://ma-graph.org/property/level> "0"^^<http://www.w3.org/2001/XMLSchema#integer> .

# assert len(fos_id_name_dict) == 19
# {'95457728': 'history', '127313418': 'geology', '162324750': 'economics', '205649164': 'geography', '185592680': 'chemistry',
#  '138885662': 'philosophy', '144024400': 'sociology', '192562407': 'materials science', '33923547': 'mathematics',
#  '86803240': 'biology', '41008148': 'computer science', '17744445': 'political science', '127413603': 'engineering',
#  '15744967': 'psychology', '39432304': 'environmental science', '144133560': 'business', '121332964': 'physics',
#  '71924100': 'medicine', '142362112': 'art'}

file_name1 = 'paper_fos_parsed_using_awk.txt'
with open('mag_paper_top_level_fos.tsv.1', 'w') as fw:
    for line in open(file_name1):
        splt = line.strip().split(' ')
        assert len(splt) == 2
        pid, fos_id = splt
        if fos_id in fos_id_name_dict:
            fw.write('\t'.join([pid, fos_id_name_dict[fos_id]]) + '\n')
        traceback.print_exc()

# file_name1 = 'mag_kg/PaperFieldsOfStudy.nt'
# with open('mag_paper_top_level_fos.tsv', 'w') as fw:
#     for line in open(file_name1):
#         try:
#             splt = line.strip().split('>')
#             assert len(splt) == 4
#             pid = splt[0][splt[0].index('entity/') + 7:].strip()
#             fos_id = splt[2][splt[2].index('entity/') + 7:].strip()
#             if fos_id in fos_id_name_dict:
#                 fw.write('\t'.join([pid, fos_id_name_dict[fos_id]]) + '\n')
#         except Exception as e:
#             traceback.print_exc()