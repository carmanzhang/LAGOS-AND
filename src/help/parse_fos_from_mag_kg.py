# download MAG KG from https://zenodo.org/record/3930398#.X9YvjnYzY5ll
import traceback

fos_names = set(map(lambda x: x.lower(),
                    ['Medicine', 'Biology', 'Chemistry', 'Computer Science', 'Engineering', 'Physics',
                     'Materials Science',
                     'Psychology', 'Mathematics', 'History', 'Sociology', 'Art', 'Political Science', 'Geography',
                     'Economics',
                     'Business', 'Geology', 'Philosophy', 'Environmental Science']))
file_name = 'mag_kg/FieldsOfStudy.nt'
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

# assert len(fos_id_name_dict) == 19


file_name1 = 'mag_kg/PaperFieldsOfStudy.nt'
with open('mag_paper_top_level_fos.tsv', 'w') as fw:
    for line in open(file_name1):
        try:
            splt = line.strip().split('>')
            assert len(splt) == 4
            pid = splt[0][splt[0].index('entity/') + 7:].strip()
            fos_id = splt[2][splt[2].index('entity/') + 7:].strip()
            if fos_id in fos_id_name_dict:
                fw.write('\t'.join([pid, fos_id_name_dict[fos_id]]) + '\n')
        except Exception as e:
            traceback.print_exc()
