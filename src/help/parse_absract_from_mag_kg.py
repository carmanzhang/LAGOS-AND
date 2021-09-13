# download MAG KG from https://zenodo.org/record/3930398#.X9YvjnYzY5ll
import traceback

temp_line = ''
closed = True

file_name = 'PaperAbstractsInvertedIndex.nt'
orcid_mag_matched_paper_id = set([line.strip() for line in open('orcid_mag_matched_paper_id.txt')])
all_need_matched = len(orcid_mag_matched_paper_id)
print(all_need_matched)
matched_cnt = 0
fw = open('orcid_mag_matched_paper_abstract.txt', 'w')
for line in open(file_name):
    line = line.strip()
    temp_line = temp_line + line + ' '
    if line.endswith('string> .'):
        # if 'string> .' in line:
        closed = True
    else:
        closed = False

    if closed:
        # print(temp_line)
        try:
            front = temp_line[:temp_line.index('terms/abstract>') + 16]
            back = temp_line[temp_line.index('terms/abstract>') + 16: temp_line.index('^^')]
            pid = front[front.index('entity/') + 7:front.index('> <')].strip()
            abstract = back
            # print(pid, abstract)
            if pid in orcid_mag_matched_paper_id:
                matched_cnt += 1
                if matched_cnt % 10000:
                    print('matched_cnt: ', matched_cnt, matched_cnt * 100.0 / all_need_matched)
                fw.write(pid + '\t' + abstract + '\n')
            # print(temp_line)
        except:
            traceback.print_exc()
        # print('-' * 100)
        temp_line = ''
