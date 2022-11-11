from tqdm import tqdm
from unidecode import unidecode

from mytookit.data_reader import DBReader

which_dataset = ['pairwise', 'block'][1]
applying_transliterating = False

sources, methods = ['S2', 'MAG'], ['derek73', 'joshfraser-NameParser']
# sources, methods = ['S2'], ['derek73']
# sources, methods = ['S2'], ['joshfraser-NameParser']
# sources, methods = ['MAG'], ['derek73']
# sources, methods = ['MAG'], ['joshfraser-NameParser']

field_map = {'derek73': 'derek73_top_lastname', 'joshfraser-NameParser': 'joshfraser_top_lastname'}


def clean_name(s):
    return unidecode(s).lower()


sql_template = '''
select matched_biblio_author_name, biblio_author_split_lastname, orcid_lastname, top_lastname
from (
         select pid,
                orcid,
                matched_biblio_author_name,
                lowerUTF8(matched_biblio_author_split_names[3]) as biblio_author_split_lastname
         from and_ds.orcid_mag_s2_author_name_split_by_various_algorithms_with_author_position
         where source = 'SOURCE' and method = 'METHOD'
         ) any
         join (select orcid,
                      orcid_lastname,
                      FIELD as top_lastname
               from and_ds.orcid_mag_s2_actual_author_name
               where source = 'SOURCE'
    ) using orcid;
'''


for source in sources:
    for method in methods:
        sql = sql_template.replace('SOURCE', source).replace('METHOD', method).replace('FIELD', field_map[method])
        print(sql)
        df = DBReader.tcp_model_cached_read("cached/XXXXX", sql, cached=False)
        print(df.shape)
        num_instances = df.shape[0]
        df['matched_biblio_author_name'] = df['matched_biblio_author_name'].apply(clean_name)
        df['biblio_author_split_lastname'] = df['biblio_author_split_lastname'].apply(clean_name)
        df['orcid_lastname'] = df['orcid_lastname'].apply(clean_name)
        df['top_lastname'] = df['top_lastname'].apply(clean_name)
        not_endwith_orcid_lastname, not_endwith_top_lastname, not_identicalwith_orcid_lastname, not_identicalwith_top_lastname = 0, 0, 0, 0
        for i, (biblio_author_name, biblio_author_lastname, orcid_lastname, top_lastname) in df.iterrows():
            if i % 100000 == 0:
                print(i * 1.0 / num_instances)
                # end with orcid lastname
            if not biblio_author_name.endswith(orcid_lastname):
                not_endwith_orcid_lastname += 1
            # end with top lastname
            if not biblio_author_name.endswith(top_lastname):
                not_endwith_top_lastname += 1
            # identical with orcid lastname
            if biblio_author_lastname != orcid_lastname:
                not_identicalwith_orcid_lastname += 1
            # identical with  top lastname
            if biblio_author_lastname != top_lastname:
                not_identicalwith_top_lastname += 1
        print(source, method, not_endwith_orcid_lastname, not_endwith_top_lastname, not_identicalwith_orcid_lastname,
              not_identicalwith_top_lastname)
        print(source, method, not_endwith_orcid_lastname * 1.0 / num_instances,
              not_endwith_top_lastname * 1.0 / num_instances, not_identicalwith_orcid_lastname * 1.0 / num_instances,
              not_identicalwith_top_lastname * 1.0 / num_instances)

# PubMed
df = DBReader.tcp_model_cached_read("cached/XXXXX", """
                                    select matched_biblio_author_lastname, orcid_lastname, top_lastname
                                    from and_ds.orcid_pubmed_author_linkage_with_author_position_with_topname;
                                    """, cached=False)
print(df.shape)
num_instances = df.shape[0]
df['matched_biblio_author_lastname'] = df['matched_biblio_author_lastname'].apply(clean_name)
df['orcid_lastname'] = df['orcid_lastname'].apply(clean_name)
df['top_lastname'] = df['top_lastname'].apply(clean_name)
not_identicalwith_orcid_lastname, not_identicalwith_top_lastname = 0, 0

for i, (biblio_author_lastname, orcid_lastname, top_lastname) in df.iterrows():
    if i % 100000 == 0:
        print(i * 1.0 / num_instances)
    if biblio_author_lastname != orcid_lastname:
        not_identicalwith_orcid_lastname += 1
    # identical with  top lastname
    if biblio_author_lastname != top_lastname:
        not_identicalwith_top_lastname += 1
print('PubMed', '-', not_identicalwith_orcid_lastname, not_identicalwith_top_lastname)
print(not_identicalwith_orcid_lastname * 1.0 / num_instances, not_identicalwith_top_lastname * 1.0 / num_instances)

# Our dataset
df = DBReader.tcp_model_cached_read("cached/XXXXX", """
select tupleElement(item, 2)                                as author_biblio_name,
       tupleElement(item, 3)                                as orcid_last_name,
       toInt64(tupleElement(arrayJoin(paper_orcid_lastname_bib_name) as item, 1) as pid) in
       (select arrayJoin(flatten(groupArray([pid1, pid2])))
        from and_ds.our_and_dataset_pairwise_gold_standard) as for_pairwise_dataset
from (
      select arrayJoin(
                 -- full_name_blocks: (num_work, orcid, same_orcidauthor_paper_positions, lastname_variations, same_orcidauthor_paper_repres)
                 -- same_orcidauthor_paper_repres: [(pid, author_position, orcid, orcid_names, matched_biblio_author, ethnic_seer, ethnea, genni, sex_mac, ssn_gender, pub_year, fos_arr), ..., ]
                     arrayZip(
                             arrayMap(x->arrayMap(y->y[1], x.3), full_name_blocks) as tmp_pids,
                             arrayMap(x->arrayMap(y->
                                                      y.4, x.5), full_name_blocks) as tmp_orcid_names,
                             arrayMap(x->arrayMap(y->
                                                      y.5, x.5), full_name_blocks) as tmp_bib_names
                         ))                                                   as paper_orcid_names,

             tupleElement(paper_orcid_names, 1)                               as pids,
             arrayMap(x->lowerUTF8(x[2]), tupleElement(paper_orcid_names, 2)) as orcid_last_names,
             tupleElement(paper_orcid_names, 3)                               as author_biblio_names,
             length(pids) = length(orcid_last_names)                          as is_valid,
             arrayZip(pids, author_biblio_names, orcid_last_names)            as paper_orcid_lastname_bib_name
      from and_ds.orcid_mag_matched_fullname_block)
;
""", cached=False)

if which_dataset == 'pairwise':
    df = df[df['for_pairwise_dataset'] == 1]

del df['for_pairwise_dataset']
print(df.shape)

if applying_transliterating:
    df['author_biblio_name'] = df['author_biblio_name'].apply(clean_name)
    df['orcid_last_name'] = df['orcid_last_name'].apply(clean_name)

not_endwith_orcid_lastname = 0
for i, (author_biblio_name, orcid_last_name) in tqdm(df.iterrows(), total=df.shape[0]):
    if not author_biblio_name.endswith(orcid_last_name):
        not_endwith_orcid_lastname += 1

print(which_dataset, '%s transliterating' % ('with' if applying_transliterating else 'without'), 'Our dataset', 'Endwith',
      not_endwith_orcid_lastname)
num_instances = df.shape[0]
print(not_endwith_orcid_lastname * 1.0 / num_instances)
# Note result
# pairwise without transliterating Our dataset Endwith 181348; 0.09719875824336029
# pairwise with transliterating Our dataset Endwith 122208; 0.06550094761124785

# block without transliterating Our dataset Endwith 722965; 0.09626437527720622
# block with transliterating Our dataset Endwith 485079; 0.06458933267183324

