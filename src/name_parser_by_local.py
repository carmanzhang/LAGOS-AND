from eutilities.name_parser import derek73_nameparser, klauslippert_personnamenorm
from myio.data_reader import DBReader, tcp_client

name_parser_method = 'derek73'
# name_parser_method = 'klauslippert'

def split_ltiple_biblio_authors(au):
    if au is None or len(au) == 0:
        return []
    else:
        splited_au = []
        for pos, au_name in au:
            if name_parser_method == 'derek73':
                name_parts = derek73_nameparser(au_name)
            else:
                name_parts = klauslippert_personnamenorm(au_name)
            splited_au.append([pos, name_parts[0], name_parts[1], name_parts[2]])
        return splited_au


df_s2 = DBReader.tcp_model_cached_read("XXXX",
                                       """select pid, biblio_authors, doi, orcid, orcid_names from and_ds.orcid_s2_paper_linkage""",
                                       cached=False)
print(df_s2.shape)
batch_insert_data = []
for i, (pid, biblio_authors, doi, orcid, orcid_names) in df_s2.iterrows():
    if i > 0 and i % 100000 == 0:
        # trigger insert here
        v = tcp_client.execute(
            query="insert into and_ds.orcid_mag_s2_author_name_split_by_various_algorithms VALUES",
            params=batch_insert_data)
        print('has inserted %d instances' % v)
        batch_insert_data = []
    batch_insert_data.append(
        [pid, biblio_authors, doi, orcid, orcid_names, 'S2', name_parser_method,
         split_ltiple_biblio_authors(biblio_authors)])

if len(batch_insert_data) != 0:
    v = tcp_client.execute(query="insert into and_ds.orcid_mag_s2_author_name_split_by_various_algorithms VALUES",
                           params=batch_insert_data, types_check=True)
    print('has inserted the last %d instances' % v)
print('inserted completed!')

# delete this obj for saving RAM
if df_s2 is not None:
    del df_s2

df_mag = DBReader.tcp_model_cached_read("XXXX",
                                        """select pid, biblio_authors, doi, orcid, orcid_names from and_ds.orcid_mag_paper_linkage""",
                                        cached=False)
print(df_mag.shape)
batch_insert_data = []
for i, (pid, biblio_authors, doi, orcid, orcid_names) in df_mag.iterrows():
    if i > 0 and i % 100000 == 0:
        # trigger insert here
        v = tcp_client.execute(
            query="insert into and_ds.orcid_mag_s2_author_name_split_by_various_algorithms VALUES",
            params=batch_insert_data)
        print('has inserted %d instances' % v)
        batch_insert_data = []
    batch_insert_data.append(
        [str(pid), biblio_authors, doi, orcid, orcid_names, 'MAG', name_parser_method,
         split_ltiple_biblio_authors(biblio_authors)])

if len(batch_insert_data) != 0:
    v = tcp_client.execute(query="insert into and_ds.orcid_mag_s2_author_name_split_by_various_algorithms VALUES",
                           params=batch_insert_data, types_check=True)
    print('has inserted the last %d instances' % v)
print('inserted completed!')
