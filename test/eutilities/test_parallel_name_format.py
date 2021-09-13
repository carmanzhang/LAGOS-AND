from concurrent.futures import as_completed
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor

from eutilities.name_parser import NameProcessor
from myio.data_reader import DBReader, tcp_client

# def split_ltiple_biblio_authors(au):
#     if au is None or len(au) == 0:
#         return []
#     else:
#         splited_au = []
#         for pos, au_name in au:
#             # name_parts = derek73_nameparser(au_name)
#             name_parts = klauslippert_personnamenorm(au_name)
#             splited_au.append([pos, name_parts[0], name_parts[1], name_parts[2]])
#         return splited_au


# # df['splited_biblio_authors'] = df['biblio_authors'].apply(split_ltiple_biblio_authors)


df_s2 = DBReader.tcp_model_cached_read("XXXX",
                                       """select pid, biblio_authors, doi, orcid, orcid_names from and_ds.orcid_s2_paper_linkage limit 500000""",
                                       cached=False)
print(df_s2.shape)
# batch_insert_data = []
# for i, (pid, biblio_authors, doi, orcid, orcid_names) in df_s2.iterrows():
#     if i > 0 and i % 100000 == 0:
#         # trigger insert here
#         v = tcp_client.execute(
#             query="insert into and_ds.orcid_mag_s2_paper_linkage_split_name_varisou_algorithms VALUES",
#             params=batch_insert_data)
#         print('has inserted %d instances' % v)
#         batch_insert_data = []
#     batch_insert_data.append(
#         [pid, biblio_authors, split_ltiple_biblio_authors(biblio_authors), doi, orcid, orcid_names, 'S2',
#          'klauslippert'])
#
# if len(batch_insert_data) != 0:
#     v = tcp_client.execute(query="insert into and_ds.orcid_mag_s2_paper_linkage_split_name_varisou_algorithms VALUES",
#                            params=batch_insert_data, types_check=True)
#     print('has inserted the last %d instances' % v)
# print('inserted completed!')
#
# # delete this obj for saving RAM
# if df_s2 is not None:
#     del df_s2
#
# df_mag = DBReader.tcp_model_cached_read("XXXX",
#                                         """select pid, biblio_authors, doi, orcid, orcid_names from and_ds.orcid_mag_paper_linkage""",
#                                         cached=False)
# print(df_mag.shape)
# batch_insert_data = []
# for i, (pid, biblio_authors, doi, orcid, orcid_names) in df_mag.iterrows():
#     if i > 0 and i % 100000 == 0:
#         # trigger insert here
#         v = tcp_client.execute(
#             query="insert into and_ds.orcid_mag_s2_paper_linkage_split_name_varisou_algorithms VALUES",
#             params=batch_insert_data)
#         print('has inserted %d instances' % v)
#         batch_insert_data = []
#     batch_insert_data.append(
#         [str(pid), biblio_authors, split_ltiple_biblio_authors(biblio_authors), doi, orcid, orcid_names, 'MAG',
#          'klauslippert'])
#
# if len(batch_insert_data) != 0:
#     v = tcp_client.execute(query="insert into and_ds.orcid_mag_s2_paper_linkage_split_name_varisou_algorithms VALUES",
#                            params=batch_insert_data, types_check=True)
#     print('has inserted the last %d instances' % v)
# print('inserted completed!')


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=16) as executor:
        batch_insert_data = []
        for i, (pid, biblio_authors, doi, orcid, orcid_names) in df_s2.iterrows():
            if i > 0 and i % 1000 == 0:
                # trigger insert here
                biblio_authors_orcid_dict = {n[3]: n[1] for n in batch_insert_data}
                futures = {executor.submit(NameProcessor(), au): orcid for orcid, au in biblio_authors_orcid_dict.items()}
                results = {}
                for f in as_completed(futures):
                    results[futures[f]] = f.result()

                new_batch_insert_data = [n + [results[n[3]]] for n in batch_insert_data]
                # v = tcp_client.execute(
                #     query="insert into and_ds.orcid_mag_s2_paper_linkage_split_name_varisou_algorithms VALUES",
                #     params=new_batch_insert_data)
                # print('has inserted %d instances' % v)
                print('need insert')
                batch_insert_data = []
            batch_insert_data.append(
                [str(pid), biblio_authors, doi, orcid, orcid_names, 'MAG',
                 'klauslippert'])

    #     batch_insert_data.append(
#         [str(pid), biblio_authors, split_ltiple_biblio_authors(biblio_authors), doi, orcid, orcid_names, 'MAG',
#          'klauslippert'])
