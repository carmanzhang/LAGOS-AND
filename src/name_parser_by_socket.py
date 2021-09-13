import json
import socket
import traceback

from myio.data_reader import DBReader, tcp_client


def recvall(sock_cli):
    BUFF_SIZE = 4096  # 4 KiB
    data = b''
    while True:
        part = sock_cli.recv(BUFF_SIZE)
        data += part
        if len(part) < BUFF_SIZE:
            # either 0 or end of data
            break
    return data


def insert_batch_data(batch_insert_data):
    paper_names = [[m[1] for m in n[1]] for n in batch_insert_data]
    json_str = json.dumps({'names': paper_names}, ensure_ascii=False)

    client = socket.socket()
    client.connect(('localhost', 38081))
    # print('connect successfully')

    client.send(json_str.encode("utf-8"))
    res = recvall(client)
    process_names = json.loads(res.decode())
    client.close()
    print(len(process_names), len(batch_insert_data))
    assert len(process_names) == len(batch_insert_data)
    batch_insert_data = [n + [process_names[i]] for i, n in enumerate(batch_insert_data)]

    v = tcp_client.execute(
        query="insert into and_ds.orcid_mag_s2_author_name_split_by_various_algorithms VALUES",
        params=batch_insert_data)
    print('has inserted %d instances' % v)


df_s2 = DBReader.tcp_model_cached_read("XXXX",
                                       """select pid, biblio_authors, doi, orcid, orcid_names from and_ds.orcid_s2_paper_linkage""",
                                       cached=False)
print(df_s2.shape)
batch_insert_data = []
for i, (pid, biblio_authors, doi, orcid, orcid_names) in df_s2.iterrows():
    if i > 0 and i % 10000 == 0:
        # trigger inserting data
        insert_batch_data(batch_insert_data)
        batch_insert_data = []
    batch_insert_data.append([pid, biblio_authors, doi, orcid, orcid_names, 'S2', 'joshfraser-NameParser'])

if len(batch_insert_data) != 0:
    insert_batch_data(batch_insert_data)
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
    if i > 0 and i % 10000 == 0:
        # trigger insert here
        try:
            insert_batch_data(batch_insert_data)
        except:
            traceback.print_exc()

        batch_insert_data = []
    batch_insert_data.append([str(pid), biblio_authors, doi, orcid, orcid_names, 'MAG', 'joshfraser-NameParser'])

if len(batch_insert_data) != 0:
    insert_batch_data(batch_insert_data)
print('inserted completed!')
