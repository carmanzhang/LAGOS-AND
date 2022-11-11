import os

import joblib
import numpy as np
import torch
from mytookit.data_reader import DBReader
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from myconfig import device, cached_dir

sql_block = r'''
select block_fullname                                                                     as block_name,
       arrayMap(x->x[1],
                arraySort(x->x[1], groupArray([pid_ao, author_group_orcid, toString(mag_author_id)])) as tmp) as pid_aos,
       arrayMap(x->x[2], tmp)                                                             as ground_truths,
       arrayMap(x->x[3], tmp)                                                             as mag_preds
from (select block_fullname,
             author_group_orcid,
             -- Note has verified all mag_author_id is successfully matched
             toString(aid)                                         as mag_author_id,
             concat(toString(pid), '_', toString(author_position)) as pid_ao
      from and_ds.our_and_dataset_block any
               left join (
          select pid, aid, author_position
          from (select PaperId as pid, AuthorId as aid, toInt64(AuthorSequenceNumber) as author_position
                from mag.paper_author_affiliation) any
                   inner join and_ds.our_and_dataset_block using pid, author_position
          ) using pid, author_position)
group by block_name
having xxHash32(block_name) %% 10=%d  
order by length(pid_aos) desc;
;'''

sql_metadata = r'''
select concat(toString(pid), '_', toString(author_position)) as pid_ao,
       block_fullname,
       author_group_orcid                                    as orcid,
       -- -- Note has verified all mag_author_id is successfully matched
       -- lowerUTF8(author_name) as author_name,
       -- arrayStringConcat(extractAll(lowerUTF8(author_affiliation), '\\w{1,}'), ' ')    as author_affiliation,
       -- coauthors,
       -- arrayStringConcat(extractAll(lowerUTF8(venue), '\\w{1,}'), ' ')    as venue,
       -- pub_year,
       arrayStringConcat(extractAll(lowerUTF8(concat(paper_title, ' ', paper_abstract)), '\\w{1,}'), ' ')   as content
from and_ds.our_and_dataset_block any
         left join (
    select pid, aid, author_position
    from (select PaperId as pid, AuthorId as aid, toInt64(AuthorSequenceNumber) as author_position
          from mag.paper_author_affiliation) any
             inner join and_ds.our_and_dataset_block using pid, author_position
    ) using pid, author_position
    where xxHash32(block_fullname) %% 10=%d
'''


def sparse_tensor_tfidf_similarity(documents):
    tfidf_csr = vectorizer.transform(documents)

    if len(documents) < 300:
        # Note computer by CPU
        m = tfidf_csr * tfidf_csr.T
        similarity = m.A
    else:
        # Note computer by GPU
        coo = tfidf_csr.tocoo()
        indices = np.vstack((coo.row, coo.col))
        st = torch.sparse.FloatTensor(torch.LongTensor(indices),
                                      torch.FloatTensor(coo.data),
                                      torch.Size(coo.shape)).to(device)
        # Note this feature require a high version of pytorch
        multipled_st = torch.sparse.mm(st, torch.transpose(st, 0, 1))
        similarity = multipled_st.to_dense().cpu().numpy()

    return similarity


for seg in range(0, 10, 1):
    sql = sql_metadata % seg
    print(sql)
    # Note prepare the paper metadata dict
    df_metadata = DBReader.tcp_model_cached_read(cached_file_path='yyy', sql=sql, cached=False)
    print(df_metadata.shape)
    print(df_metadata.head())

    md_block_fullname_dict = dict(zip(df_metadata['pid_ao'].values, df_metadata['block_fullname'].values))
    md_content_dict = dict(zip(df_metadata['pid_ao'].values, df_metadata['content'].values))

    del df_metadata

    # Note generate the pairwise similarity
    documents = md_content_dict.values()
    vectorizer = TfidfVectorizer()  # tokenizer=normalize, stop_words='english'
    print('fit tfidf model')
    vectorizer = vectorizer.fit(documents)

    all_block_feature = {}
    sql = sql_block % seg
    print(sql)
    df_block = DBReader.tcp_model_cached_read(cached_file_path='xxx', sql=sql, cached=False)
    print(df_block.shape)

    for ij, row in tqdm(df_block.iterrows(), total=df_block.shape[0]):
        block_name, pid_aos, ground_truths, mag_preds = row
        documents = [md_content_dict[pid_ao] for pid_ao in pid_aos]
        tfidf_similarity = sparse_tensor_tfidf_similarity(documents)
        tfidf_similarity = np.array(tfidf_similarity, dtype=np.float16)
        all_block_feature[block_name] = tfidf_similarity

    joblib.dump(all_block_feature,
                filename=os.path.join(cached_dir, 'cluster_feature/tfidf-feature-%d.pkl' % seg))
