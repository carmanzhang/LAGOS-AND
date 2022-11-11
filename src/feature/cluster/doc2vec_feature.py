import os
from multiprocessing import Pool

import joblib
import torch
from gensim.models import Doc2Vec
from mytookit.data_reader import DBReader
from nltk.corpus import stopwords
from tqdm import tqdm

from myconfig import cached_dir, device

en_stopwords_set = set(stopwords.words('english'))

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

# load doc2vec model
print('begin load models... ')
doc2vec_model = Doc2Vec.load('../cached/doc2vec_model')
print('end load models... ')


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    b_norm = b_norm.transpose(0, 1)
    # print(a_norm.shape, b_norm.shape)
    sim_mt = torch.mm(a_norm, b_norm)
    return sim_mt


for seg in range(0, 10, 1):
    sql = sql_metadata % seg
    print(sql)
    # Note prepare the paper metadata dict
    df_metadata = DBReader.tcp_model_cached_read(cached_file_path='XXX', sql=sql, cached=False)
    print(df_metadata.shape)
    print(df_metadata.head())

    md_block_fullname_dict = dict(zip(df_metadata['pid_ao'].values, df_metadata['block_fullname'].values))

    # md_orcid_dict = dict(zip(df_metadata['pid_ao'].values, df_metadata['orcid'].values))
    # md_content_word_dict = dict(zip(df_metadata['pid_ao'].values, df_metadata['content'].apply(
    #     lambda x: set([w for w in x.split(' ') if not w in en_stopwords_set])).values))

    # for index, (pid_ao, content) in tqdm(df_metadata[['pid_ao', 'content']].iterrows(), total=df_metadata.shape[0]):
    #     doc2vec_model.infer_vector(content.split(' '), steps=12, alpha=0.025)
    # Note this step will be very slow
    # md_doc2vec_emd_dict = dict(zip(df_metadata['pid_ao'].values, df_metadata['content'].apply(
    #     lambda x: doc2vec_model.infer_vector(x.split(' '), steps=12, alpha=0.025)).values))

    def infer_vector_worker(document):
        vector = doc2vec_model.infer_vector(document.split(' '), steps=12, alpha=0.025)
        return vector


    with Pool(processes=14) as pool:
        doc2vec_emds = pool.map(infer_vector_worker, df_metadata['content'].values)

    md_doc2vec_emd_dict = dict(zip(df_metadata['pid_ao'].values, doc2vec_emds))
    del df_metadata

    all_block_feature = {}
    sql = sql_block % seg
    print(sql)
    df_block = DBReader.tcp_model_cached_read(cached_file_path='XXX', sql=sql, cached=False)
    print(df_block.shape)
    for ij, row in tqdm(df_block.iterrows(), total=df_block.shape[0]):
        block_name, pid_aos, ground_truths, mag_preds = row

        # Note calculate the similarity between different metadata according to pid_ao
        num_instances = len(pid_aos)
        # if ij % 10 == 0:
        #     print(num_instances)
        embeddings = torch.tensor([md_doc2vec_emd_dict[pid_ao] for pid_ao in pid_aos], device=device)
        pairwise_feature_matrix = sim_matrix(embeddings, embeddings)
        pairwise_feature_matrix = pairwise_feature_matrix.cpu().numpy()

        all_block_feature[block_name] = pairwise_feature_matrix

    joblib.dump(all_block_feature, filename=os.path.join(cached_dir, 'cluster_feature/doc2vec-feature-%d.pkl' % seg))
