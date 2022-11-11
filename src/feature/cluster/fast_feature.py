import os

import joblib
import numpy as np
from mytookit.data_reader import DBReader
from nltk.corpus import stopwords
from tqdm import tqdm

from eutilities.string_utils import jaccard_similarity, ngram_sequence, convert_unicode_to_ascii
from myconfig import cached_dir

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
       lowerUTF8(author_name) as author_name,
       arrayStringConcat(extractAll(lowerUTF8(author_affiliation), '\\w{1,}'), ' ')    as author_affiliation,
       -- coauthors,
       arrayStringConcat(extractAll(lowerUTF8(venue), '\\w{1,}'), ' ')    as venue,
       pub_year,
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

num_features = 5

for seg in range(0, 10, 1):
    sql = sql_metadata % seg
    print(sql)
    # Note prepare the paper metadata dict
    df_metadata = DBReader.tcp_model_cached_read(cached_file_path='yyy', sql=sql, cached=False)
    print(df_metadata.shape)
    print(df_metadata.head())

    md_block_fullname_dict = dict(zip(df_metadata['pid_ao'].values, df_metadata['block_fullname'].values))
    md_orcid_dict = dict(zip(df_metadata['pid_ao'].values, df_metadata['orcid'].values))
    md_author_name_dict = dict(zip(df_metadata['pid_ao'].values,
                                   df_metadata['author_name'].apply(
                                       lambda x: ngram_sequence(convert_unicode_to_ascii(x))).values))
    md_author_affiliation_dict = dict(
        zip(df_metadata['pid_ao'].values, df_metadata['author_affiliation'].apply(lambda x: x.split(' ')).values))
    # md_coauthors_dict = dict(zip(df_metadata['pid_ao'].values, df_metadata['coauthors'].values))
    md_venue_dict = dict(zip(df_metadata['pid_ao'].values, df_metadata['venue'].apply(lambda x: x.split(' ')).values))
    md_pub_year_dict = dict(zip(df_metadata['pid_ao'].values, df_metadata['pub_year'].values))
    # md_content_dict = dict(zip(df_metadata['pid_ao'].values, df_metadata['content'].values))
    md_content_word_dict = dict(zip(df_metadata['pid_ao'].values, df_metadata['content'].apply(
        lambda x: set([w for w in x.split(' ') if not w in en_stopwords_set])).values))
    # md_doc2vec_emd_dict = dict(zip(df_metadata['pid_ao'].values, df_metadata['content'].apply(
    #     lambda x: doc2vec_model.infer_vector(x.split(' '), steps=12, alpha=0.025)).values))

    del df_metadata

    all_block_feature = {}
    sql = sql_block % seg
    print(sql)
    df_block = DBReader.tcp_model_cached_read(cached_file_path='xxx', sql=sql, cached=False)
    print(df_block.shape)
    for ij, row in tqdm(df_block.iterrows(), total=df_block.shape[0]):
        block_name, pid_aos, ground_truths, mag_preds = row

        # Note calculate the similarity between different metadata according to pid_ao
        num_instances = len(pid_aos)
        # if ij % 10 == 0:
        #     print(num_instances)

        pairwise_feature_matrix = np.zeros(shape=(num_instances, num_instances, num_features), dtype=np.float16)
        for i, pid_ao_i in enumerate(pid_aos):
            for j, pid_ao_j in enumerate(pid_aos):
                author_names1, author_names2 = md_author_name_dict[pid_ao_i], md_author_name_dict[pid_ao_j]
                aff_arr1, aff_arr2 = md_author_affiliation_dict[pid_ao_i], md_author_affiliation_dict[pid_ao_j]

                orcid1, orcid2 = md_orcid_dict[pid_ao_i], md_orcid_dict[pid_ao_j]
                content_word1, content_word2 = md_content_word_dict[pid_ao_i], md_content_word_dict[pid_ao_j]

                venue1, venue2 = md_venue_dict[pid_ao_i], md_venue_dict[pid_ao_j]
                pub_year1, pub_year2 = md_pub_year_dict[pid_ao_i], md_pub_year_dict[pid_ao_j]

                # if author_names1 != convert_unicode_to_ascii(author_names1):
                #     print(author_names1, convert_unicode_to_ascii(author_names1))

                name_similarity = jaccard_similarity(author_names1, author_names2)

                pub_year_diff = 1.0 * (abs(pub_year1 - pub_year2) if pub_year1 > 0 and pub_year2 > 0 else -1)

                paper_title_abstract_similarity = jaccard_similarity(content_word1, content_word2, remove_stop_word=False)

                venue_similarity = jaccard_similarity(venue1, venue2)

                aff_similarity = jaccard_similarity(aff_arr1, aff_arr2)

                pairwise_feature_matrix[i][j] = [name_similarity,
                                                 pub_year_diff,
                                                 paper_title_abstract_similarity,
                                                 venue_similarity,
                                                 aff_similarity]
        all_block_feature[block_name] = pairwise_feature_matrix

    joblib.dump(all_block_feature, filename=os.path.join(cached_dir, 'cluster_feature/five-fast-features-%d.pkl' % seg))
