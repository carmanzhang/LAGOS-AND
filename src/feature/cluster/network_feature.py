import os
import sys

sys.path.append('../../')

from myconfig import glove840b300d_path, cached_dir, device
import joblib
import numpy as np
import torch
from torchtext import vocab
from tqdm import tqdm
from model.nn import MatchGRU
from mytookit.data_reader import DBReader
from nltk.corpus import stopwords
from torch.utils.data.dataset import Dataset

en_stopwords_set = set(stopwords.words('english'))

glove_vocab_size = ['6B', '840B'][1]
underlying_dataset = ['pairwise-gold-standard', 'block-gold-standard'][0]

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
order by length(pid_aos) desc
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

# Note #############################################################################################
# Note test the performance of learnable method.
print('begin load models... ')
glove = vocab.GloVe(name=glove_vocab_size, dim=300, cache=glove840b300d_path)
pad_idx = 0
batch_size = 640
max_sql_len = 300
print(max_sql_len)

print('use device: ', device)


def word_token(txt):
    words = txt.lower().split()
    tokens = [glove.stoi[word] for word in words if word in glove.stoi]
    tokens = tokens[:max_sql_len] if len(tokens) >= max_sql_len else tokens + [pad_idx] * (
            max_sql_len - len(tokens))
    tokens = np.array(tokens)
    return tokens


# the model accept the GloVe pretrained word embedding
model = MatchGRU(glove, hidden_dim=64, num_layers=2,
                 # num_hand_craft_feature=len(train_set.num_hand_craft_feature_set),
                 bidirectional=True, output_dim=2).to(device)

model_path = os.path.join(cached_dir, 'match-checkpoint-glove%s-%s.pkl' % (glove_vocab_size, underlying_dataset))
print('model_path: %s' % model_path)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print('end load models... ')


class MyDataset(Dataset):
    def __init__(self, all_XL, all_XR):
        self.all_XL = all_XL
        self.all_XR = all_XR

    def __getitem__(self, index):
        XL = self.all_XL[index]
        XR = self.all_XR[index]
        return XL, XR

    def __len__(self):
        return self.all_XL.size(0)


def compute_batch_pairwise_similarity(pairwise_dataset):
    all_input_loader = torch.utils.data.DataLoader(dataset=pairwise_dataset,
                                                   batch_size=batch_size,
                                                   # Note shuffle should not be True,
                                                   # Note we do not perform shuffle because the output should be in order with the inout samples
                                                   shuffle=False)

    prediction1 = torch.tensor([], device=device)
    prediction2 = torch.tensor([], device=device)

    for batch_idx, (XL, XR) in enumerate(all_input_loader):
        # Note matching similarity
        # XL, XR = torch.tensor(word_token(content1), device=device), torch.tensor(word_token(content2), device=device)
        with torch.no_grad():
            output = model([XL, XR])

            # Note if using BCELogistiLoss, the model does not contain the activation layer
            pred = output.sigmoid()

            prediction1 = torch.cat((prediction1, pred))
            # prediction2 = torch.cat((prediction2, pred2))

    # pred_label_numpy = [1 if n[1] > 0.5 else 0 for n in prediction.cpu().numpy()]

    # for i, pid_ao_i in enumerate(pid_aos):
    #     for j, pid_ao_j in enumerate(pid_aos):
    #         content_word1, content_word2 = md_content_word_dict[pid_ao_i], md_content_word_dict[pid_ao_j]
    #         # content1, content2 = md_content_dict[pid_ao_i], md_content_dict[pid_ao_j]
    #         # doc2vec_v1, doc2vec_v2 = md_doc2vec_emd_dict[pid_ao_i], md_doc2vec_emd_dict[pid_ao_j]
    #
    #         # Note matching similarity
    #         XL, XR = torch.tensor(word_token(content1), device=device), torch.tensor(word_token(content2), device=device)
    #         prediction = torch.tensor([], device=device)
    #         with torch.no_grad():
    #             output = model([XL, XR])
    #             pred = output.sigmoid()
    #             prediction = torch.cat((prediction, pred))
    #
    #         pred_label_numpy = [1 if n[1] > 0.5 else 0 for n in prediction.cpu().numpy()]
    #         pred_prob = [n[1] for n in prediction.cpu().numpy()]

    return prediction1, prediction2


for seg in list(range(0, 10, 1))[::-1]:
    sql = sql_metadata % seg
    print(sql)
    # Note prepare the paper metadata dict
    # df_metadata = DBReader.tcp_model_cached_read(cached_file_path='yyy', sql=sql, cached=False)
    df_metadata = DBReader.tcp_model_cached_read(
        cached_file_path=os.path.join(cached_dir, 'block_data/block_metadata_%d.pkl' % seg), sql=sql, cached=True)

    print(df_metadata.shape)
    # print(df_metadata[['pid_ao', 'content']].values[:100])

    md_block_fullname_dict = dict(zip(df_metadata['pid_ao'].values, df_metadata['block_fullname'].values))
    md_orcid_dict = dict(zip(df_metadata['pid_ao'].values, df_metadata['orcid'].values))

    md_content_word_embedding = dict(
        zip(df_metadata['pid_ao'].values, df_metadata['content'].apply(lambda x: word_token(x)).values))

    del df_metadata

    all_block_feature = {}
    sql = sql_block % seg
    print(sql)
    # df_block = DBReader.tcp_model_cached_read(cached_file_path='xxx', sql=sql, cached=False)
    df_block = DBReader.tcp_model_cached_read(cached_file_path=os.path.join(cached_dir, 'block_data/block_data_%d.pkl' % seg),
                                              sql=sql, cached=True)
    for ij, row in tqdm(df_block.iterrows(), total=df_block.shape[0]):
        block_name, pid_aos, ground_truths, mag_preds = row

        # Note calculate the similarity between different metadata according to pid_ao
        num_instances = len(pid_aos)

        if num_instances > 700:
            block_content_term_ids = torch.tensor(np.array([md_content_word_embedding[pid_ao] for pid_ao in pid_aos]),
                                                  device=device)
            # Note this block is very large that can not fit into the GPU RAM, thus, we should process the each XL individually.
            all_XR = block_content_term_ids

            prediction1, prediction2 = torch.tensor([], device=device), torch.tensor([], device=device)
            for i in range(num_instances):
                one_XL = block_content_term_ids[i]
                one_XL = one_XL.unsqueeze(0).repeat(num_instances, 1)
                assert one_XL.shape == all_XR.shape

                pairwised_dataset = MyDataset(one_XL, all_XR)
                tmp_prediction1, tmp_prediction2 = compute_batch_pairwise_similarity(pairwised_dataset)
                prediction1 = torch.cat((prediction1, tmp_prediction1))
                prediction2 = torch.cat((prediction2, tmp_prediction2))
        else:
            block_content_term_ids = torch.tensor(np.array([md_content_word_embedding[pid_ao] for pid_ao in pid_aos]),
                                                  device=device)
            all_XL = block_content_term_ids.repeat(1, block_content_term_ids.size(0)).view(-1, block_content_term_ids.size(-1))
            all_XR = block_content_term_ids.repeat(block_content_term_ids.size(0), 1)
            pairwised_dataset = MyDataset(all_XL, all_XR)
            prediction1, prediction2 = compute_batch_pairwise_similarity(pairwised_dataset)

        pred_prob = [prediction1.reshape(num_instances, -1).cpu().numpy().astype(np.float16),
                     prediction2.reshape(num_instances, -1).cpu().numpy().astype(np.float16)]

        all_block_feature[block_name] = pred_prob

    joblib.dump(all_block_feature, filename=os.path.join(cached_dir,
                                                         'cluster_feature/matching-features-glove840B-%d-with-model-trained-on-%s.pkl' % (
                                                             seg, underlying_dataset)))
