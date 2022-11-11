import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from mytookit.data_reader import DBReader
from torch.utils.data.dataset import Dataset
from torchtext import vocab

from eutilities import train_utils
from eutilities.preprocessor import down_sample
from model.nn import MatchGRU
from myconfig import cached_dir, glove840b300d_path

underlying_dataset = 'pairwise-gold-standard'
print(underlying_dataset)
glove_vocab_size = ['6B', '840B'][1]

need_balance_dataset = True

# # Note we use the 840B model as the word embedding
glove = vocab.GloVe(name=glove_vocab_size, dim=300, cache=glove840b300d_path)
pad_idx = 0
batch_size = 128
epochs = 30
lr = 5e-5
max_sql_len = 300
print(max_sql_len)

# evice = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('use device: ', device)


def word_token(txt):
    words = txt.lower().split()
    tokens = [glove.stoi[word] for word in words if word in glove.stoi]
    tokens = tokens[:max_sql_len] if len(tokens) >= max_sql_len else tokens + [pad_idx] * (
            max_sql_len - len(tokens))
    return tokens


class ANDDataset(Dataset):
    def __init__(self, df):
        # self.num_hand_craft_feature_set = ['name_similarity', 'same_biblio_aid', 'pub_year_diff', 'venue_similarity',
        #                                    'aff_similarity']
        # df[self.num_hand_craft_feature_set] = MinMaxScaler().fit_transform(df[self.num_hand_craft_feature_set])
        # df[self.num_hand_craft_feature_set] = StandardScaler().fit_transform(df[self.num_hand_craft_feature_set])
        self.df = df

    def __getitem__(self, index):
        data_item = self.df.iloc[index]
        pid1 = data_item.pid1
        ao1 = data_item.ao1
        pid2 = data_item.pid2
        ao2 = data_item.ao2
        same_author = data_item.same_author
        # train1_test0_val2 = data_item.train1_test0_val2
        content1 = data_item.content1
        content2 = data_item.content2
        # hand-craft features
        # HF = torch.tensor([data_item[n] for n in self.num_hand_craft_feature_set],
        #                   dtype=torch.float)
        XL, XR = torch.tensor(word_token(content1)), torch.tensor(word_token(content2))
        Y = torch.tensor([0], dtype=torch.float) if same_author == 0 else torch.tensor([1], dtype=torch.float)
        MT = [int(pid1), int(ao1), int(pid2), int(ao2), int(same_author)]
        return MT, XL, XR, Y

    def __len__(self):
        return len(self.df)


df = DBReader.tcp_model_cached_read("XXXX",
                                    sql="""select * from and_ds.matching_network_train_corpus""",
                                    cached=False)
print(df.shape)
# df = df.dropna(0)

if underlying_dataset == 'pairwise-gold-standard':
    data_split_field = 'train1_test0_val2'
    print(set(df[data_split_field].values))
    df_train_set = df[df[data_split_field].astype(int) == 1]
    # df_train_set = df_train_set.sample(frac=0.8, random_state=42)
    df_val_set = df[df[data_split_field].astype(int) == 2]
    # Note because we need to give all the instances a similar score, so the infer_set used here is all the instances
    df_infer_set = df
elif underlying_dataset == 'block-gold-standard':
    pass

# Note for the training dataset, try to balance the dataset
if need_balance_dataset:
    print('pos_samples_num: ', df_train_set[df_train_set['same_author'] == 1].shape[0])
    print('neg_samples_num: ', df_train_set[df_train_set['same_author'] == 0].shape[0])
    df_train_set = down_sample(df_train_set, percent=4)
    print('after balancing dataset shape: ', df_train_set.shape)
    print('pos_samples_num: ', df_train_set[df_train_set['same_author'] == 1].shape[0])
    print('neg_samples_num: ', df_train_set[df_train_set['same_author'] == 0].shape[0])

df_train_set.reset_index(inplace=True, drop=True)
df_val_set.reset_index(inplace=True, drop=True)
df_infer_set.reset_index(inplace=True, drop=True)

print('df_train shape:', df_train_set.shape, 'df_val shape:', df_val_set.shape, 'df_test shape:', df_infer_set.shape)
train_set = ANDDataset(df_train_set)
val_set = ANDDataset(df_val_set)
infer_set = ANDDataset(df_infer_set)

# Instantiate the dataset and get data loaders. The training dataset is split into train_set and test_set.
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,
                                           num_workers=8)  # collate_fn=pad_collate
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False,
                                         num_workers=8)  # collate_fn=pad_collate
infer_loader = torch.utils.data.DataLoader(dataset=infer_set, batch_size=batch_size, shuffle=False,
                                           num_workers=8)  # collate_fn=pad_collate

model = MatchGRU(glove, hidden_dim=64, num_layers=2,
                 # num_hand_craft_feature=len(train_set.num_hand_craft_feature_set),
                 bidirectional=True, output_dim=2).to(device)
print(model)

# pos_weight (Tensor, optional): a weight of positive examples. Must be a vector with length equal to the number of classes.
pos_weight = len(df_train_set[df_train_set['same_author'] == 0]) * 1.0 / len(df_train_set[df_train_set['same_author'] == 1])

# criterion = nn.MSELoss()
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

parameters = model.parameters()
optimizer = optim.Adam(parameters, lr=lr)

losst, lossv = [], []
for epoch in range(1, epochs + 1):
    train_utils.train(model, train_loader, criterion, optimizer, epoch, epochs, losst)
    train_utils.validate(model, val_loader, criterion, lossv)
    if lossv[-1] == min(lossv):  # Current best model, push to disk
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losst': losst[-1],
            'lossv': lossv[-1]
        }, os.path.join(cached_dir, 'match-checkpoint-glove%s-%s.pkl' % (glove_vocab_size, underlying_dataset)))

plt.figure(figsize=(5, 3))
plt.plot(np.arange(1, len(losst) + 1), losst, label="training")
plt.plot(np.arange(1, len(lossv) + 1), lossv, label="validation")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid()
plt.title('loss vs epoch')
plt.show()
plt.savefig(os.path.join(cached_dir, 'match-network-training-loss.png'))

checkpoint = torch.load(os.path.join(cached_dir, 'match-checkpoint-glove%s-%s.pkl' % (glove_vocab_size, underlying_dataset)))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print('Epoch:', checkpoint['epoch'])
print('losst:', checkpoint['losst'])
print('lossv:', checkpoint['lossv'])
model.eval()

# The following code is used for inferring the similarity scores for the pairwise dataset, the simple text matching neural network acts as a content-based feature generator.
# In doing so, we find that passing the paired input to the network with different orders (LEFT-RIGHT or RIGHT-LEFT) may yield different results,
# So, we simply calcualte the same paired input twice but with different orders and calcualte the averaged scores as the final sorces.

# Note inferring LEFT_RIGHT input
d1 = dict()
metadata, true_label_numpy, pred_label_numpy, pred_prob = train_utils.validate(model, infer_loader, criterion, [],
                                                                               switch_input=False)
print(metadata.shape)
assert metadata.shape[1] == len(true_label_numpy) == len(pred_label_numpy)

same_author_metadata = metadata[4]
for i, n in enumerate(true_label_numpy):
    k = '-'.join(list(map(lambda x: str(x), [metadata[0][i], metadata[1][i], metadata[2][i], metadata[3][i]])))
    m = same_author_metadata[i]
    assert n == m
    prob = pred_prob[i]
    print(n, pred_label_numpy[i], prob)
    d1[k] = str(prob)

# Note inferring RIGHT_LEFT input
d2 = dict()
metadata, true_label_numpy, pred_label_numpy, pred_prob = train_utils.validate(model, infer_loader, criterion, [],
                                                                               switch_input=True)
print(metadata.shape)
assert metadata.shape[1] == len(true_label_numpy) == len(pred_label_numpy)

same_author_metadata = metadata[4]
for i, n in enumerate(true_label_numpy):
    k = '-'.join(list(map(lambda x: str(x), [metadata[0][i], metadata[1][i], metadata[2][i], metadata[3][i]])))
    m = same_author_metadata[i]
    assert n == m
    prob = pred_prob[i]
    print(n, pred_label_numpy[i], prob)
    d2[k] = str(prob)

d1_keys, d2_keys = set(d1.keys()), set(d2.keys())
print('number exclusive elements: %d; %d' % (
    len(d1_keys.difference(d1_keys.intersection(d2_keys))), len(d2_keys.difference(d1_keys.intersection(d2_keys)))))

d = {}
for k in d1_keys:
    d[k] = [d1[k], d2[k]]

with open(os.path.join(cached_dir, 'matching-score-glove%s-%s.json' % (glove_vocab_size, underlying_dataset)), 'w') as fw:
    fw.write(json.dumps(d) + '\n')
