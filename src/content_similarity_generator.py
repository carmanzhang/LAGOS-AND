import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data.dataset import Dataset
from torchtext import vocab

from eutilities import train_utils
from model.nn import MatchGRU
from myio.data_reader import DBReader

glove = vocab.GloVe(name='6B', dim=100)
pad_idx = 0
batch_size = 128
epochs = 30
max_sql_len = 550
print(max_sql_len)

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
        Y = torch.tensor([1, 0], dtype=torch.float) if same_author == 0 else torch.tensor([0, 1], dtype=torch.float)
        MT = [int(pid1), int(ao1), int(pid2), int(ao2), int(same_author)]
        return MT, XL, XR, Y

    def __len__(self):
        return len(self.df)


df = DBReader.tcp_model_cached_read("cached/matching_corpus.pkl",
                                    """select * from and_ds.matching_network_train_corpus""",
                                    cached=False)
print(df.shape)
# df = df.dropna(0)

print(set(df['train1_test0_val2'].values))
df_train_set = df[df['train1_test0_val2'].astype(int) == -1]
# # TODO
# df_train_set = df_train_set.sample(frac=0.8, random_state=42)
df_val_set = df[df['train1_test0_val2'].astype(int) == 2]
df_test_set = pd.concat([df[df['train1_test0_val2'].astype(int) == 0], df[df['train1_test0_val2'].astype(int) == 1]])

df_train_set.reset_index(inplace=True, drop=True)
df_val_set.reset_index(inplace=True, drop=True)
df_test_set.reset_index(inplace=True, drop=True)

print('df_train shape:', df_train_set.shape, 'df_val shape:', df_val_set.shape, 'df_test shape:', df_test_set.shape)
train_set = ANDDataset(df_train_set)
val_set = ANDDataset(df_val_set)
test_set = ANDDataset(df_test_set)

# Instantiate the dataset and get data loaders. The training dataset is split into train_set and test_set.
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,
                                           num_workers=8)  # collate_fn=pad_collate
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False,
                                         num_workers=8)  # collate_fn=pad_collate
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False,
                                          num_workers=8)  # collate_fn=pad_collate

model = MatchGRU(glove, hidden_dim=64, num_layers=2,
                 # num_hand_craft_feature=len(train_set.num_hand_craft_feature_set),
                 bidirectional=True, output_dim=2).to(device)
print(model)
# Actual training
criterion = nn.BCEWithLogitsLoss()
parameters = model.parameters()
optimizer = optim.Adam(parameters, lr=1e-4)

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
        }, 'cached/match-checkpoint.pkl')

plt.figure(figsize=(5, 3))
plt.plot(np.arange(1, len(losst) + 1), losst, label="training")
plt.plot(np.arange(1, len(lossv) + 1), lossv, label="validation")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid()
plt.title('loss vs epoch')
plt.show()

checkpoint = torch.load('cached/match-checkpoint.pkl')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print('Epoch:', checkpoint['epoch'])
print('losst:', checkpoint['losst'])
print('lossv:', checkpoint['lossv'])
model.eval()

d = dict()
metadata, true_label_numpy, pred_label_numpy, pred_prob = train_utils.validate(model, test_loader, criterion, [])
print(metadata.shape)
assert metadata.shape[1] == len(true_label_numpy) == len(pred_label_numpy)

same_author_metadata = metadata[4]
for i, n in enumerate(true_label_numpy):
    k = '-'.join(list(map(lambda x: str(x), [metadata[0][i], metadata[1][i], metadata[2][i], metadata[3][i]])))
    m = same_author_metadata[i]
    assert n == m
    prob = pred_prob[i]
    print(n, pred_label_numpy[i], prob)
    d[k] = str(prob)

with open('cached/matching_score.json', 'w') as fw:
    fw.write(json.dumps(d) + '\n')
