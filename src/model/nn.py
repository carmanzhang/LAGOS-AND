import torch
import torch.nn as nn


class MatchGRU(nn.Module):
    def __init__(self, glove, hidden_dim=64, num_layers=2, num_hand_craft_feature=5, bidirectional=True, output_dim=1):
        super(MatchGRU, self).__init__()
        embedding_dim = len(glove.vectors[0])
        self.embedding = nn.Embedding.from_pretrained(glove.vectors, freeze=True)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=0.5, bidirectional=bidirectional)
        # self.match_fc = nn.Linear(2 * num_layers * hidden_dim * (2 if bidirectional else 1), 5)
        self.match_fc = nn.Sequential(
            nn.Linear(2 * num_layers * hidden_dim * (2 if bidirectional else 1), 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.5),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.5),

            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, input):
        XL, XR = input

        # output: [batch-size, Sequence-len, embedding-dim]
        XL = self.embedding(XL)
        XL, hl = self.gru(XL)
        # print(XL.shape, hl.shape)
        hl = torch.cat([hl[i] for i in range(len(hl))], dim=1)
        # print(hl.shape)

        # output: [batch-size, Sequence-len, embedding-dim]
        XR = self.embedding(XR)
        XR, hr = self.gru(XR)
        hr = torch.cat([hr[i] for i in range(len(hr))], dim=1)

        res = torch.cat([hl, hr], dim=1)
        res = self.match_fc(res)

        # convert to 0-1 possibility distribution
        # res = torch.softmax(res, dim=1)

        # add hand-craft features
        # res = torch.cat([HF, res], dim=1)
        # res = self.ml_hidden_fc(res)
        # print(res.shape)
        return res
