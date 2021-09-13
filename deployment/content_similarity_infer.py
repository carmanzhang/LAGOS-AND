import json
import traceback

import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from torch.utils.data.dataset import Dataset
from torchtext import vocab

# from model.nn import MatchGRU
# init flask server
server = Flask(__name__)
server.config['JSON_AS_ASCII'] = False
CORS(server, supports_credentials=True)
port = 38080

model_pkl_path = './match-checkpoint.pkl'
glove = vocab.GloVe(name='6B', dim=100)
pad_idx = 0
batch_size = 128
epochs = 30
max_sql_len = 550
print(max_sql_len)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('use device: ', device)


class MatchGRU(nn.Module):
    def __init__(self, glove, hidden_dim=64, num_layers=2, num_hand_craft_feature=5, bidirectional=True, output_dim=2):
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
            nn.Linear(16, 2),
            nn.ReLU(),
        )

    def forward(self, input):
        XL, XR = input
        XL = self.embedding(XL)
        XL, hl = self.gru(XL)
        hl = torch.cat([hl[i] for i in range(len(hl))], dim=1)
        XR = self.embedding(XR)
        XR, hr = self.gru(XR)
        hr = torch.cat([hr[i] for i in range(len(hr))], dim=1)

        res = torch.cat([hl, hr], dim=1)
        res = self.match_fc(res)
        return res


def word_token(txt):
    words = txt.lower().split()
    tokens = [glove.stoi[word] for word in words if word in glove.stoi]
    tokens = tokens[:max_sql_len] if len(tokens) >= max_sql_len else tokens + [pad_idx] * (
            max_sql_len - len(tokens))
    return tokens


class InputBulkDataset(Dataset):
    def __init__(self, input_data):
        self.input_data = input_data

    def __getitem__(self, index):
        data_item = self.input_data[index]
        content1, content2 = data_item[0], data_item[1]
        XL, XR = torch.tensor(word_token(content1)), torch.tensor(word_token(content2))
        return XL, XR

    def __len__(self):
        return len(self.input_data)


# load model
model = MatchGRU(glove, hidden_dim=64, num_layers=2,
                 # num_hand_craft_feature=len(train_set.num_hand_craft_feature_set),
                 bidirectional=True, output_dim=2).to(device)

# Todo switch to GPU if your machine support
# checkpoint = torch.load(model_pkl_path)
checkpoint = torch.load(model_pkl_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def get_request_content(req, arg='content'):
    if req.method == 'POST':
        content = req.form.get(arg)
    else:
        content = req.args.get(arg)
    return content


def input_check(content: str):
    # content = json.loads(content)['contents']
    # if type(content) != 'list' or len(content) == 0:
    #     content
    return True


def model_infer(input_data):
    input_data = InputBulkDataset(input_data)
    # here, we must set shuffle=False, because we want infer the similarity in place
    input_data_loader = torch.utils.data.DataLoader(dataset=input_data, batch_size=batch_size, shuffle=False,
                                                    num_workers=1)

    prediction = torch.tensor([], device=device)
    with torch.no_grad():
        for (XL, XR) in input_data_loader:
            XL, XR = XL.to(device), XR.to(device)
            output = model([XL, XR])
            pred = output.sigmoid()
            prediction = torch.cat((prediction, pred))
    pred_prob = [n[1] for n in prediction.cpu().numpy()]
    return pred_prob


@server.route('/score', methods=['post', 'get'])
@cross_origin()
def score():
    try:
        content = get_request_content(request)
        # print(content)
        if not input_check(content):
            res = {'err_code': 1, 'err_msg': 'bas input', 'scores': []}
        else:
            contents = json.loads(content)
            contents = contents['contents']
            scores = model_infer(contents)
            # convert foloat list to string list, in order to be dumps to json string
            scores = [str(n) for n in scores]
            assert len(scores) == len(contents)
            # print(scores)
            res = {'err_code': 0, 'err_msg': '', 'scores': scores}
    except Exception as e:
        res = {'err_code': 1, 'err_msg': str(e), 'scores': []}
        traceback.print_exc()
    return jsonify(res)


server.run(host='0.0.0.0', port=port)
