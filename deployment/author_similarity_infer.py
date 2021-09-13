import json
import pickle
import re
import traceback
import unicodedata

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from nltk import ngrams

port = 38081

# endpoint for the content similarity module
url = 'http://127.0.0.1:38080/score'

# init flask server
server = Flask(__name__)
server.config['JSON_AS_ASCII'] = False
CORS(server, supports_credentials=True)

# load the ML-classifier model
file_name = 'lagos-and-rf-model.pkl'
model = pickle.load(open(file_name, 'rb'))
print("successfully loaded the ML classifier model")

en_stopwords_set = set(
    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
     'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
     "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
     'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
     'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
     'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
     'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
     'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
     'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
     's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
     'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
     "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
     'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
     "wouldn't"])


def get_request_content(req, arg='metadata'):
    if req.method == 'POST':
        content = req.form.get(arg)
    else:
        content = req.args.get(arg)
    return content


def input_check(content: str):
    return True


def model_infer(features):
    scores = model.predict_proba(features)
    scores = [n[1] for n in scores]
    return scores


def remote_compute_content_similarity(paired_content: list):
    data = json.dumps({"contents": paired_content})
    print(data)
    ret = requests.post(url, data={"content": data})
    ret = json.loads(str(ret.text))
    # {'err_code': 0, 'err_msg': '', 'scores': scores}
    if ret['err_code'] != 0:
        return [], ret['err_msg']

    # this step in important here
    scores = [float(n) if float(n) > 0.5 else 0 for n in ret['scores']]
    return scores, None


def jaccard_similarity(a, b, remove_stop_word=False):
    if a is None or b is None:
        return 0.0
    if remove_stop_word:
        a = [n for n in a if n not in en_stopwords_set]
        b = [n for n in b if n not in en_stopwords_set]
    unions = len(set(a).union(set(b)))
    if unions == 0:
        return 0.0
    intersections = len(set(a).intersection(set(b)))
    return 1. * intersections / unions


def ngram_sequence(s, n=2):
    grams = ngrams(list(s), n)
    grams = [''.join(gram) for gram in grams]
    return grams


def convert_unicode_to_ascii(s):
    s = s.lower()
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def extract_word_list(string):
    return re.findall(r'\w+', string)


# make sure the features are in the right order
# ['name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity', 'match_score']
def input_data_to_features(input_data):
    features, paired_content = [], []
    for i, ([author_names1, author_names2], [pub_year1, pub_year2], [venue1, venue2], [aff1, aff2], [
        paper_content1, paper_content2]) in enumerate(input_data):
        author_names1, author_names2 = author_names1.lower(), author_names2.lower()
        # name similarity
        name_similarity = jaccard_similarity(ngram_sequence(convert_unicode_to_ascii(author_names1)),
                                             ngram_sequence(convert_unicode_to_ascii(author_names2)))

        pub_year_diff = abs(pub_year1 - pub_year2) if pub_year1 > 0 and pub_year2 > 0 else -1

        venue_similarity = jaccard_similarity(extract_word_list(str(venue1).lower()),
                                              extract_word_list(str(venue2).lower()))

        aff_similarity = jaccard_similarity(extract_word_list(str(aff1).lower()),
                                            extract_word_list(str(aff2).lower()))
        features.append(
            [name_similarity,
             pub_year_diff,
             venue_similarity,
             aff_similarity,
             ])
        paired_content.append([paper_content1.lower(), paper_content2.lower()])

    # add content similarity scored by the neural network
    scores, err_msg = remote_compute_content_similarity(paired_content)
    if err_msg is None:
        assert len(scores) == len(features)
        features = [n + [scores[i]] for i, n in enumerate(features)]
    else:
        raise Exception(err_msg)

    # features = [n + [0.0] for i, n in enumerate(features)]
    return features


@server.route('/score', methods=['post', 'get'])
@cross_origin()
def score():
    try:
        metadata = get_request_content(request)
        # print(content)
        if not input_check(metadata):
            res = {'err_code': 1, 'err_msg': 'bas input', 'scores': []}
        else:
            metadata = json.loads(metadata)
            features = input_data_to_features(metadata)
            scores = model_infer(features)
            # convert foloat list to string list, in order to be dumps to json string
            scores = [str(n) for n in scores]
            assert len(scores) == len(metadata)
            # print(scores)
            res = {'err_code': 0, 'err_msg': '', 'scores': scores}
    except Exception as e:
        res = {'err_code': 1, 'err_msg': str(e), 'scores': []}
        traceback.print_exc()
    return jsonify(res)


server.run(host='0.0.0.0', port=port)
