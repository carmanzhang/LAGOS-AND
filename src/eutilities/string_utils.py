import json
import os
import re
import string
import unicodedata

import geograpy
import jaro
from nltk import ngrams

os.environ['JAVAHOME'] = "/usr/local/jdk-11.0.1"
from Levenshtein.StringMatcher import StringMatcher
from nltk.tag import StanfordNERTagger
from nltk.corpus import stopwords

def extract_email(affi):
    match = re.search(r'[\w\.-]+@[\w\.-]+', affi)
    if match is not None:
        result = match.group(0)

        if result[-1] == '.':
            result = result[:len(result) - 1]
        return result
    return None


def extract_inner_words(string):
    replaced = re.sub('[^a-z]', " ", string)
    splts = replaced.split(' ')
    return [s for s in splts if len(s) > 2]


def extract_word_list(string):
    return re.findall(r'\w+', string)


def extract_key_wods_list(key_words_str):
    key_words = []
    key_words_dict = json.loads(key_words_str)
    if key_words_dict == None:
        return []
    for item in key_words_dict:
        if 'keyword' in item:
            keyword_ = item['keyword']
            keyword_ = extract_inner_words(keyword_)
            key_words += keyword_
    return key_words


# 有28895个不重复的 mesh heading
def extract_mesh_headings(raw_str: str):
    s = json.loads(raw_str)
    if s == None:
        return []
    desc_name_list = []
    for item in s:
        if 'descriptorName' in item:
            # TODO 'qualifierNameList'
            descriptorname_ = item['descriptorName']
            descriptorname_ = extract_inner_words(descriptorname_)
            desc_name_list += descriptorname_
    return desc_name_list


def edit_distinct_diff_chars(str1, str2):
    str_matcher = StringMatcher()
    if len(str1) < len(str2):
        str1, str2 = str2, str1
    str_matcher.set_seqs(str1, str2)
    editops = str_matcher.get_editops()
    # print(editops)
    diff_chars = []
    for model, pos1, pos2 in editops:
        if model == 'delete':
            # print('delete: ', str1[pos1])
            diff_chars.append(str1[pos1])
        elif model == 'replace':
            # print('replace: ', str1[pos1])
            diff_chars.append(str1[pos1])
        elif model == 'insert':
            # print('insert: ', str2[pos2])
            diff_chars.append(str2[pos2])
    return diff_chars


def jaro_winkler_similarity(s1, s2):
    if s1 is None or s2 is None:
        return 0.0
    return jaro.jaro_winkler_metric(s1, s2)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
all_letters = string.ascii_letters + " -"
all_letters = set([c for c in all_letters])
n_letters = len(all_letters)


def convert_unicode_to_ascii(s):
    s = s.lower()
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def ngram_sequence(s, n=2):
    grams = ngrams(list(s), n)
    grams = [''.join(gram) for gram in grams]
    return grams


en_stopwords_set = set(stopwords.words('english'))


def intersection(a, b, remove_stop_word=False):
    if a is None or b is None:
        return 0
    if remove_stop_word:
        a = [n for n in a if n not in en_stopwords_set]
        b = [n for n in b if n not in en_stopwords_set]
    intersections = len(set(a).intersection(set(b)))
    return intersections


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


# 3 class:	Location, Person, Organization
# 4 class:	Location, Person, Organization, Misc
# 7 class:	Location, Person, Organization, Money, Percent, Date, Time
# english.all.3class.caseless.distsim.crf.ser.gz
# english.conll.4class.caseless.distsim.crf.ser.gz
# english.muc.7class.caseless.distsim.crf.ser.gz
stanford_ner_base_path = '/home/zhangli/mydisk-2t/apps/stanford-ner-4.0.0/'
st = StanfordNERTagger(
    model_filename=('%sclassifiers/english.all.3class.distsim.crf.ser.gz' % stanford_ner_base_path),
    path_to_jar=('%sstanford-ner.jar' % stanford_ner_base_path))


def ner(s):
    if s is None or len(s) == 0:
        return [], []
    res = st.tag(s.split())
    print(res)
    l = len(res)
    broken_point = [i + 1 for i in range(l - 1) if res[i][1] != res[i + 1][1]]
    start = [0] + broken_point
    end = broken_point + [l]
    locs, orgs = [], []
    for s, e in zip(start, end):
        if e <= s:
            continue
        entities_with_class = res[s:e]
        cls = entities_with_class[0][1]
        entity = ' '.join([n[0] for n in entities_with_class])
        if cls == 'ORGANIZATION':
            orgs.append(entity)
        elif cls == 'LOCATION':
            locs.append(entity)
    return locs, orgs


cached_extracted_geo = dict()


def extract_geo(s):
    if s is None or len(s) == 0:
        return [[], [], [], []]
    if s not in cached_extracted_geo:
        # places = geograpy.Extractor(text=s).find_geoEntities()
        places = geograpy.get_geoPlace_context(text=s)
        cached_extracted_geo[s] = [
            [n.lower() for n in places.countries],
            [n.lower() for n in places.regions],
            [n.lower() for n in places.cities],
            [n.lower() for n in places.other]
        ]
        # print(places)
    return cached_extracted_geo[s]


if __name__ == '__main__':
    import time

    t1 = time.time()
    # s = "University of Minnesota, Minneapolis, Minnesota 55455, USA."
    s = "University of California, San Diego, La Jolla, California 92093, USA."
    for _ in range(10):
        chars = ner(s)
    t2 = time.time()
    print(t2 - t1)
    print(chars)
    print()
    for _ in range(10):
        chars = extract_geo(s)
    print(chars)
    t3 = time.time()
    print(t3 - t2)
