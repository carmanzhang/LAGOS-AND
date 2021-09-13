import json
import pickle
import warnings
from random import random

import numpy as np
import pandas as pd
import seaborn as sb

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sb.set_theme(style="ticks", rc=custom_params)

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from eutilities.customized_print import pprint
from eutilities.preprocessor import drop_missing_items, down_sample
from metric.metric import calc_metrics, metric_names
from model.available_model import ModelName
from model.classification import use_classifier

warnings.filterwarnings('ignore')

feature_file = 'cached/our_and_dataset_feature_full.tsv'
df = pd.read_csv(feature_file, sep='\t')
column_names = df.columns.values.tolist()
print('column_names: ', column_names)

df = df[df['train1_test0_val2'] != -1]  # -1 represents for the corpus for training neural network module
df = df[df['train1_test0_val2'] != 2]  # -2 represents for the corpus for evaluating neural network module
print('dataset size before deduplication', df.shape)
print('pos_samples_num: ', df[df['same_author'] == 1].shape[0])
print('neg_samples_num: ', df[df['same_author'] == 0].shape[0])
df.drop_duplicates(keep='first', inplace=True)
print('dataset size after deduplication', df.shape)
print('pos_samples_num: ', df[df['same_author'] == 1].shape[0])
print('neg_samples_num: ', df[df['same_author'] == 0].shape[0])

mode_names = ModelName.available_modes()
print('available_modes: ', mode_names)

matching_score_dict = json.loads(open('cached/matching_score.json').readline())
print(len(matching_score_dict))


def get_score(row):
    try:
        k = '-'.join(map(lambda x: str(x), row.values.astype(int)))
        if k in matching_score_dict:
            # print('hit')
            v = float(matching_score_dict[k])
            v = v if v > 0.5 else 0
            return v
        else:
            # print('nan')
            return np.nan
    except Exception as e:
        print(e)
        return np.nan


df['random'] = df['pid1'].apply(lambda x: random())

df['match_score'] = df[['pid1', 'ao1', 'pid2', 'ao2']].apply(
    lambda row: get_score(row), axis=1)

# feature_names = ['tfidf_cosin_similarity']
# feature_names = ['paper_title_abstract_similarity']
# feature_names = ['content_cosin_similarity']
# feature_names = ['match_score']

feature_names_groups = [
    # ['random'],
    # ['same_biblio_aid'],
    # ['name_similarity'],
    # ['same_biblio_aid', 'name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity'],
    # ['same_biblio_aid', 'name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity', 'paper_title_abstract_similarity'],
    # ['same_biblio_aid', 'name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity', 'tfidf_cosin_similarity'],
    # ['same_biblio_aid', 'name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity', 'content_cosin_similarity'],
    # ['same_biblio_aid', 'name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity', 'match_score']

    # ['name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity'],
    # ['name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity', 'paper_title_abstract_similarity'],
    # ['name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity', 'tfidf_cosin_similarity'],
    # ['name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity', 'content_cosin_similarity'],
    ['name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity', 'match_score']
]

formal_feature_name_dict = {'same_biblio_aid': 'MAG Author ID', 'name_similarity': 'Name Similarity',
                            'pub_year_diff': 'Publication Year Gap',
                            'venue_similarity': 'Venue Similarity', 'aff_similarity': 'Affiliation Similarity',
                            'paper_title_abstract_similarity': r'Content Similarity $cf_{jaccard}$',
                            'tfidf_cosin_similarity': r'Content Similarity $cf_{tfidf}$',
                            'content_cosin_similarity': r'Content Similarity $cf_{doc2vec}$',
                            'match_score': r'Content Similarity $cf_{nn}$'}

if __name__ == '__main__':
    df = shuffle(df)

    print('original shape: ', df.shape)
    df = drop_missing_items(df)
    print('after dropping none shape: ', df.shape)

    print('pos_samples_num: ', df[df['same_author'] == 1].shape[0])
    print('neg_samples_num: ', df[df['same_author'] == 0].shape[0])
    df = down_sample(df)
    print('after balancing dataset shape: ', df.shape)
    print('pos_samples_num: ', df[df['same_author'] == 1].shape[0])
    print('neg_samples_num: ', df[df['same_author'] == 0].shape[0])

    for feature_names in feature_names_groups:
        for idx, model_switch in enumerate(mode_names):
            df_copy = df.copy(deep=True)
            print('-' * 160)
            print(str(model_switch) + '\tused features:\n', '\t'.join(feature_names))
            Y = np.array(df_copy['same_author'].astype('int'))
            X = df_copy[feature_names]
            # X = scale(X)
            X = np.array(X)

            num_fold = 5
            kf = KFold(n_splits=num_fold, shuffle=True)
            avg_metrics = []
            indx_split = kf.split(Y)
            for round_idx, (train_index, test_index) in enumerate(indx_split):
                train_X, train_y = X[train_index], Y[train_index]
                test_X, test_y = X[test_index], Y[test_index]
                pred_y, model = use_classifier(train_X, train_y, test_X, model_switch=model_switch)
                # save the model
                file_name = 'cached/lagos-and-rf-model.pkl'
                pickle.dump(model, open(file_name, 'wb'))

                importances = model.feature_importances_
                pprint(list(zip(feature_names, importances)), sep='\t')
                if round_idx == 0:
                    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
                    plt.figure(figsize=(6, 4), dpi=300)
                    plt.grid(linestyle='dashed', linewidth=1, axis='y')

                    plt.errorbar([formal_feature_name_dict[n] for n in feature_names], importances, yerr=std,
                                 fmt='D',
                                 # mfc='#C9A66B',
                                 # mec='#662E1C',
                                 ms=3,
                                 mew=3,
                                 ecolor='#AF4425',
                                 lw=3,
                                 ls=':',
                                 color='#AF4425',
                                 capsize=6)
                    plt.ylabel('feature contribution', loc='center')  # 'top'
                    plt.xticks(fontsize=8, rotation=10, ha='center')
                    plt.tight_layout()
                    plt.savefig('cached/feature_contribution.png', dpi=600)
                    plt.show()

                fpr, tpr, threshold = roc_curve(test_y, pred_y, pos_label=1)
                roc_auc = auc(fpr, tpr)

                # select best Cutoff Value(Decision Threshold)
                metric_dict = calc_metrics(test_y, pred_y)
                metric_tuple = [(m, metric_dict[m]) for m in metric_names]
                # pprint(metric_tuple, pctg=True, sep='\t')
                avg_metrics.append(metric_dict)

            avg_metric_vals = [np.average([item[m] for item in avg_metrics]) for m in metric_names]
            print(metric_names)
            pprint(list(zip(metric_names, avg_metric_vals)), pctg=True, sep='\t')
