import json
import os
import warnings
from random import random

import numpy as np
import pandas as pd
import seaborn as sb

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sb.set_theme(style="ticks", rc=custom_params)

from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from eutilities.customized_print import pprint
from eutilities.preprocessor import drop_missing_items
from eutilities.metric import calc_metrics, metric_names
from model.available_model import ModelName
from model.classification import use_classifier
from myconfig import cached_dir, latex_doc_base_dir
from mytookit.data_reader import DBReader

warnings.filterwarnings('ignore')

glove_vocab_size = '840B'
underlying_dataset = ['pairwise-gold-standard', 'block-gold-standard'][0]
cluster_uniq_author_gt_1 = False
# cluster_uniq_author_gt_1 = True

print(underlying_dataset)

feature_file = os.path.join(cached_dir, 'pairwise_and_dataset_feature_full.tsv')
df = pd.read_csv(feature_file, sep='\t')
column_names = df.columns.values.tolist()
print('column_names: ', column_names)
del df['content1'], df['content2']

# if we use the pairwise AND model to disambiguate trimmed-blocks-based dataset, the training dataset should contain "large" fullname
if cluster_uniq_author_gt_1:
    num_instances = len(df)
    block_uniq_author_gt_1 = set(DBReader.tcp_model_cached_read(cached_file_path='xxxxx',
                                                                sql='''select block_fullname from and_ds.our_and_dataset_block where num_unique_author_inblock = 1;''',
                                                                cached=False)['block_fullname'].values)
    df = df[df['fullname'].apply(lambda x: x not in block_uniq_author_gt_1) == 1]
    num_instances1 = len(df)
    print('removed %d instances which are in small blocks' % (num_instances - num_instances1))

print('dataset size before deduplication', df.shape)
print('pos_samples_num: ', df[df['same_author'] == 1].shape[0])
print('neg_samples_num: ', df[df['same_author'] == 0].shape[0])
df.drop_duplicates(keep='first', inplace=True)
print('dataset size after deduplication', df.shape)
print('pos_samples_num: ', df[df['same_author'] == 1].shape[0])
print('neg_samples_num: ', df[df['same_author'] == 0].shape[0])

mode_names = ModelName.available_modes()
print('available_modes: ', mode_names)

matching_score_dict = json.loads(open(os.path.join(cached_dir,
                                                   'matching-score-glove%s-%s.json' % (glove_vocab_size, underlying_dataset))
                                      ).readline())
print(len(matching_score_dict))


def get_score(row):
    try:
        k = '-'.join(map(lambda x: str(x), row.values.astype(int)))
        if k in matching_score_dict:
            # print('hit')
            v_left_right, v_right_left = matching_score_dict[k]
            v_left_right, v_right_left = float(v_left_right), float(v_right_left)
            # v = v if v > 0.5 else 0
            v = (v_left_right + v_right_left) / 2
            return v
        else:
            # print('nan')
            return np.nan
    except Exception as e:
        print('error: ', e)
        return np.nan


df['random'] = df['pid1'].apply(lambda x: random())

df['match_score'] = df[['pid1', 'ao1', 'pid2', 'ao2']].apply(
    lambda row: get_score(row), axis=1)

print(df.shape, df[['name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity', 'paper_title_abstract_similarity',
                    'tfidf_cosin_similarity', 'content_cosin_similarity', 'match_score']].mean())

feature_names_groups = [
    ['rand', ['random']],
    ['magaid', ['same_biblio_aid']],
    ['match_score', ['match_score']],
    ['name', ['name_similarity']],
    ['bf', ['name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity']],
    ['bf-cfjaccard',
     ['name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity', 'paper_title_abstract_similarity']],
    ['bf-cftfidf', ['name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity', 'tfidf_cosin_similarity']],
    ['bf-cfdoc2vec', ['name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity', 'content_cosin_similarity']],
    ['bf-cfnn', ['name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity', 'match_score']]
]

formal_feature_name_dict = {'same_biblio_aid': 'MAG Author ID', 'name_similarity': 'Name Similarity',
                            'pub_year_diff': 'Publication Year Gap',
                            'venue_similarity': 'Venue Similarity', 'aff_similarity': 'Affiliation Similarity',
                            'paper_title_abstract_similarity': r'Content Similarity $cf_{jaccard}$',
                            'tfidf_cosin_similarity': r'Content Similarity $cf_{tfidf}$',
                            'content_cosin_similarity': r'Content Similarity $cf_{doc2vec}$',
                            'match_score': r'Content Similarity $cf_{nn}$'}

if __name__ == '__main__':
    # df.to_csv('tmp.tsv', sep='\t', index=False)
    df = shuffle(df)
    print(df.head())

    print('original shape: ', df.shape)
    df = drop_missing_items(df)
    print('after dropping none shape: ', df.shape)

    print('pos_samples_num: ', df[df['same_author'] == 1].shape[0])
    print('neg_samples_num: ', df[df['same_author'] == 0].shape[0])
    # df = down_sample(df)
    # print('after balancing dataset shape: ', df.shape)
    # print('pos_samples_num: ', df[df['same_author'] == 1].shape[0])
    # print('neg_samples_num: ', df[df['same_author'] == 0].shape[0])

    for feature_group_name, feature_names in feature_names_groups:
        for idx, model_switch in enumerate(mode_names):
            df_copy = df.copy(deep=True)
            print('-' * 160)
            print(str(model_switch) + '\tused features:\n', '\t'.join(feature_names))
            Y = np.array(df_copy['same_author'].astype('int'))
            X = df_copy[feature_names]
            # X = scale(X) # TODO scale will improve the performance sightly
            X = np.array(X)

            avg_metrics = []

            # Note we do not using cross validation because the test set is very large
            train_test_index = df_copy['train1_test0_val2'].astype('int')
            indx_split = [
                ([i for i, n in enumerate(train_test_index) if n == 1],
                 [i for i, n in enumerate(train_test_index) if n == 0])
            ]

            for round_idx, (train_index, test_index) in enumerate(indx_split):
                train_X, train_y = X[train_index], Y[train_index]
                test_X, test_y = X[test_index], Y[test_index]

                if len(feature_names) == 1:
                    # Note if only one feature, then no need to use any classifier. 0.5 is the cut-off value
                    pred_y, model = test_X, None
                else:
                    # Note if only multiple features, then using a classifier
                    pred_y, model = use_classifier(train_X, train_y, test_X, model_switch=model_switch)

                    # pred_y, model = use_regression(train_X, train_y, test_X, model_switch=model_switch)
                    # save the model
                    # file_name = 'cached/lagos-and-rf-model.pkl'
                    # pickle.dump(model, open(file_name, 'wb'))
                    importances = model.feature_importances_
                    pprint(list(zip(feature_names, importances)), sep='\t')

                if round_idx == 0 and model is not None:
                    # Note save the model
                    # joblib.dump(model, os.path.join(cached_dir,
                    #                                 'pairwise_and_models/rf-model-with-feature-%s-trained-on-lagos-and-%s-%s-based-dataset.pkl' % (
                    #                                     feature_group_name, underlying_dataset,
                    #                                     'trimmed' if cluster_uniq_author_gt_1 else 'original')))

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
                    plt.ylabel('Feature Contribution', loc='center')  # 'top'
                    plt.xticks(fontsize=8, rotation=10, ha='center')
                    plt.tight_layout()
                    if not cluster_uniq_author_gt_1 and feature_group_name == 'bf-cfnn':
                        plt.savefig(os.path.join(cached_dir, 'feature-contributions.png'), dpi=600)
                        plt.savefig(os.path.join(latex_doc_base_dir, 'figs/feature-contributions.png'), dpi=600)
                    plt.show()

                df_test = pd.DataFrame(df_copy.values[test_index], columns=df_copy.columns.values.tolist())
                df_test[feature_names] = test_X
                df_test['test_y'] = test_y
                df_test['pred_y'] = pred_y
                df_test.to_csv(feature_group_name + '_test_instance_predictions.tsv', sep='\t', index=False)
                metric_dict = calc_metrics(test_y, pred_y)
                metric_tuple = [(m, metric_dict[m]) for m in metric_names]
                # pprint(metric_tuple, pctg=True, sep='\t')
                avg_metrics.append(metric_dict)

            avg_metric_vals = [np.average([item[m] for item in avg_metrics]) for m in metric_names]
            print(metric_names)
            pprint(list(zip(metric_names, avg_metric_vals)), pctg=True, sep='\t')
