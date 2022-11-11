import os
import sys

sys.path.append('../../')
import joblib
import numpy as np
from beard import metrics
from mytookit.data_reader import DBReader
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from myconfig import cached_dir, cli_args, tuned_best_cluster_setting

''' Note
This script aims to evaluate some semi-supervised learning baseline methods. To do this, We 
1) trained pairwise disambiguation models on the TRAINING set;
2) use these model to search the most effect parameters on the DEV set;
3) last, using the pairwise AND models and the optimal clustering parameters to perform clustering on the TEST set.

Note that we also conduct experiments on the slimmer LAGOS-AND dataset created by removing those blocks containing only one author   
'''

# Note hyper-parameters ################################################################################
underlying_dataset = 'pairwise-gold-standard'
# cluster_uniq_author_gt_1 = False
cluster_uniq_author_gt_1 = True
which_model = cli_args.which_model
print(which_model)
HAC_distance_threshold_trials = range(1, 11, 1)

# Note step 1 ##########################################################################################
# Note load the trained model, which is trained on the block-based LAGOS-AND training set
cached_file_base_dir = os.path.join(cached_dir, 'cluster_feature')
available_model_names = ['name', 'bf', 'bf-cfjaccard', 'bf-cftfidf', 'bf-cfdoc2vec', 'bf-cfnn']
available_feature_masks = [[0], [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 5], [0, 1, 2, 3, 6], [0, 1, 2, 3, 7]]

# Note load all the pairwise AND models
available_models = []
for n in available_model_names:
    model_path = os.path.join(cached_dir,
                              'pairwise_and_models/rf-model-with-feature-%s-trained-on-lagos-and-%s-%s-based-dataset.pkl' % (
                                  n, underlying_dataset, 'trimmed' if cluster_uniq_author_gt_1 else 'original'))
    print(model_path)
    available_models.append(joblib.load(model_path))

current_model = available_model_names[which_model]
ml_model = available_models[which_model]
feature_mask = available_feature_masks[which_model]
print('current_model: ', current_model)

# Note step 2 ##########################################################################################
# Note load the DEV and TEST set
df_blocks = DBReader.tcp_model_cached_read(cached_file_path=os.path.join(cached_dir, 'lagos-and-block-info.pkl'),
                                           sql=r'''select block_name, pid_aos, ground_truths, mag_preds, seg, train1_test0_val2, num_unique_author_inblock, num_citaion_in_block from and_ds.our_and_dataset_block_with_block_info;''',
                                           cached=True)
print(df_blocks.shape)

# Note this is very important here as it will greatly reduce the size of the dataset
if cluster_uniq_author_gt_1:
    num_instances = len(df_blocks)
    df_blocks = df_blocks[df_blocks['num_unique_author_inblock'] > 1]
    num_instances1 = len(df_blocks)
    print('removed %d instances, enable each block containing more than one unique authors' % (num_instances - num_instances1))
# del df_blocks['num_unique_author_inblock'], df_blocks['num_citaion_in_block']

df_train_blocks = df_blocks[df_blocks['train1_test0_val2'] == 1]
df_test_blocks = df_blocks[df_blocks['train1_test0_val2'] == 0]
df_val_blocks = df_blocks[df_blocks['train1_test0_val2'] == 2]
print('train/val/test block sizes', df_train_blocks.shape, df_val_blocks.shape, df_test_blocks.shape)
del df_blocks, df_train_blocks['train1_test0_val2'], df_test_blocks['train1_test0_val2'], df_val_blocks['train1_test0_val2']


# Note step 3 ##########################################################################################
# Note eval on the DEV set, trying to find the best clustering parameters
def merge_feature(five_fast_feature, tfidf_feature, dov2vec_feature, matching_feature):
    # set1==set2 compares for equality of each element in both the sets,
    # and evaluates to true if and only if both the sets are exactly same.
    assert five_fast_feature.keys() == tfidf_feature.keys() == dov2vec_feature.keys() == matching_feature.keys()
    avg_feature_values = []
    merged_feature_map = {}
    for k in matching_feature.keys():
        fv1, fv2, fv3, (fv41, fv42) = five_fast_feature[k], tfidf_feature[k], dov2vec_feature[k], matching_feature[k]
        assert fv1.shape[:2] == fv2.shape[:2] == fv3.shape[:2] == fv41.shape[:2]
        num_authors = fv1.shape[0]
        #  Note all these are numpy array while permuted the feature order,
        #  Note making it aligned with the order of the original feature training the AND models

        # feature_names_groups = [
        #     # ['rand', ['random']],
        #     # ['magaid', ['same_biblio_aid']],
        #     ['name', ['name_similarity']],
        #     ['bf', ['name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity']],
        #     ['bf-cfjaccard',
        #      ['name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity', 'paper_title_abstract_similarity']],
        #     ['bf-cftfidf', ['name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity', 'tfidf_cosin_similarity']],
        #     ['bf-cfdoc2vec', ['name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity', 'content_cosin_similarity']],
        #     ['bf-cfnn', ['name_similarity', 'pub_year_diff', 'venue_similarity', 'aff_similarity', 'match_score']]
        # ]

        # fv41[fv41 <= 0.5] = 0

        # Note convert the 2D numpy to a symmetry 2D matrix
        fv41 = (fv41 + fv41.T) / 2

        tmp_concat_feature = np.concatenate(
            (np.expand_dims(fv1[:, :, 0], axis=2),  # name_similarity, 0
             np.expand_dims(fv1[:, :, 1], axis=2),  # pub_year_diff, 1
             np.expand_dims(fv1[:, :, 3], axis=2),  # venue_similarity, 2
             np.expand_dims(fv1[:, :, 4], axis=2),  # aff_similarity, 3
             np.expand_dims(fv1[:, :, 2], axis=2),  # paper_title_abstract_similarity, 4
             np.expand_dims(fv2, axis=2),  # tfidf, 5
             np.expand_dims(fv3, axis=2),  # dov2vec, 6
             np.expand_dims(fv41, axis=2),  # nn1 sigmoid, 7
             ),
            axis=2)

        tmp_avg_feature_value = [[num_authors * num_authors, np.sum(tmp_concat_feature[:, :, i].view().reshape(-1))] for i in
                                 range(0, 8, 1)]
        avg_feature_values.append(tmp_avg_feature_value)

        # print(tmp_concat_feature.shape)
        merged_feature_map[k] = tmp_concat_feature

    # avg_feature_values = np.array(avg_feature_values)
    # avg_feature_values = [np.sum(avg_feature_values[:, i, 1]) / np.sum(avg_feature_values[:, i, 0]) for i in range(0, 8, 1)]
    # print('feature average values: ', avg_feature_values)
    # feature average values: [0.8576142289367379, 5.90870462549071, 0.17193027950163528, 0.32441071932990906, 0.08016590954082886, 0.13715612273098102, 0.2845889716223027, 0.7829814895889327, 0.8295835544720577]
    return merged_feature_map


def data_precision_round(arr, precision=2, pctg=True):
    return [round(x * 100 if pctg else x, precision) for x in arr]


def clustering_over_input_blocks(cluster_algo, input_df_blocks):
    all_clustering_metrics = []
    all_clustering_predictions = {}
    segments = range(0, 10, 1)
    for seg in segments:
        # Note loading DEV block information
        df_seg = input_df_blocks[input_df_blocks['seg'] == seg]
        del df_seg['seg']

        # Note loading the cached feature data
        merged_feature_path = os.path.join(cached_file_base_dir, 'merged_features-gold-standard-%d.pkl' % seg)
        # merged_feature_path = os.path.join(cached_dir, 'temp/merged_features-%d.pkl' % seg)
        if os.path.exists(merged_feature_path):
            merged_feature_map = joblib.load(merged_feature_path)
        else:
            # Note consolidating the features into one feature file
            five_fast_feature = joblib.load(os.path.join(cached_file_base_dir, 'five-fast-features-%d.pkl' % seg))
            tfidf_feature = joblib.load(os.path.join(cached_file_base_dir, 'tfidf-feature-%d.pkl' % seg))
            dov2vec_feature = joblib.load(os.path.join(cached_file_base_dir, 'doc2vec-feature-%d.pkl' % seg))
            # matching_feature = joblib.load(os.path.join(cached_file_base_dir, 'matching-features-%d.pkl' % seg))
            # matching_feature = joblib.load(os.path.join(cached_file_base_dir, 'matching-features-glove840B-%d.pkl' % seg))
            matching_feature = joblib.load(os.path.join(cached_file_base_dir,
                                                        'matching-features-glove840B-%d-with-model-trained-on-%s.pkl' % (
                                                            seg, underlying_dataset)))

            print(len(five_fast_feature), len(dov2vec_feature), len(matching_feature))
            merged_feature_map = merge_feature(five_fast_feature, tfidf_feature, dov2vec_feature, matching_feature)
            del five_fast_feature, tfidf_feature, dov2vec_feature, matching_feature
            joblib.dump(merged_feature_map, merged_feature_path)

        for ij, row in tqdm(df_seg.iterrows(), total=df_seg.shape[0]):
            block_name, pid_aos, ground_truths, mag_preds, num_unique_author_inblock, num_citaiton_in_block = row
            num_authors = len(pid_aos)

            block_feature_matrix = merged_feature_map[block_name]
            assert block_feature_matrix.shape[:2] == (num_authors, num_authors)

            # Note squared predictions based on the giving features
            block_feature_matrix = block_feature_matrix[:, :, feature_mask]
            block_flatten_feature_vector = block_feature_matrix.view().reshape(-1, block_feature_matrix.shape[-1])
            block_flatten_predictions = ml_model.predict_proba(block_flatten_feature_vector)[:, 1]

            # ground_truths_1D = np.array([[1 if aa == bb else 0 for aa in ground_truths] for bb in ground_truths]).reshape(-1)
            # for k, _ in enumerate(feature_mask):
            #     print(k, stats.spearmanr(block_flatten_feature_vector[:, k], ground_truths_1D)[0])
            # print(stats.spearmanr(block_flatten_predictions, ground_truths_1D)[0])

            block_flatten_predictions = 1 - block_flatten_predictions  # convert to distance matrix
            block_squared_predictions = block_flatten_predictions.reshape(num_authors, num_authors)

            # block_squared_predictions = 1 - block_feature_matrix[:, :, 8]

            # Note clustering on the block_squared_predictions using DBSCAN
            # cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')  # , n_jobs=-1 ``-1`` means using all processors
            # cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, affinity='precomputed',
            #                                   linkage='single')

            # Note the input of clustering algorithm is the distance matrix
            cluster_labels = cluster_algo.fit_predict(X=block_squared_predictions)
            all_clustering_predictions[block_name] = [cluster_labels, ground_truths]

            # print(block_name, len(ground_truths), len(set(ground_truths)), cluster_labels)

            # Note compare the cluster_labels with the ground truth and calculate the metrics
            block_metrics_b3 = metrics.b3_precision_recall_fscore(labels_true=ground_truths, labels_pred=cluster_labels)
            block_metrics_pairwisef = metrics.paired_precision_recall_fscore(labels_true=ground_truths,
                                                                             labels_pred=cluster_labels)
            all_clustering_metrics.append(
                [block_name] + data_precision_round(list(block_metrics_b3 + block_metrics_pairwisef), pctg=False))

            # if np.random.random() < 0.001:
            #     print('intermediate results: ', np.array([n[1:] for n in all_clustering_metrics]).mean(axis=0))

    return all_clustering_metrics, all_clustering_predictions


if tuned_best_cluster_setting is None:
    print('evaluating ...')
    best_cluster_setting = None
    best_metric = -1
    metric_tendencies = []
    for cluster_setting in HAC_distance_threshold_trials:
        distance_threshold = 0.2 + cluster_setting * 0.01

        cluster_algo = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold,
                                               affinity='precomputed', linkage='single')
        all_clustering_metrics, all_clustering_predictions = clustering_over_input_blocks(cluster_algo, df_val_blocks)

        # Note computer average metrics
        avg_metrics = np.array([n[1:] for n in all_clustering_metrics]).mean(axis=0)
        print(avg_metrics)
        bp, br, bf, pp, pr, pf = avg_metrics
        metric_tendencies.append([metric_tendencies, bf])
        if best_metric < bf:
            print('updated the best clustering B3-F1 metric from %f to %f, and the the corresponding clustering setting is %f' % (
                best_metric, bf, distance_threshold))
            best_metric = bf
            best_cluster_setting = distance_threshold

    # plt.plot([n[0] for n in metric_tendencies], [n[1] for n in metric_tendencies])
    # plt.title(current_model)
    # plt.savefig(os.path.join(cached_dir, 'cluster_parameter_tuning/%s.png' % current_model), dpi=600)
    # plt.show()

    print('the best_cluster_setting for current_model: %s is %f' % (current_model, best_cluster_setting))
    tuned_best_cluster_setting = best_cluster_setting

# Note step 3 ##########################################################################################
# Note clustering on the the block-based LAGOS-AND test set and calculating the metrics
print('evaluating on the test set using the parameter %f ...' % tuned_best_cluster_setting)
tuned_cluster_algo = AgglomerativeClustering(n_clusters=None, distance_threshold=tuned_best_cluster_setting,
                                             affinity='precomputed', linkage='single')
all_clustering_metrics, all_clustering_predictions = clustering_over_input_blocks(tuned_cluster_algo,
                                                                                  # Note must use the TEST set
                                                                                  df_test_blocks)
# Note computer average metrics
avg_metrics = np.array([n[1:] for n in all_clustering_metrics]).mean(axis=0)
print('avg_metrics: ', avg_metrics)
bp, br, bf, pp, pr, pf = avg_metrics

joblib.dump([avg_metrics, all_clustering_metrics, all_clustering_predictions],
            os.path.join(cached_dir, 'cluster_metrics/all-metrics-predictions-%s-%f.pkl' %
                         (current_model, tuned_best_cluster_setting))
            )
