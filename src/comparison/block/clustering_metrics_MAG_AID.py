import os

import pandas as pd
from beard import metrics
from mytookit.data_reader import DBReader
from tqdm import tqdm

from myconfig import cached_dir

minimal_uniq_author_in_block = 0

df_blocks = DBReader.tcp_model_cached_read(cached_file_path='xxx',
                                           sql=r'''select block_name, pid_aos, ground_truths, mag_preds, num_unique_author_inblock from and_ds.our_and_dataset_block_test_set_mag_prediction;''',
                                           cached=False)

num_instances = len(df_blocks)
df_blocks = df_blocks[df_blocks['num_unique_author_inblock'] > minimal_uniq_author_in_block]
num_instances1 = len(df_blocks)
print('removed %d instances, enable each block containing more than one unique authors' % (num_instances - num_instances1))

print(df_blocks.shape)


# Note #############################################################################################
# Note test the performance of MAG author identifier
# Note the clustering evaluation can not provide the Random baseline because it can not generate the ``labels_pred``

def data_precision_round(arr, precision=2, pctg=True):
    return [round(x * 100 if pctg else x, precision) for x in arr]


mag_metrics = []
for index, row in tqdm(df_blocks.iterrows(), total=df_blocks.shape[0]):
    block_name, pid_aos, ground_truths, mag_preds, num_unique_author_inblock = row
    # print('block-size: %d' % len(pm_aos))

    # note calculate the paired-F1 and the B3-F1 score
    mag_metrics_b3 = metrics.b3_precision_recall_fscore(labels_true=ground_truths, labels_pred=mag_preds)
    mag_metrics_pairedf = metrics.paired_precision_recall_fscore(labels_true=ground_truths, labels_pred=mag_preds)

    mag_metrics.append([block_name] + data_precision_round(list(mag_metrics_pairedf + mag_metrics_b3)))

# Note using block_name as the index row
result_file = os.path.join(cached_dir, 'clustering-results-lagos-and-MAG-AID.tsv')
columns = ['Block', 'pP', 'pR', 'pF', 'bP', 'bR', 'bF']
df = pd.DataFrame(mag_metrics, columns=columns)
df.to_csv(result_file, sep='\t')
mean_metrics = df._get_numeric_data().mean()
print(mean_metrics)
print(columns)
print('minimal_uniq_author_in_block: ', minimal_uniq_author_in_block,
      data_precision_round(mean_metrics.values.tolist(), pctg=False))
