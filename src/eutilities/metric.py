from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

metric_names = ['acc', 'p', 'r', 'f1', 'macro_f1', 'macro_weighted_f1', 'micro_f1', 'auc']


def calc_metrics(test_y, pred_y, average='macro_f1', search_cut_off=True):
    prob = 0.5
    pred_y_label = [1 if i > prob else 0 for i in pred_y]

    acc = accuracy_score(test_y, pred_y_label)
    p = precision_score(test_y, pred_y_label)
    r = recall_score(test_y, pred_y_label)

    macro_f1 = f1_score(test_y, pred_y_label, average='macro')
    macro_weighted_f1 = f1_score(test_y, pred_y_label, average='weighted')
    micro_f1 = f1_score(test_y, pred_y_label, average='micro')

    pos_label_f1 = f1_score(test_y, pred_y_label, average='binary')
    rocauc = roc_auc_score(y_true=test_y, y_score=pred_y)
    # neg_label_f1 = f1_score(test_y, pred_y_label, pos_label=0, average='binary')
    # print(confusion_matrix(test_y, pred_y_label))
    return dict(
        zip(metric_names,
            [acc, p, r, pos_label_f1, macro_f1, macro_weighted_f1, micro_f1, rocauc]))
