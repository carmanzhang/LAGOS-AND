from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

metric_names = ['acc', 'p', 'r', 'f1']


def calc_metrics(test_y, pred_y):
    prob = 0.5
    pred_y_label = [1 if i > prob else 0 for i in pred_y]

    acc = accuracy_score(test_y, pred_y_label)
    p = precision_score(test_y, pred_y_label)
    r = recall_score(test_y, pred_y_label)
    pos_label_f1 = f1_score(test_y, pred_y_label, average='binary')

    return dict(
        zip(metric_names,
            [acc, p, r, pos_label_f1]))
