from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from sklearn.metrics import roc_curve, auc
import sys; sys.path.append('../')
# import os; os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from Tools import *
import DataSetLink as DLSet
from C_Recognize.RecognizeProxy import RecognizeProxy
from A_PreProcess.DataProxy import DataProxy


def predict(rp, pairs):
    result = []
    for each in pairs:
        result.append(rp.get_distance(each[0], each[1]))
    result = np.array(result).ravel()
    return result


def eval_cmc(rp, dp_test, n_round=1, np_ratio=19, batch_size=1):
    score = [0] * (np_ratio + 1)
    for _ in range(n_round):
        # get a batch data and predict
        te_pairs, te_y = dp_test.get_batch(batch_size, np_ratio)
        # print(te_pairs.shape)
        # print(te_y.shape)
        y_pred = predict(rp, te_pairs)

        # calculate score

        # print(y_pred.shape)
        sort_index = np.argsort(y_pred)
        # print(sort_index)
        for i in range(np_ratio, -1, -1):
            score[i] += 1
            if sort_index[i] == 0:
                break
    for i in range(np_ratio + 1):
        score[i] /= n_round
    return score


def eval_roc(rp, dp_test, n_round=1, np_ratio=1, batch_size=10):
    te_pairs, te_y = dp_test.get_batch(-1, np_ratio, True)
    y_pred = predict(rp, te_pairs)

    fpr, tpr, thresholds = roc_curve(te_y.ravel(), y_pred, pos_label=0)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def main():
    # 1. load model
    rp = RecognizeProxy(threshold=1)

    # 2. load data
    dp_test = DataProxy(DLSet.test_data_link, DLSet.test_label_link)

    # 3. eval
    # 3.1 eval ROC
    n_round = 400
    np_ratio = 1
    result = eval_roc(rp, dp_test, n_round, np_ratio)
    store_obj(result, DLSet.roc_res_link % 'faceNet')

    result = load_obj(DLSet.roc_res_link % 'faceNet')
    fpr, tpr, roc_auc = result
    print('* ROC on test set: %0.2f%%' % (float(roc_auc) * 100))
    plot_img(fpr, tpr, 'AUC = {}'.format(roc_auc), 'Boundary', 'Ratio')

    # 3.2 eval PR
    te_pairs, te_y = dp_test.get_batch(-1, 1)
    y_pred = predict(rp, te_pairs)
    te_acc = compute_accuracy(te_y, y_pred, threshold=1.0)
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    # 3.3 eval CMC
    n_round = 400
    np_ratio = 19

    score = eval_cmc(rp, dp_test, n_round, np_ratio)
    store_obj(score, DLSet.cmc_res_link % 'faceNet')

    score = load_obj(DLSet.cmc_res_link % 'faceNet')
    plot_img(np.arange(1, np_ratio + 1 + 1).astype(dtype=np.str), score, 'CMC', 'Boundary', 'Ratio')


if __name__ == '__main__':
    main()

# Record:
# ROC AUC | 95.49%
# Accuracy: 92.59%
