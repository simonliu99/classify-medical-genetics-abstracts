print('.IMPORTING PACKAGES')
from time import time
mark = time()

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from joblib import dump, load
print('..DONE in %.2f s' % (time() - mark))


def import_folds(n_folds, sfx):
    train_fn = '../tmp/%dfold/train-f%s-%s.pickle' % (n_folds, '%d', sfx)
    valid_fn = '../tmp/%dfold/valid-f%s-%s.pickle' % (n_folds, '%d', sfx)
    test_fn = '../tmp/%dfold/test-%s.pickle' % (n_folds, sfx)
    train_folds = [pd.read_pickle(train_fn % n) for n in range(n_folds)]
    valid_folds = [pd.read_pickle(valid_fn % n) for n in range(n_folds)]
    test = pd.read_pickle(test_fn)
    return train_folds, valid_folds, test


def binarize(train, valid):
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(train)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        train_bin = pd.DataFrame.sparse.from_spmatrix(mlb.transform(train), columns=mlb.classes_)
        valid_bin = pd.DataFrame.sparse.from_spmatrix(mlb.transform(valid), columns=mlb.classes_)
    return train_bin, valid_bin, mlb


def binarize_folds(train_folds, valid_folds, ngram=2):
    train_bins, valid_bins, mlbs = [], [], []
    for n in range(len(train_folds)):
        train_bin, valid_bin, mlb = binarize(train_folds[n]['Stemmed %dngram' % ngram],
                                             valid_folds[n]['Stemmed %dngram' % ngram])
        train_bins.append(train_bin)
        valid_bins.append(valid_bin)
        mlbs.append(mlb)
    return train_bins, valid_bins, mlbs


def label(train, valid, config='multi'):
    if config == 'multi':
        return train, valid
    elif config == 'bin':
        return [1 if v == 1 else 0 for v in train], [1 if v == 1 else 0 for v in valid]
    elif config == 'bin136':
        return [1 if v in [1,3,6] else 0 for v in train], [1 if v in [1,3,6] else 0 for v in valid]


def label_folds(train_folds, valid_folds, config='multi'):
    train_labels, valid_labels = [], []
    for n in range(len(train_folds)):
        train_label, valid_label = label(train_folds[n].Category.tolist(),
                                         valid_folds[n].Category.tolist(),
                                         config=config)
        train_labels.append(train_label)
        valid_labels.append(valid_label)
    return train_labels, valid_labels


def train_rf(X_train, y_train, X_valid, y_valid, output_dir, ngram, fold):
    model_fn = 'models/k%d.joblib' % fold
    probs_fn = 'probs/k%d.npy' % fold
    clf = RandomForestClassifier(n_jobs=-1, random_state=420, max_depth=X_train.shape[1]/2, class_weight='balanced')
    clf.fit(X_train, y_train)
    dump(clf, output_dir % model_fn)
    with open(output_dir % probs_fn, 'wb') as f:
        np.save(f, clf.predict_proba(X_valid))


def train_rf_folds(X_trains, y_trains, X_valids, y_valids, output_dir, ngram):
    for n in range(len(X_trains)):
        train_mark = time()
        train_rf(X_trains[n], y_trains[n], X_valids[n], y_valids[n], output_dir, ngram, n)
        print('...FOLD %d DONE IN %.2f s' % (n, time() - train_mark))


def main(ngram, n_folds, config, sfx):
    print('.USING SETTINGS %dngram %d-fold %s %s' % (ngram, n_folds, config, sfx))
    output_dir = '../output/rf%d-%dfold-%s/%s' % (ngram, n_folds, config, '%s')
    Path(output_dir % 'models').mkdir(parents=True, exist_ok=True)
    Path(output_dir % 'probs').mkdir(parents=True, exist_ok=True)
    Path(output_dir % 'binarizers').mkdir(parents=True, exist_ok=True)

    print('.IMPORTING DATA')
    mark = time()
    train_folds, valid_folds, test = import_folds(n_folds, sfx)
    print('..DONE in %.2f s' % (time() - mark))

    # binarize
    print('.BINARIZING')
    mark = time()
    X_trains, X_valids, mlbs = binarize_folds(train_folds, valid_folds, ngram=ngram)
    for n, mlb in enumerate(mlbs):
        dump(mlb, output_dir % ('binarizers/k%d.joblib' % n))
    print('..DONE in %.2f s' % (time() - mark))

    # labels
    print('.CONFIG LABELS')
    mark = time()
    y_trains, y_valids = label_folds(train_folds, valid_folds, config=config)
    print('..DONE in %.2f s' % (time() - mark))

    # training
    print('.TRAINING')
    mark = time()
    train_rf_folds(X_trains, y_trains, X_valids, y_valids, output_dir, ngram)
    print('..DONE in %.2f s' % (time() - mark))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random forest classifier with k-fold cross validation for PubMed abstracts project')
    parser.add_argument('-n', metavar='NGRAMS', type=int, nargs=1, help='ngrams used for feature generation')
    parser.add_argument('-f', metavar='FOLDS', type=int, nargs=1, help='number of folds in cross validation')
    parser.add_argument('-c', metavar='CONFIG', nargs=1, help='label configuration (bin, bin136, multi)')
    args = vars(parser.parse_args())

    ngram = args['n'][0]
    n_folds = args['f'][0]
    config = args['c'][0]
    sfx = '8.26.21-0.9-10.20.21'

    main(ngram, n_folds, config, sfx)
