import os
import sys

import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing.image import img_to_array, load_img

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import model_selection

from LeafNet import LeafNet
import data_visualization, generate_new_features, utils, visualize_filters

from collections import namedtuple

def trainer(x, y):
    classifiers = [
        KNeighborsClassifier(n_jobs=-1, n_neighbors=5),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_jobs=-1, n_estimators=500, warm_start=True, max_depth=6, min_samples_leaf=2, max_features='sqrt'),
        SVC(kernel='linear', C=0.025, probability=True)
    ]

    eclf = VotingClassifier(estimators=[('knn', classifiers[0]),
                                        ('rf', classifiers[2]),
                                        ('svc', classifiers[3])],
                            voting='soft')
    classifiers.append(eclf)
    acc_dict = dict()
    std_dict = dict()
    
    labels = ['KNN', 'DecisionTree', 'RandomForest', 'SVM', "Ensemble"]
    for i, clf in enumerate(classifiers):

        label = labels[i]

        scores = model_selection.cross_val_score(clf, x, y, 
                                                 cv=5, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
              % (scores.mean(), scores.std(), label))
        acc_dict[label] = scores.mean()
        std_dict[label] = scores.std()
    
    sns_acc_dict = {
        'Classifier': acc_dict.keys(),
        'Accuracy': acc_dict.values()
    }

    sns_std_dict = {
        'Classifier': std_dict.keys(),
        'StandardDeviation': std_dict.values()
    }
    
    return eclf, sns_acc_dict, sns_std_dict

def plot_acc_std(sns_acc_dict, sns_std_dict, new_features=True):
    if new_features:
        tar = 'new features'
    else:
        tar = 'numerical features'
    sns_data = pd.DataFrame(sns_acc_dict)
    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Classifier', data=sns_data, color="b")

    plt.xlabel('Accuracy %')
    plt.title('Classifier Accuracy for' + ' ' + tar)
    plt.savefig('Classifier Accuracy for' + ' ' + tar + '.png')
    plt.show()

    sns_data = pd.DataFrame(sns_std_dict)
    sns.set_color_codes("muted")
    sns.barplot(x='StandardDeviation', y='Classifier', data=sns_data, color="g")

    plt.xlabel('StandardDeviation %')
    plt.title('Classifier StandardDeviation for' + ' ' + tar)
    plt.savefig('Classifier StandardDeviation for' + ' ' + tar + '.png')
    plt.show()

def main():
    # read in data
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    train, labels, test, test_ids, classes = utils.encode(train, test)
    
    # data visualization
    data_visualization.viz_pca(train, labels)
    data_visualization.viz_tsne(train, labels)
    
    # train CNN
    leafnet = LeafNet(epochs=1)
    leafnet.fulltrain_fit()
    # visualize CNN arch
    digraph = mx.viz.plot_network(symbol = leafnet.out, save_format='jpg')
    digraph.render()
    # visualize learned filters
    visualize_filters.visualize_filters(leafnet)
    new_features, viz_mod = generate_new_features.generate_new_features(leafnet)
    
    # train models
    new_eclf, sns_acc_dict, sns_std_dict = trainer(new_features, labels)
    # plot results
    plot_acc_std(sns_acc_dict, sns_std_dict, True)
    
    # get new features for test data
    ID, nmr = leafnet.load_numeric_test()
    img = leafnet.readin_image(ID.values.tolist())

    new_test = []
    Batch = namedtuple('Batch', ['data'])
    for ind, val in enumerate(img):
        viz_mod.forward(Batch([mx.nd.array(img[ind].reshape(1, 1, 96, 96)),
                               mx.nd.array(nmr[ind].reshape(1, 192))]))
        features = viz_mod.get_outputs()[0].asnumpy()
        new_test.append(np.array(features))
    new_test = np.array(new_test).reshape(-1, 1500)
    
    # evaluation on new test data
    yhat = new_eclf.fit(new_features, labels).predict_proba(new_test)
    yhat = utils.standardize_output(yhat)
    utils.make_submit_file(test_ids, classes, yhat, '../submission/ensemble_new_features.csv')
    
    # to compare the results, original 192 numerical features are evaluated
    old_eclf, sns_acc_dict, sns_std_dict = trainer(train, labels)
    # plot results
    plot_acc_std(sns_acc_dict, sns_std_dict, False)
    
    # evaluation on old test data
    yhat = old_eclf.fit(train, labels).predict_proba(test)
    yhat = utils.standardize_output(yhat)
    utils.make_submit_file(test_ids, classes, yhat, '../submission/ensemble_old_features.csv')

if __name__ == "__main__":
    main()
