import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import FastICA
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn import svm
import pickle
from sklearn.tree import DecisionTreeClassifier


def extraTrees(datafile):
    data = pd.read_csv(datafile)
    ica = FastICA(n_components=10)
    X = data.drop(columns=['target', 'EEG.Counter', 'EEG.Interpolated'],axis=1)
    y = data['target']
    X_trans = ica.fit_transform(X)
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.25)
    clf = ExtraTreesClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)*100
    prec = precision_score(y_test, y_pred)*100
    f1 = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    pickle.dump(clf, open(datafile+"(Extra_Trees_Model).pkl", "wb"))
    return acc, prec, f1, rec

def RandomForest(datafile):
    data = pd.read_csv(datafile)
    ica = FastICA(n_components=10)
    X = data.drop(columns=['target', 'EEG.Counter', 'EEG.Interpolated'],axis=1)
    y = data['target']
    X_trans = ica.fit_transform(X)
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.25)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)*100
    prec = precision_score(y_test, y_pred)*100
    f1 = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    pickle.dump(clf, open(datafile+"(Random_Forest).pkl", "wb"))
    return acc, prec, f1, rec

def KNNModel(datafile):
    data = pd.read_csv(datafile)
    ica = FastICA(n_components=10)
    X = data.drop(columns=['target', 'EEG.Counter', 'EEG.Interpolated'],axis=1)
    y = data['target']
    X_trans = ica.fit_transform(X)
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.25)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)*100
    prec = precision_score(y_test, y_pred)*100
    f1 = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    pickle.dump(clf, open(datafile+"(KNN).pkl", "wb"))
    return acc, prec, f1, rec

def DecisionTreeModel(datafile):
    data = pd.read_csv(datafile)
    ica = FastICA(n_components=10)
    X = data.drop(columns=['target', 'EEG.Counter', 'EEG.Interpolated'],axis=1)
    y = data['target']
    X_trans = ica.fit_transform(X)
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.25)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)*100
    prec = precision_score(y_test, y_pred)*100
    f1 = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    pickle.dump(clf, open(datafile+"(Decision_Tree_Model).pkl", "wb"))
    return acc, prec, f1, rec

def svm_svc_model(datafile):
    data = pd.read_csv(datafile)
    ica = FastICA(n_components=10)
    X = data.drop(columns=['target', 'EEG.Counter', 'EEG.Interpolated'],axis=1)
    y = data['target']
    X_trans = ica.fit_transform(X)
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.25)
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)*100
    prec = precision_score(y_test, y_pred)*100
    f1 = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    pickle.dump(clf, open(datafile+"(SVM_SVC_Model).pkl", "wb"))
    return acc, prec, f1, rec
    

