#Install
!pip install catboost
!pip install xlsxwriter
!pip install catboost
!pip install scikit-optimize
!pip install xgboost
!pip install lightgbm
!pip install openpyxl
#
#
#Import
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, accuracy_score,precision_score, average_precision_score, make_scorer, f1_score,recall_score,roc_auc_score,balanced_accuracy_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import *
from threading import Thread
from openpyxl import load_workbook
import pandas as pd
import os
from sklearn.preprocessing import LabelBinarizer
from pandas import read_csv
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import xlsxwriter
from pathlib import Path
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
#
#
#Data file path
data_path = 'shallue_all_local.csv'
data = pd.read_csv(data_path, sep = ",")
#
#
# Definition of input and label in the tabular format required by scikit-learn
data_input = data.copy()
label = data_input.pop(data_input.columns[len(data_input.columns)-1])
X = data_input.values #limited size for quick testing
y = label.values #limited size for quick testing
#
#
#normalization
norm_data = data_input.copy()
norm_data = norm_data.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)
X_ = norm_data.values #limited size for quick testing
#
#
#label binário
lb = LabelBinarizer()
y = lb.fit_transform(label)
y = y.reshape(-1) #limited size for quick testing
#
#
#Definition of the experiment
def experiment(model_name, model, params, X_, y, book_name='Local', pos_label=1):
    # Folder path to save the file
    data_path = 'Classicos/Local'
  
    # File name to save the results
    results_file = os.path.join(data_path, model_name+ "_" + book_name + '.xlsx')

    # Configure the cross-validation procedure
    cv_outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

    results = []

    for i, (train_ix, test_ix) in enumerate(cv_outer.split(X_, y)):
        # split data
        X_train, X_test = X_[train_ix, :], X_[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        # Configure the cross-validation procedure
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

        # Define search
        search = BayesSearchCV(model, params, scoring='accuracy', cv=cv_inner, n_iter=10, refit=True, random_state=1, n_jobs=1)

        # Run search
        result = search.fit(X_train, y_train)

        # Get the best performing model fit on the whole training set
        best_model = result.best_estimator_

        # Evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)

        # Evaluate the model
        acc = accuracy_score(y_test, yhat)
        prec = precision_score(y_test, yhat, pos_label=pos_label, average='macro', zero_division=1)
        rec = recall_score(y_test, yhat, pos_label=pos_label)
        f1 = f1_score(y_test, yhat, pos_label=pos_label)

        # Store the result
        results.append([model_name, i, acc, rec, prec, f1, result.best_score_, result.best_params_])

        # Report progress
        #print(f"{model_name} {i} > acc={acc:.3f}, est={result.best_score_:.3f}, cfg={result.best_params_}")
        f = open('saida.txt', 'a')
        f.write(f"{model_name} {i} > acc={acc:.3f}, est={result.best_score_:.3f}, cfg={result.best_params_}\n")
        f.close()

    # Summarize the estimated performance of the model
    mean_acc = sum(r[2] for r in results) / len(results)
    mean_rec = sum(r[3] for r in results) / len(results)
    mean_prec = sum(r[4] for r in results) / len(results)
    mean_f1 = sum(r[5] for r in results) / len(results)

    # Save results to file
    df = pd.DataFrame(results, columns=['model', 'run', 'acc', 'rec', 'prec', 'f1', 'best_score', 'best_params'])
    df.to_excel(results_file, index=False)

#
#
# Definition of models and parameters
model_params = {
          'lr': {'model': LogisticRegression(),
                'params': {
                            'C': Real(1e-4, 1e4, prior='log-uniform'),
                            'fit_intercept': Categorical([True, False]),
                            'solver': Categorical(['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
                            'max_iter':[500],
                            'random_state': [1]}},

          'knn': {'model': KNeighborsClassifier(),
                  'params': {
                            'n_neighbors': Integer(1, 50),
                            'weights': Categorical(['uniform', 'distance']),
                            'algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),
                            'p': Integer(1, 5)}},

          'nb': {'model': GaussianNB(),
                'params': {
                         'var_smoothing': Real(1e-10, 1e-1, prior='log-uniform')}},

          'dt': {'model': DecisionTreeClassifier(),
                'params': {
                            'criterion': Categorical(['gini', 'entropy']),
                            'splitter': Categorical(['best', 'random']),
                            'max_depth': Integer(3, 30),
                            'min_samples_split': Integer(2, 10),
                            'min_samples_leaf': Integer(1, 10),
                            'max_features': Real(0.1, 1.0, prior='uniform'),
                            'random_state': [1]}},

          'svm': {'model': SVC(),
                  'params': {
                            'C': Real(2**-5, 2**5, prior='log-uniform'),
                            'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
                            'degree': Integer(2, 5),  # Somente relevante para o kernel 'poly'
                            'coef0': Real(0, 1),      # Relevante para os kernels 'poly' e 'sigmoid'
                            'gamma': Real(2**-9, 2**1, prior='log-uniform'),
                            'random_state': [1]}},

          'gpc': {'model': GaussianProcessClassifier(),
                  'params': {
                            'optimizer': Categorical(['fmin_l_bfgs_b', None]),
                            'n_restarts_optimizer': Integer(0, 10),
                            'max_iter_predict': [500],
                            'random_state': [1]}},

          'mlp': {'model': MLPClassifier(),
                  'params': {
                            'hidden_layer_sizes': Integer(10,100),
                            'activation': Categorical(['logistic', 'tanh', 'relu']),
                            'solver': Categorical(['sgd', 'adam']),
                            'max_iter': [5000],
                            'random_state': [1]}},

          'ridge': {'model': RidgeClassifier(),
                    'params': {
                                'alpha': Real(1e-4, 1e4, prior='log-uniform'),
                                'fit_intercept': Categorical([True, False]),
                                'solver': Categorical(['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
                                'random_state': [1]}},

          'rf': {'model': RandomForestClassifier(),
                'params': {
                          'n_estimators': Integer(10, 500),
                          'criterion': Categorical(['gini', 'entropy']),
                         'max_depth': Integer(3, 30),
                          'random_state': [1]}},

          'qda': {'model': QuadraticDiscriminantAnalysis(),
                  'params': {
                            'reg_param': Real(0, 1, prior='uniform'),
                            'store_covariance': Categorical([True, False]),
                            'tol': Real(1e-5, 1e-1, prior='log-uniform')}},

          'ada': {'model': AdaBoostClassifier(),
                  'params': {
                            'n_estimators': Integer(10, 500),
                            'learning_rate': Real(1e-3, 1, prior='log-uniform'),
                           'algorithm': Categorical(['SAMME', 'SAMME.R']),
                           'random_state': [1]}},

          'gbc': {'model': GradientBoostingClassifier(),
                  'params': {    
                            'n_estimators': Integer(10, 500),
                            'learning_rate': Real(1e-3, 1, prior='log-uniform'),
                            'max_depth': Integer(3, 10),
                            'random_state': [1]}},

          'lda': {'model': LinearDiscriminantAnalysis(),
                 'params': {
                           'solver': Categorical(['svd', 'lsqr', 'eigen']),
                           'shrinkage': Real(0, 1, prior='uniform'),
                           'tol': Real(1e-6, 1e-4, prior='log-uniform')}},

          'et': {'model': ExtraTreesClassifier(),
                 'params': {
                         'n_estimators': Integer(10, 500),
                         'criterion': Categorical(['gini', 'entropy']),
                         'max_depth': Integer(3, 30)}},

          'xgboost': {'model': XGBClassifier(),
                      'params': {
                                'learning_rate': Real(0.01, 0.3, prior='uniform'),
                                'n_estimators': Integer(50, 500),
                                'max_depth': Integer(3, 10),
                                'gamma': Real(0, 1, prior='uniform'),
                                }},
                
          'lightgbm': {'model': LGBMClassifier(),
                      'params': {
                                'learning_rate': Real(1e-3, 1, prior='log-uniform'),
                                'n_estimators': Integer(10, 500),
                                'num_leaves': Integer(2, 100),
                                'max_depth': Integer(3, 10)}}

          'catboost': {'model': CatBoostClassifier(verbose=0),
                     'params': {
                               'learning_rate': Real(1e-3, 1, prior='log-uniform'),
                               'iterations': Integer(10, 500),
                               'depth': Integer(3, 10),
                               'l2_leaf_reg': Real(1, 10, prior='uniform'),
                               'border_count': Integer(1, 255),
                               'bagging_temperature': Real(0, 1, prior='uniform'),
                               'random_strength': Real(1e-9, 10, prior='log-uniform')}}
                   
}
#
#
# The experiments for each algorithm will be executed concurrently
threads = []

# Starts a thread for each ML algorithm
# The experiments for each algorithm will be executed concurrently
for model_name, mp in model_params.items():
  exp = Thread(target=experiment,args=[model_name, mp['model'],mp['params'], X_, y])
  exp.start() #inicia thread
  threads.append(exp) #adiciona na lista para salvar a referencia da thread

for i in range (len(threads)):
  threads[i].join() #retoma o resultado para o programa chamador
#
#
# The experiments for each algorithm will be executed in line
for model_name, mp in model_params.items():
  experiment(model_name, mp['model'],mp['params'], X_, y)
