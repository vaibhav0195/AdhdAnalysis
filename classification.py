'''
References:
    ROC curves for kNN: https://stackoverflow.com/questions/52910061/implementing-roc-curves-for-k-nn-machine-learning-algorithm-using-python-and-sci
'''

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
import sys

'''
Calculated and displays an ROC curve for the given model. This is done by getting the y_scores array for the model (confidence in the predictions) and then iunputting that into scikit's roc_curve function. This data is plotted along with a line on y=x. The graph is then annotated and displayed.

Predict_proba returns a 2D array, so we need to select only the values relevent to us (th y scores). Otherwise, we can just use the output directly as the y_score array.
'''
def visualize_roc_curve(X, y, model, title, fname, for_SVC=False):
    if for_SVC:
        y_score = model.decision_function(X)
        fpr, tpr, _ = roc_curve(y, y_score)
    else:
        y_score = model.predict_proba(X)
        y_score = y_score[:, -1]
        fpr, tpr, _ = roc_curve(y, y_score)

    plt.cla()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    plt.savefig(fname, bbox_inches='tight')

    return fpr, tpr

'''
Since this project consists of running largely the same procuedure accross two datasets, I wrote a function to do individual analysis on each dataset. It runs through all of the steps needed for part 1 of the assignment.
'''
def adhd_analysis(data):
    X = data[:, :-1]
    y = data[:, -1]

    #Select hyper-parameters and create model
    C = 25 #cross_validate_C_LR(X, y)
    lr_model = LogisticRegression(penalty='l1', solver='liblinear', C=C).fit(X, y)

    C = 5 #cross_validate_C_SVM(X, y)
    svm_model = LinearSVC(penalty='l1', dual=False, max_iter=100000, C=C).fit(X, y)

    #Select hyperparameter k for kNN model
    k = 27 #cross_validate_k(X, y)
    knn_model = KNeighborsClassifier(n_neighbors=k, weights='uniform').fit(X, y)

    #Create baselines
    random_baseline = DummyClassifier('stratified').fit(X, y)
    majority_baseline = DummyClassifier('most_frequent').fit(X, y)

    #Print model params
    print('LR Params (coef, then intercept):')
    print(lr_model.coef_)
    print(lr_model.intercept_)
    print()

    print('SVM Params (coef, then intercept):')
    print(svm_model.coef_)
    print(svm_model.intercept_)
    print()
    
    #Get the confusion matricies for the models and baselines
    print('LR:')
    print(confusion_matrix(y, lr_model.predict(X)))
    print()

    print('SVM:')
    print(confusion_matrix(y, svm_model.predict(X)))
    print()

    print('kNN:')
    print(confusion_matrix(y, knn_model.predict(X)))
    print()

    print('Random:')
    print(confusion_matrix(y, random_baseline.predict(X)))
    print()

    print('Majority:')
    print(confusion_matrix(y, majority_baseline.predict(X)))
    print()

    #ROC curve, AUC
    fpr, tpr = visualize_roc_curve(X, y, lr_model, 'ROC Curve for Logistic Regression Model', 'roc_lr.png')
    print('LR AUC')
    print(auc(fpr, tpr))
    print()

    fpr, tpr = visualize_roc_curve(X, y, svm_model, 'ROC Curve for Linear SVM Model', 'roc_svm.png', for_SVC=True)
    print('SVM AUC')
    print(auc(fpr, tpr))
    print()

    fpr, tpr = visualize_roc_curve(X, y, knn_model, 'ROC Curve for kNN Model', 'roc_knn.png')
    print('kNN AUC')
    print(auc(fpr, tpr))
    print()

    fpr, tpr = visualize_roc_curve(X, y, random_baseline, 'ROC Curve for Random Baseline', 'roc_rand.png')
    print('Random Baseline AUC')
    print(auc(fpr, tpr))
    print()

    fpr, tpr = visualize_roc_curve(X, y, majority_baseline, 'ROC Curve for Majority Baseline', 'roc_maj.png')
    print('Majority Baseline AUC')
    print(auc(fpr, tpr))
    print()

'''
This runs k-fold cross-validation (where k=10) for determining C, the dividend of the L1 penalty term. The results of each roundn of validation are plotted on a graph, with error bars to represent standard deviation. The approproate C can then be manually determined from the graph.
'''
def cross_validate_C_SVM(X, y):
    #Choose cross-validation parameters
    Cs = [1, 5, 25, 75, 125, 250]
    fold = 10

    #Get means and stdevs for all Cs using k-fold crosss-validation
    kf = KFold(n_splits=fold, shuffle=True)
    mean_errors = []
    std_errors = []
    for C in Cs:
        print(C)
        MSEs = []
        model = LinearSVC(penalty='l1', dual=False, max_iter=100000, C=C).fit(X, y)
        for train, test in kf.split(X):
            model = LinearSVC(penalty='l1', dual=False, max_iter=100000, C=C).fit(X[train], y[train])
            ypred = model.predict(X[test])
            MSEs.append(mean_squared_error(y[test], ypred))            
        MSEs = np.array(MSEs)
        mean_errors.append(MSEs.mean())
        std_errors.append(MSEs.std())

    #Plot the results
    plt.cla()
    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.errorbar(Cs, mean_errors, yerr=std_errors, linewidth=3)
    plt.xlabel('C')
    plt.ylabel('Mean square error')
    plt.title('Cross-validation on C for Linear SVM')
    plt.savefig('Crossval_C_SVM.png', bbox_inches='tight')

def cross_validate_C_LR(X, y):
    #Choose cross-validation parameters
    Cs = [1, 5, 25, 75, 125, 250]
    fold = 10

    #Get means and stdevs for all Cs using k-fold crosss-validation
    kf = KFold(n_splits=fold, shuffle=True)
    mean_errors = []
    std_errors = []
    for C in Cs:
        print(C)
        MSEs = []
        model = LogisticRegression(penalty='l1', solver='liblinear', C=C).fit(X, y)
        for train, test in kf.split(X):
            model = LogisticRegression(penalty='l1', solver='liblinear', C=C).fit(X[train], y[train])
            ypred = model.predict(X[test])
            MSEs.append(mean_squared_error(y[test], ypred))            
        MSEs = np.array(MSEs)
        mean_errors.append(MSEs.mean())
        std_errors.append(MSEs.std())

    #Plot the results
    plt.cla()
    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.errorbar(Cs, mean_errors, yerr=std_errors, linewidth=3)
    plt.xlabel('C')
    plt.ylabel('Mean square error')
    plt.title('Cross-validation on C for Logistic Regression')
    plt.savefig('Crossval_C_LR.png', bbox_inches='tight')

'''
This runs k-fold cross-validation (where k=10) for determining kNN's k parameter. The results of each roundn of validation are plotted on a graph, with error bars to represent standard deviation. The approproate kNN k value can then be manually determined from the graph.
'''
def cross_validate_k(X, y):
    #Choose cross-validation parameters
    ks = [1, 3, 9, 27, 35, 51, 81]
    fold = 10

    #Get means and stdevs for all qs using k-fold crosss-validation
    kf = KFold(n_splits=fold, shuffle=True)
    mean_errors = []
    std_errors = []
    for k in ks:
        print(k)
        MSEs = []
        model = KNeighborsClassifier(n_neighbors=k, weights='uniform').fit(X, y)
        for train, test in kf.split(X):
            model = KNeighborsClassifier(n_neighbors=k, weights='uniform').fit(X[train], y[train])
            ypred = model.predict(X[test])
            MSEs.append(mean_squared_error(y[test], ypred))
        MSEs = np.array(MSEs)
        mean_errors.append(MSEs.mean())
        std_errors.append(MSEs.std())
    
    plt.cla()
    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.errorbar(ks, mean_errors, yerr=std_errors, linewidth=3)
    plt.xlabel('k')
    plt.ylabel('Mean square error')
    plt.title('Cross-validation on k')
    plt.savefig('Crossval_k.png', bbox_inches='tight')

'''
This function loads data from the given file as a numpy array. It is sued to collect hte training data to be used for the models.
'''
def load_data(adhd_file, adhdPart_file):
    adhd = pd.read_csv(adhd_file, header=0)
    del adhd['filename']
    indicator = [1 for _ in range(len(adhd))]
    adhd['indicator'] = indicator

    adhdPart = pd.read_csv(adhdPart_file, header=0)
    del adhdPart['filename']
    indicator = [0 for _ in range(len(adhdPart))]
    adhdPart['indicator'] = indicator

    return adhd.append(adhdPart).to_numpy()

'''
This functions starts the program. It runs the full analysis for the data set
'''
def main():
    adhd_file = sys.argv[1]
    adhdPart_file = sys.argv[2]
    data = load_data(adhd_file, adhdPart_file)
    adhd_analysis(data)

if __name__ == '__main__':
    main()
