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

def get_categories():
    all_categories =  'nwords,Admiration/Awe_GALC,Amusement_GALC,Anger_GALC,Anxiety_GALC,Beingtouched_GALC,Boredom_GALC,Compassion_GALC,Contempt_GALC,Contentment_GALC,Desperation_GALC,Disappointment_GALC,Disgust_GALC,Dissatisfaction_GALC,Envy_GALC,Fear_GALC,Feelinglove_GALC,Gratitude_GALC,Guilt_GALC,Happiness_GALC,Hatred_GALC,Hope_GALC,Humility_GALC,Interest/Enthusiasm_GALC,Irritation_GALC,Jealousy_GALC,Joy_GALC,Longing_GALC,Lust_GALC,Pleasure/Enjoyment_GALC,Pride_GALC,Relaxation/Serenity_GALC,Relief_GALC,Sadness_GALC,Shame_GALC,Surprise_GALC,Tension/Stress_GALC,Positive_GALC,Negative_GALC,Anger_EmoLex,Anticipation_EmoLex,Disgust_EmoLex,Fear_EmoLex,Joy_EmoLex,Negative_EmoLex,Positive_EmoLex,Sadness_EmoLex,Surprise_EmoLex,Trust_EmoLex,Valence,Valence_nwords,Arousal,Arousal_nwords,Dominance,Dominance_nwords,pleasantness,attention,sensitivity,aptitude,polarity,vader_negative,vader_neutral,vader_positive,vader_compound,hu_liu_pos_perc,hu_liu_neg_perc,hu_liu_pos_nwords,hu_liu_neg_nwords,hu_liu_prop,Positiv_GI,Negativ_GI,Pstv_GI,Affil_GI,Ngtv_GI,Hostile_GI,Strong_GI,Power_GI,Weak_GI,Submit_GI,Active_GI,Passive_GI,Pleasur_GI,Pain_GI,Feel_GI,Arousal_GI,Emot_GI,Virtue_GI,Vice_GI,Ovrst_GI,Undrst_GI,Academ_GI,Doctrin_GI,Econ_2_GI,Exch_GI,Econ_GI,Exprsv_GI,Legal_GI,Milit_GI,Polit_2_GI,Polit_GI,Relig_GI,Role_GI,Coll_GI,Work_GI,Ritual_GI,Socrel_GI,Race_GI,Kin_2_GI,Male_GI,Female_GI,Nonadlt_GI,Hu_GI,Ani_GI,Place_GI,Social_GI,Region_GI,Route_GI,Aquatic_GI,Land_GI,Sky_GI,Object_GI,Tool_GI,Food_GI,Vehicle_GI,Bldgpt_GI,Comnobj_GI,Natobj_GI,Bodypt_GI,Comform_GI,Com_GI,Say_GI,Need_GI,Goal_GI,Try_GI,Means_GI,Persist_GI,Complet_GI,Fail_GI,Natrpro_GI,Begin_GI,Vary_GI,Increas_GI,Decreas_GI,Finish_GI,Stay_GI,Rise_GI,Exert_GI,Fetch_GI,Travel_GI,Fall_GI,Think_GI,Know_GI,Causal_GI,Ought_GI,Perceiv_GI,Compare_GI,Eval_2_GI,Eval_GI,Solve_GI,Abs_2_GI,Abs_GI,Quality_GI,Quan_GI,Numb_GI,Ord_GI,Card_GI,Freq_GI,Dist_GI,Time_2_GI,Time_GI,Space_GI,Pos_GI,Dim_GI,Rel_GI,Color_GI,Self_GI,Our_GI,You_GI,Name_GI,Yes_GI,No_GI,Negate_GI,Intrj_GI,Iav_GI,Dav_GI,Sv_GI,Ipadj_GI,Indadj_GI,Powgain_Lasswell,Powloss_Lasswell,Powends_Lasswell,Powaren_Lasswell,Powcon_Lasswell,Powcoop_Lasswell,Powaupt_Lasswell,Powpt_Lasswell,Powdoct_Lasswell,Powauth_Lasswell,Powoth_Lasswell,Powtot_Lasswell,Rcethic_Lasswell,Rcrelig_Lasswell,Rcgain_Lasswell,Rcloss_Lasswell,Rcends_Lasswell,Rctot_Lasswell,Rspgain_Lasswell,Rsploss_Lasswell,Rspoth_Lasswell,Rsptot_Lasswell,Affgain_Lasswell,Affloss_Lasswell,Affpt_Lasswell,Affoth_Lasswell,Afftot_Lasswell,Wltpt_Lasswell,Wlttran_Lasswell,Wltoth_Lasswell,Wlttot_Lasswell,Wlbgain_Lasswell,Wlbloss_Lasswell,Wlbphys_Lasswell,Wlbpsyc_Lasswell,Wlbpt_Lasswell,Wlbtot_Lasswell,Enlgain_Lasswell,Enlloss_Lasswell,Enlends_Lasswell,Enlpt_Lasswell,Enloth_Lasswell,Enltot_Lasswell,Sklasth_Lasswell,Sklpt_Lasswell,Skloth_Lasswell,Skltot_Lasswell,Trngain_Lasswell,Trnloss_Lasswell,Tranlw_Lasswell,Meanslw_Lasswell,Endslw_Lasswell,Arenalw_Lasswell,Ptlw_Lasswell,Nation_Lasswell,Anomie_Lasswell,Negaff_Lasswell,Posaff_Lasswell,Surelw_Lasswell,If_Lasswell,Notlw_Lasswell,Timespc_Lasswell,formlw_Lasswell,negative_adjectives_component,social_order_component,action_component,positive_adjectives_component,joy_component,affect_friends_and_family_component,fear_and_digust_component,politeness_component,polarity_nouns_component,polarity_verbs_component,virtue_adverbs_component,positive_nouns_component,respect_component,trust_verbs_component,failure_component,well_being_component,economy_component,certainty_component,positive_verbs_component,objects_component'
    components = ','.join(x for x in all_categories.split(',') if '_component' in x)
    return components

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

    C = 25 #cross_validate_C_SVM(X, y)
    svm_model = LinearSVC(penalty='l1', dual=False, max_iter=100000, C=C).fit(X, y)

    # #Select hyperparameter k for kNN model
    k = 27 #cross_validate_k(X, y)
    knn_model = KNeighborsClassifier(n_neighbors=k, weights='uniform').fit(X, y)

    #Create baselines
    random_baseline = DummyClassifier('stratified').fit(X, y)
    majority_baseline = DummyClassifier('most_frequent').fit(X, y)

    #Print model params
    seance_params = get_categories()

    print('LR Model Coefficients')
    print(seance_params)
    out_str = ''
    coefficients = lr_model.coef_[0]
    for i, coeff in enumerate(coefficients):
        if i == 0:
            out_str += str(coeff)
        else:
            out_str += ',' + str(coeff)
    print(out_str)
    print()

    print('LR Model Intercept')
    print(lr_model.intercept_[0])
    print()

    print('SVM Model Coefficients')
    print(seance_params)
    out_str = ''
    coefficients = svm_model.coef_[0]
    for i, coeff in enumerate(coefficients):
        if i == 0:
            out_str += str(coeff)
        else:
            out_str += ',' + str(coeff)
    print(out_str)
    print()
    
    print('SVM Model Intercept')
    print(svm_model.intercept_[0])
    print()
    
    #Get the confusion matricies for the models and baselines
    print('LR:')
    print('True Negative,False Positive,False Negative,True Positive')
    tn, fp, fn, tp = confusion_matrix(y, lr_model.predict(X)).ravel()
    print(tn, fp, fn, tp, sep=',')
    print()

    print('SVM:')
    print('True Negative,False Positive,False Negative,True Positive')
    tn, fp, fn, tp = confusion_matrix(y, svm_model.predict(X)).ravel()
    print(tn, fp, fn, tp, sep=',')
    print()

    print('kNN:')
    print('True Negative,False Positive,False Negative,True Positive')
    tn, fp, fn, tp = confusion_matrix(y, knn_model.predict(X)).ravel()
    print(tn, fp, fn, tp, sep=',')
    print()

    print('Random:')
    print('True Negative,False Positive,False Negative,True Positive')
    tn, fp, fn, tp = confusion_matrix(y, random_baseline.predict(X)).ravel()
    print(tn, fp, fn, tp, sep=',')
    print()

    print('Majority:')
    print('True Negative,False Positive,False Negative,True Positive')
    tn, fp, fn, tp = confusion_matrix(y, majority_baseline.predict(X)).ravel()
    print(tn, fp, fn, tp, sep=',')
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
        #model = LogisticRegression(penalty='l1', solver='liblinear', C=C).fit(X, y)
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
    #ADHD data
    adhd = pd.read_csv(adhd_file, header=0)
    del adhd['filename']
    indicator = [1 for _ in range(len(adhd))]
    adhd['indicator'] = indicator

    #ADHD Partner data
    adhdPart = pd.read_csv(adhdPart_file, header=0)
    del adhdPart['filename']
    indicator = [0 for _ in range(len(adhdPart))]
    adhdPart['indicator'] = indicator

    #Total DF
    data = adhd.append(adhdPart)

    #Only keep components
    for header in data.columns.values:
        if not '_component' in header and header != 'indicator':
            del data[header]

    return data.to_numpy()

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
