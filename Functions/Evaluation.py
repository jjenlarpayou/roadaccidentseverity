import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, make_scorer, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split, cross_validate, RepeatedStratifiedKFold
import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def evaluation(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1score = f1_score(y_test, y_pred, average='weighted')
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f1score)
    print("--------------------------------------------")
    target_names = ['Fatal', "Hospitalisation", 'Medical treatment', 'Minor injury', 'Property damage']
    class_report = classification_report(y_pred, y_test, target_names=target_names)
    print(class_report)
    conf_matrix = confusion_matrix(y_pred, y_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    plt.show()
    
def evaluation_class_1(y_pred, y_test):
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_pred, y_test, average='weighted')
    recall = recall_score(y_pred, y_test, average='weighted')
    f1score = f1_score(y_pred, y_test, average='weighted')
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f1score)
    print("--------------------------------------------")
    target_names = ['Minor injury', 'Property damage only', 'Serious injury']
    # target_names = ['Medical treatment', 'Minor injury', 'Property damage only', 'Serious injury']
    class_report = classification_report(y_pred, y_test, target_names=target_names)
    print(class_report)
    conf_matrix = confusion_matrix(y_pred, y_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    plt.show()

def evaluation_class_2(y_pred, y_test):
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_pred, y_test, average='weighted')
    recall = recall_score(y_pred, y_test, average='weighted')
    f1score = f1_score(y_pred, y_test, average='weighted')
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f1score)
    print("--------------------------------------------")
    target_names = ['Property damage', 'Minor injury', 'Hospitalisation', 'Fatal']
    class_report = classification_report(y_pred, y_test, target_names=target_names)
    print(class_report)
    conf_matrix = confusion_matrix(y_pred, y_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()

def evaluation_class_3(y_pred, y_test):
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_pred, y_test, average='weighted')
    recall = recall_score(y_pred, y_test, average='weighted')
    f1score = f1_score(y_pred, y_test, average='weighted')
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f1score)
    print("--------------------------------------------")
    target_names = ["Serious injury", 'Property damange', 'Minor injury']
    class_report = classification_report(y_pred, y_test, target_names=target_names)
    print(class_report)
    conf_matrix = confusion_matrix(y_pred, y_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()

def evaluation_class_4(y_pred, y_test):
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_pred, y_test, average='weighted')
    recall = recall_score(y_pred, y_test, average='weighted')
    f1score = f1_score(y_pred, y_test, average='weighted')
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f1score)
    print("--------------------------------------------")
    target_names = ["Minor injury ", "Medical treatment", "Hospitalisation", 'Fatal']
    class_report = classification_report(y_pred, y_test, target_names=target_names)
    print(class_report)
    conf_matrix = confusion_matrix(y_pred, y_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    
def evaluation_class_5(y_pred, y_test):
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_pred, y_test, average='weighted')
    recall = recall_score(y_pred, y_test, average='weighted')
    f1score = f1_score(y_pred, y_test, average='weighted')
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f1score)
    print("--------------------------------------------")
    target_names = ["Serious injury", "Minor injury", 'Medical treatment']
    class_report = classification_report(y_pred, y_test, target_names=target_names)
    print(class_report)
    conf_matrix = confusion_matrix(y_pred, y_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()

def evaluation_class_6(y_pred, y_test):
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_pred, y_test, average='weighted')
    recall = recall_score(y_pred, y_test, average='weighted')
    f1score = f1_score(y_pred, y_test, average='weighted')
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f1score)
    print("--------------------------------------------")
    target_names = ["Minor injury", "Hospitalisation", 'Fatal']
    class_report = classification_report(y_pred, y_test, target_names=target_names)
    print(class_report)
    conf_matrix = confusion_matrix(y_pred, y_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    
def evaluation_class_7(y_pred, y_test):
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_pred, y_test, average='weighted')
    recall = recall_score(y_pred, y_test, average='weighted')
    f1score = f1_score(y_pred, y_test, average='weighted')
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f1score)
    print("--------------------------------------------")
    # target_names = ["Fatal", 'Minor injury', 'Serious injury']
    class_report = classification_report(y_pred, y_test)
    print(class_report)
    conf_matrix = confusion_matrix(y_pred, y_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    
def evaluation_class_8(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1score = f1_score(y_test, y_pred, average='weighted')
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f1score)
    print("--------------------------------------------")
    target_names = ['Fatal', 'Minor injury', 'Property damage', 'Serious injury']
    class_report = classification_report(y_test, y_pred, target_names=target_names)
    print(class_report)
    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    plt.show()
    
def evaluation_class_8_MLP(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f1score)
    print("--------------------------------------------")
    target_names = ['Fatal', 'Minor injury', 'Property damage', 'Serious injury']
    class_report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    print(class_report)
    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    plt.show()