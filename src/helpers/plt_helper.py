import logging
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# import scikitplot as sk_plt
from pandas.plotting import scatter_matrix
# from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import roc_curve, auc

from src.iot23 import get_feature_selection, decode_labels


def plot_correlations(data, title):
    # Replace scikit-plot correlation plot with matplotlib equivalent
    # Example code using matplotlib to plot correlations
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    plt.matshow(correlation_matrix, cmap='coolwarm')
    plt.title(title)
    plt.colorbar()
    plt.show()

def plot_class_values_distribution(data, title):
    # Replace scikit-plot class values distribution plot with matplotlib equivalent
    # Example code using matplotlib to plot class values distribution
    plt.figure(figsize=(10, 6))
    data['Class'].value_counts().plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

def plot_attr_values_distribution(data, attribute, title):
    # Replace scikit-plot attribute values distribution plot with matplotlib equivalent
    # Example code using matplotlib to plot attribute values distribution
    plt.figure(figsize=(10, 6))
    data[attribute].hist(color='lightcoral', bins=20)
    plt.title(title)
    plt.xlabel(attribute)
    plt.ylabel('Frequency')
    plt.show()

def plot_confusion_ma3x(output_dir,
                        y_test,
                        predictions,
                        title="Confusion Matrix",
                        file_name="conf_ma3x.png"):
    classes = unique_labels(y_test, predictions)
    cnt = len(classes)
    cnt = cnt * 2 if cnt < 10 else cnt * 0.7
    # labels = decode_labels(classes)

    sk_plt.metrics.plot_confusion_matrix(y_test,
                                         predictions,
                                         normalize=True,
                                         title=title + " (Normalized)",
                                         title_fontsize="large",
                                         figsize=(cnt, cnt))
    export_plt(output_dir + file_name + '_n.png')


def plot_confusion_ma3x_v2(output_dir,
                           y_test,
                           predictions,
                           title="Confusion Matrix",
                           file_name="conf_ma3x_v2.png",
                           export=True):
    classes = unique_labels(y_test, predictions)
    cnt = len(classes)
    small = cnt < 10
    left = 0.2 if small else 0.12
    size = cnt * 1.5 if small else cnt * 0.8
    labels = decode_labels(classes)

    fig, ax = plt.subplots(1, 1, figsize=(size, size))
    fig.subplots_adjust(left=left)
    ax = sk_plt.metrics.plot_confusion_matrix(y_test,
                                              predictions,
                                              normalize=True,
                                              title=title + " (Normalized)",
                                              title_fontsize="large",
                                              ax=ax)
    ax.set_xticklabels(labels, rotation=35)
    ax.set_yticklabels(labels)
    export_plt(output_dir + file_name)

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots the confusion matrix.
    
    Parameters:
    y_true (numpy.ndarray): True labels.
    y_pred (numpy.ndarray): Predicted labels.
    classes (list): List of class labels.
    normalize (bool): Whether to normalize the confusion matrix.
    title (str): Title of the plot.
    cmap (matplotlib.colors.Colormap): Colormap for the plot.
    """
    import itertools

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def plot_model_roc_curve(y_true, y_pred_proba):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.
    
    Parameters:
    y_true (numpy.ndarray): True labels.
    y_pred_proba (numpy.ndarray): Predicted probabilities.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_roc_custom(output_dir, y_true, y_prob, name, model_name, file_name, type):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.subplots_adjust(top=0.8, right=0.65)
        sk_plt.metrics.plot_roc(y_true,
                                y_prob,
                                title=name + "\n\n" + model_name + "\n" + type + " Curve\n",
                                cmap='nipy_spectral',
                                ax=ax,
                                plot_micro=False,
                                plot_macro=False)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='best', fontsize='medium')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='best', fontsize='medium')
        export_plt(output_dir + file_name + '_' + type + '.png')
    except:
        logging.error("Oops! Could not plot " + type + " Curve for model " + name)


def plot_model_precision_recall_curve(output_dir,
                                      model,
                                      model_name,
                                      x_test,
                                      y_true,
                                      experiment_name,
                                      title="Precision Recall Curve",
                                      file_name="pr_recall_curve.png"):
    try:
        y_prob = model.predict_proba(x_test)
        plot_precision_recall_curve_custom(output_dir, y_true, y_prob, experiment_name, model_name, file_name, "Precision-Recall")
    except:
        try:
            y_decision_auc = model.decision_function(x_test)
            plot_precision_recall_curve_custom(output_dir, y_true, y_decision_auc, experiment_name, model_name, file_name, "Precision-Recall_AUC")
        except:
            logging.error("Run test data fix to plot ROC for " + model_name)


def plot_precision_recall_curve_custom(output_dir, y_true, y_prob, name, model_name, file_name, type):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.subplots_adjust(top=0.8, right=0.55)
        sk_plt.metrics.plot_precision_recall(y_true, y_prob, title=name + "\n\n" + model_name + "\n" + type + " Curve\n", cmap='nipy_spectral', ax=ax, plot_micro=False)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='best', fontsize='medium')
        export_plt(output_dir + file_name + '_' + type + '.png')
    except:
        logging.error("Oops! Could not export Precision/ Recall Curve for model " + model_name)


def plot_feature_importance(results_location,
                            model_name,
                            experiment_name,
                            feat_importance,
                            title="Feature Importance",
                            file_name="feat_imp.png"):
    feature_names = get_feature_selection(experiment_name)

    values = list(feat_importance.values())
    x_pos = [x for x in range(len(values))]

    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.25, top=0.75, left=0.15)
    ax.bar(x_pos, values, color='orange', alpha=0.6)
    ax.set_title(title)
    ax.set_ylabel('Importance')
    ax.set_xlabel('Features')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        feature_names[0:len(values)],
        rotation=35,
        ha="right",
        rotation_mode="anchor")
    export_plt(results_location + file_name)


def export_plt(file_path, export=True):
    if export:
        plt.savefig(file_path)
        plt.close()
        plt.cla()
    else:
        plt.show()


def export_sns(fig, file_path, export=True):
    if export:
        fig.savefig(file_path)
        plt.close()
        plt.cla()
    else:
        plt.show()


def plot_permutation_importance(results_location,
                                model_name,
                                experiment_name,
                                permutation_importance,
                                title="Permutation Importance",
                                file_name="permut_imp.png"):
    columns = permutation_importance['columns']
    sorted_idx = permutation_importance.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(permutation_importance.importances[sorted_idx].T, labels=columns[sorted_idx], vert=False)
    ax.set_title("Permutation Importances")
    fig.tight_layout()
    export_plt(results_location + file_name)
    export_plt(results_location + file_name)


def print_scatter_matrix(output_dir,
                         df,
                         title='Scatter MAtrix',
                         file_name="feature_distribution.png",
                         export=True):
    file_path = output_dir + file_name
    cnt = len(df.columns)

    plt.style.use('ggplot')
    pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(cnt * 2, cnt * 2), color='black')
    export_plt(file_path, export=export)
