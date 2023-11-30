import numpy as np
from fractions import Fraction
from matplotlib.pyplot import figure, show
from scipy.cluster.hierarchy import linkage, dendrogram
from basic_operations import choose_mode
from scipy.spatial.distance import squareform
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import sklearn.metrics.cluster as cluster_metrics
import math
from exam_toolbox import *


# decision tree impurity gini entropy 
def impurity(classes, impurity_mode='gini'):
    total_elements = np.sum(classes)
    if total_elements == 0:
        return 0.0  # Nodo vuoto, impurità zero

    probabilities = classes / total_elements

    if impurity_mode == 'gini':
        impurity_value = 1.0 - np.sum(probabilities ** 2)
    elif impurity_mode == 'entropy':
        impurity_value = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    elif impurity_mode == 'class_error':
        impurity_value = 1.0 - np.max(probabilities)
    else:
        raise ValueError("Invalid impurity_mode. Choose from 'gini', 'entropy', or 'class_error'.")

    return impurity_value

def impurity_gain(root_classes, child_classes, impurity_mode='gini'):
    root_impurity = impurity(root_classes, impurity_mode)
    total_root_elements = np.sum(root_classes)

    print("Root Impurity:", root_impurity)

    child_impurity = 0.0
    for i, child_class in enumerate(child_classes):
        total_child_elements = np.sum(child_class)
        child_impurity_i = (total_child_elements / total_root_elements) * impurity(child_class, impurity_mode)
        child_impurity += child_impurity_i
        print(f"Child {i + 1} Impurity:", child_impurity_i)

    gain = root_impurity - child_impurity
    return gain


# # Esempio di utilizzo
# root_classes = np.array([18, 18, 18])  # Esempio con due classi
# child1_classes = np.array([6, 9,3])
# child2_classes = np.array([4,6,10])
# child3_classes = np.array([8,3,5])
# #child4_classes = np.array([8,3,2])

# gain = impurity_gain(root_classes, [child1_classes, child2_classes, child3_classes], impurity_mode = "entropy")
# print("Purity Gain:", gain)


def kmeans2_main(data, centroids):
    c1, c2 = centroids
    dif1, dif2 = data - c1, data - c2
    cat1, cat2 = [], []

    for i in range(0, len(data)):
        if abs(dif1[i]) <= abs(dif2[i]):
            cat1.append(data[i])
        elif abs(dif2[i]) <= abs(dif1[i]):
            cat2.append(data[i])
        else:
            print("ERROR")

    # Print clusterings
    print(cat1, cat2)

    # Return new centroids
    return np.array([np.mean(cat1), np.mean(cat2)])


def adaboost(delta, rounds):
    """
    delta : list of misclassified observations, 
    0 = correctly classified, 1 = misclassified

    rounds [int] : how many rounds to run
    
    !!!CALL FOR EACH ROUND!!!
    
    Example: 
    Given a classification problem with 25 observations in total, 
    with 5 of them being misclassified in round 1, the weights can be calculated as:
    miss = np.zeros(25) 
    miss[:5] = 1 
    te.adaboost(miss, 1)

    The weights are printed 
    """
    # Initial weights
    delta = np.array(delta)
    n = len(delta)
    weights = np.ones(n) / n

    # Run all rounds
    for i in range(rounds):
        eps = np.mean(delta == 1)
        alpha = 0.5 * np.log((1 - eps) / eps)
        s = np.array([-1 if d == 0 else 1 for d in delta])
        print(alpha)
        
        # Calculate weight vector and normalize it
        weights = weights.T * np.exp(s * alpha)
        weights /= np.sum(weights)

        # Print resulting weights
    for i, w in enumerate(weights):
        print('w[%i]: %f' % (i, w))



def cunfusion_matrix_stats(TN, TP, FN, FP):
    accuracy = Fraction(TP + TN, TP + TN + FP + FN)
    recall = Fraction(TP, TP + FN)
    specificity = Fraction(TN, TN + FP)
    precision = Fraction(TP, TP + FP)
    f1_score = Fraction(2 * (precision * recall), precision + recall)
    FPR = Fraction(1 - specificity)
    TNR = Fraction(TN, TN + FP)
    balanced_accuracy = Fraction(recall + specificity, 2)

    metrics = {
        "Accuracy": accuracy,
        "Recall": recall,
        "Specificity": specificity,
        "Precision": precision,
        "F1-score": f1_score,
        "FPR": FPR,
        "TNR": TNR,
        "Balanced Accuracy": balanced_accuracy
    }

    results = {}
    for metric_name, value in metrics.items():
        decimal_value = float(value)
        fractional_value = f" {value.numerator}/{value.denominator} "
        results[metric_name] = f"{metric_name} = {fractional_value} = {decimal_value:.5f}"

    return results

# # Esempio di utilizzo
# TN = 14
# TP = 14
# FN = 18
# FP = 10

# metrics = ef.cunfusion_matrix_stats(TN, TP, FN, FP)
# for metric, value in metrics.items():
#     print(value)



def draw_dendrogram(x, method='single', metric='euclidean'):
    """
    :param x:
    :param method: single complete average centroid median ward weighted
    single is min, complete is max
    :param metric:
    :return: prints result
    """
    data = np.array(choose_mode(x))
    data = np.array(data)
    z = linkage(squareform(data), method=method, metric=metric, optimal_ordering=True)
    figure(2, figsize=(10, 4))
    dendrogram(z, count_sort='descendent', labels=list(range(1, len(data[0]) + 1)))
    show()


# a = """O1 O2 O3 O4 O5 O6 O7 O8 O9 O10
# O1 0 8.55 0.43 1.25 1.14 3.73 2.72 1.63 1.68 1.28
# O2 8.55 0 8.23 8.13 8.49 6.84 8.23 8.28 8.13 7.66
# O3 0.43 8.23 0 1.09 1.10 3.55 2.68 1.50 1.52 1.05
# O4 1.25 8.13 1.09 0 1.23 3.21 2.17 1.29 1.33 0.56
# O5 1.14 8.49 1.10 1.23 0 3.20 2.68 1.56 1.50 1.28
# O6 3.73 6.84 3.55 3.21 3.20 0 2.98 2.66 2.50 3.00
# O7 2.72 8.23 2.68 2.17 2.68 2.98 0 2.28 2.30 2.31
# O8 1.63 8.28 1.50 1.29 1.56 2.66 2.28 0 0.25 1.46
# O9 1.68 8.13 1.52 1.33 1.50 2.50 2.30 0.25 0 1.44
# O10 1.28 7.66 1.05 0.56 1.28 3.00 2.31 1.46 1.44 0"""
# draw_dendrogram(a, method='average')


# a2 = '5.7 6.0 6.2 6.3 6.4 6.6 6.7 6.9 7.0 7.4'

# print(draw_dendrogram(a2, method='average'))



def clustering_metrics(cluster1, cluster2):
    if len(cluster1) != len(cluster2):
        raise ValueError("I due vettori devono avere la stessa lunghezza.")

    n = len(cluster1)
    agree = 0
    total_pairs = 0

    for pair in combinations(range(n), 2):
        a, b = pair
        a_same_cluster = (cluster1[a] == cluster1[b])
        b_same_cluster = (cluster2[a] == cluster2[b])

        if a_same_cluster == b_same_cluster:
            agree += 1

        total_pairs += 1

    rand_index = agree / total_pairs

    set_cluster1 = set(cluster1)
    set_cluster2 = set(cluster2)
    intersection = len(set_cluster1.intersection(set_cluster2))
    union = len(set_cluster1.union(set_cluster2))
    jaccard = 1 - (intersection / union)

    cluster1 = np.array(cluster1)
    cluster2 = np.array(cluster2)
    cosine = 1 - (np.dot(cluster1, cluster2) / (np.linalg.norm(cluster1) * np.linalg.norm(cluster2)))

    return {
        "Rand Index - SMC": rand_index,
        "JACCARD distnace": jaccard,
        "COSINE distance ": cosine
    }
    
    


def ARD(df, obs, K):
    """
    Calculates average relative density
    ----------------------
    parameters:
    ----------------------
    df = symmetric matrix with distances
    obs = the observation to calculate ARD for  (0 index)
    K = the number of nearest neighbors to consider
    """

    O = df.loc[obs, :].values
    dist_sort = np.argsort(O)
    k_nearest_ind = dist_sort[1 : K + 1]

    densO = 1 / (1 / K * np.sum(O[k_nearest_ind]))

    densOn = []

    for n in k_nearest_ind:
        On = df.loc[n, :].values
        dist_sort_n = np.argsort(On)
        k_nearest_ind_n = dist_sort_n[1 : K + 1]
        densOn.append(1 / (1 / K * np.sum(On[k_nearest_ind_n])))

    ARD = densO / (1 / K * np.sum(densOn))

    print("The density for observation O{} is {}".format(obs + 1, densO))
    print(
        "The average relative density for observation O{} is {}".format(
            obs + 1, ARD
        )
    )

    return ARD

# data =[[0,8.55,0.43,1.25,1.14,3.73,2.72,1.63,1.68,1.28],
#         [8.55,0,8.23,8.13,8.49,6.84,8.23,8.28,8.13,7.66],
#         [0.43,8.23,0,1.09,1.10,3.55,2.68,1.50,1.52,1.05],
#         [1.25,8.13,1.09,0,1.23,3.21,2.17,1.29,1.33,0.56],
#         [1.14,8.49,1.10,1.23,0,3.20,2.68,1.56,1.50,1.28],
#         [3.73,6.84,3.55,3.21,3.20,0,2.98,2.66,2.50,3.00],
#         [2.72,8.23,2.68,2.17,2.68,2.98,0,2.28,2.30,2.31],
#         [1.63,8.28,1.50,1.29,1.56,2.66,2.28,0,0.25,1.46],
#         [1.68,8.13,1.52,1.33,1.50,2.50,2.30,0.25,0,1.44],
#         [1.28,7.66,1.05,0.56,1.28,3.00,2.31,1.46,1.44,0]]
# df = pd.DataFrame(data)

# ARD(df,1,2)

def plot_roc(true_val, pred_val):
    """
    calculates the fpr and tpr and plots a roc curve
    to compare the outputtet graph with the possible answers, look at where the plot has a elbow
    -----------------------------------------------
    parameters:
    -----------
    true_val = list of the correct labels, must be binarised so that 1 = positve class and 0 = negative class
    pred_val = list of the predicted labels, must be binarised so that 1 = positve class and 0 = negative class

    returns the area under the curve (AUC)
    """
    fpr, tpr, _ = metrics.roc_curve(true_val, pred_val)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()




def normalized_mutual_info_score(y, clusterid):
    '''
    normalization of the Mutual Information (MI) score to scale the results between 0 (no mutual information) and 1 (perfect correlation)
    param:
    y = list of correct labels
    clusterid = list of labels assigned by the clustering

    returns the normalized mutual informationof the two clusterings (the correct clustering and the other one). average_method is set to default to geometric (exercises in the exam want
    geometric method (always?)). other average methods are min, max.
    '''

    NMI = cluster_metrics.normalized_mutual_info_score(y,clusterid, average_method='geometric')
    return NMI

# y =[0,0,0,1,1,1,1,1,2,2]
# clusterid =[0,2,0,1,0,1,0,0,0,0]
# print(normalized_mutual_info_score(y,clusterid))


def empirical_correlation_from_covariance(cov_matrix):
        """
        cov_matrix : 2d array of covariance matrix, eg: [[0.2639, 0.0803], [0.0803, 0.0615]]
        Calculates the correlation between the two x's in the covariance matrix
        The correlation coefficient is defined as: p=cov(x,y)/(sigma_x*sigma_y)
        """
        cov = np.array(cov_matrix)
        p = cov[1][0]/math.sqrt((cov[0][0]*cov[1][1]))
        print (p)
        return p



def rule_stats(df, X, Y, index=0):
        """
        calculates support and confidence of an association rule
        ----------------------
        parameters:
        -----------
        df = binary dataframe with rows as transactions and colums representing the attributes
        X = list with the X itemsets
        Y = list with the Y itemsets
        """
        X = np.array(X) - index
        Y = np.array(Y) - index

        X_item = df.loc[:, X]
        Y_item = df.loc[:, Y]
        items = pd.concat([X_item, Y_item], axis=1)

        support = np.mean(np.mean(items, axis=1) == 1)
        confidence = np.sum(np.mean(items, axis=1) == 1) / np.sum(
            np.mean(X_item, axis=1) == 1
        )

        the_rule = "{{{}}} ---> {{{}}}".format(X, Y)

        print("The rule is {}".format(the_rule))
        print("The support for the rule is {}".format(support))
        print("The confidence for the rule is {}".format(confidence))

##example:
# a = association_mining()
# data=[[1,0,1,0,1,0,1,0,1,0,1,0],
# [0,1,0,1,0,1,0,1,0,1,0,1],
# [1,0,0,1,1,0,1,0,1,0,1,0],
# [1,0,1,0,1,0,0,1,0,1,1,0],
# [0,1,1,0,1,0,1,0,1,0,1,0],
# [0,1,0,1,0,1,0,1,0,1,0,1],
# [0,1,1,0,1,0,0,1,0,1,0,1],
# [1,0,1,0,1,0,1,0,0,1,0,1],
# [0,1,0,1,1,0,1,0,0,1,0,1],
# [1,0,0,1,0,1,0,1,0,1,1,0]]

# df= pd.DataFrame(data)
# x=[3,5,7,9]
# #x=pd.DataFrame(x)
# y=[11]
# #y=pd.DataFrame(y)
# a.rule_stats(df,x,y)



def apriori_algorithm(df, support_min):
        """
        df: dataframe with each row being a basket, and each column being an item
        support_min: minimum support level
        Remember that the printed itemsets start from 0!
        return the frequent itemsets, that is to say the itemsets with support>support_min
        """
        itemsets = []
        n = len(df)
        for itsetSize in np.arange(1, len(df.columns) + 1): # Start with 1-itemsets, keep going till n_attributes-itemsets
            itemsets.append("L: ")
            itemsets.append(itsetSize)
            for combination in IT.combinations(df.columns, itsetSize):
                itemset = df[list(combination)]
                baskets = itemset.iloc[:,0].copy()
                for col in itemset.columns[1:]:
                    baskets = baskets & itemset[col]
                sup =  baskets.sum() / float(len(baskets))
                if sup > support_min:
                    itemsets.append(set(combination))
        print(itemsets)
        return itemsets

#esempio:
# data=[[1,0,1,0,1,0,1,0,1,0,1,0],
# [0,1,0,1,0,1,0,1,0,1,0,1],
# [1,0,0,1,1,0,1,0,1,0,1,0],
# [1,0,1,0,1,0,0,1,0,1,1,0],
# [0,1,1,0,1,0,1,0,1,0,1,0],
# [0,1,0,1,0,1,0,1,0,1,0,1],
# [0,1,1,0,1,0,0,1,0,1,0,1],
# [1,0,1,0,1,0,1,0,0,1,0,1],
# [0,1,0,1,1,0,1,0,0,1,0,1],
# [1,0,0,1,0,1,0,1,0,1,1,0]]

# df= pd.DataFrame(data)
# apriori_algorithm(df, 0.4)
#l'output è del tipo: ['L: ', 1, {0}, {1}, {2}, {3}, {4}, {6}, {7}, {9}, {10}, {11}, 'L: ', 2, {2, 4}, {4, 6}, {9, 7}, {9, 11}, 'L: ', 3, 'L: ', 4, 'L: ', 5 etc etc
# L:1 significa itemset con una sola feature, L:2 istemset con due features etc etc

def var_explained(S, plot=True, show_df=True):
    """
    S = list of variances of components (can be read from the S/Sigma matrix)
    plot = to plot or not
    show_df = to show df with varaince explained or not
    """
    S = np.array(S)

    # df with variance explained
    df_var_exp = pd.DataFrame(columns=["k", "var_explained"])
    for i in range(len(S)):
        t = np.sum(S[0 : i + 1] ** 2) / np.sum(S ** 2)
        df_var_exp.loc[i] = [i + 1, t]
    if plot:
        # plot of variance explained
        plt.plot(df_var_exp["k"], df_var_exp["var_explained"])
        plt.scatter(df_var_exp["k"], df_var_exp["var_explained"])
        plt.title("Variance explained")
        plt.xlim(np.min(df_var_exp["k"]), np.max(df_var_exp["k"]))
        plt.ylim(0, 1)
        plt.show()

    if show_df:
        print(df_var_exp)
    return df_var_exp



def gmm_weighted_density(sigma, dim, vect_mean, weights):
    """
    Computes the weighted probability density in a Gaussian Mixture Model (GMM).

    Args:
    sigma (float): Standard deviation of the distribution
    dim (int): Dimensionality of the distribution (number of observations)
    vect_mean (list): List of mean values of observations (or distances from the matrix given)
    weights (list): List of weights corresponding to the observations (usually equally weigthed)

    Returns:
    float: Calculated probability density considering different weights for each observation.
    """
    # Convert the weights into a NumPy array
    weights_array = np.array(weights)
    
    # Calculate the prefactors for each observation
    prefactor = weights_array / ((2 * np.pi * sigma**2) ** (dim / 2))

    # Calculate exponentials of the observation mean values
    exp_tot = np.exp(-np.array(vect_mean) ** 2 / (2 * sigma**2))

    # Compute the weighted sum of prefactors and exponentials
    result = np.dot(prefactor, exp_tot)

    return result

## EXAMPLE OF USE
# weigths = [0.3333, 0.3333, 0.3333]
# vect_dist = np.array([2.11, 1.15, 1.09])
# sigma = 0.5
# dim = 10

# gmm_weighted_density(sigma, dim, vect_dist, weigths)
