import numpy as np
from fractions import Fraction

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
