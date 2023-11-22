import numpy as np

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


