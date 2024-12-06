import numpy as np
import pandas as pd

# Sample data in a DataFrame
data = [
    ['Rainy', 'Hot', 'High', 'f', 'no'],
    ['Rainy', 'Hot', 'High', 't', 'no'],
    ['Overcast', 'Hot', 'High', 'f', 'yes'],
    ['Sunny', 'Mild', 'High', 'f', 'yes'],
    ['Sunny', 'Cool', 'Normal', 'f', 'yes'],
    ['Sunny', 'Cool', 'Normal', 't', 'no'],
    ['Overcast', 'Cool', 'Normal', 't', 'yes'],
    ['Rainy', 'Mild', 'High', 'f', 'no'],
    ['Rainy', 'Cool', 'Normal', 'f', 'yes'],
    ['Sunny', 'Mild', 'Normal', 'f', 'yes'],
    ['Rainy', 'Mild', 'Normal', 't', 'yes'],
    ['Overcast', 'Mild', 'High', 't', 'yes'],
    ['Overcast', 'Hot', 'Normal', 'f', 'yes'],
    ['Sunny', 'Mild', 'High', 't', 'no']
]
columns = ['Outlook', 'Temperature', 'Humidity', 'Windy', 'Play']
df = pd.DataFrame(data, columns=columns)

# Entropy function
def entropy(col):
    counts = np.unique(col, return_counts=True)
    N = float(col.shape[0])
    ent = 0.0
    for ix in counts[1]:
        p = ix / N
        ent += (-1.0 * p * np.log2(p))
    return ent

# Divide data based on a categorical feature and value
def divide_data(X, feature, value):
    X_left = X[X[feature] == value].reset_index(drop=True)
    X_right = X[X[feature] != value].reset_index(drop=True)
    return X_left, X_right

# Information Gain function
def information_gain(X, feature, value):
    left, right = divide_data(X, feature, value)
    l = float(left.shape[0]) / X.shape[0]
    r = float(right.shape[0]) / X.shape[0]
    
    if left.shape[0] == 0 or right.shape[0] == 0:
        return -1000000
    
    inf_gain = entropy(X['Play']) - (l * entropy(left['Play']) + r * entropy(right['Play']))
    return inf_gain

# Recursive function to build the tree
def build_tree(X, depth=0, max_depth=3):
    if depth == max_depth or len(X['Play'].unique()) == 1:
        return X['Play'].mode()[0]  # Return the majority class

    # Finding the best feature and value to split on
    best_feature = None
    best_value = None
    best_gain = -float("inf")

    for feature in X.columns[:-1]:  # Exclude the target column 'Play'
        for value in X[feature].unique():
            gain = information_gain(X, feature, value)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_value = value

    if best_gain == -float("inf"):
        return X['Play'].mode()[0]

    # Split the data
    left, right = divide_data(X, best_feature, best_value)
    
    # Recursively build the left and right subtrees
    tree = {}
    tree["feature"] = best_feature
    tree["value"] = best_value
    tree["left"] = build_tree(left, depth + 1, max_depth)
    tree["right"] = build_tree(right, depth + 1, max_depth)
    
    return tree

# Prediction function
def predict(tree, test):
    if not isinstance(tree, dict):
        return tree

    feature = tree["feature"]
    value = tree["value"]

    if test[feature] == value:
        return predict(tree["left"], test)
    else:
        return predict(tree["right"], test)

# Function to print the tree in a readable format
def print_tree(tree, depth=0):
    if not isinstance(tree, dict):
        print("\t" * depth + "-> " + str(tree))
        return

    feature = tree["feature"]
    value = tree["value"]
    print("\t" * depth + f"[{feature} = {value}]")
    print("\t" * depth + "Left:")
    print_tree(tree["left"], depth + 1)
    print("\t" * depth + "Right:")
    print_tree(tree["right"], depth + 1)

# Build the decision tree
tree = build_tree(df)

# Test example
x_test = {'Outlook': 'Sunny', 'Temperature': 'Mild', 'Humidity': 'High', 'Windy': 'f'}
prediction = predict(tree, x_test)

# Print the decision tree
print("Decision Tree:")
print_tree(tree)
print("\nPrediction for test example:", prediction)
