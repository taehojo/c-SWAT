import sys
import numpy as np
import pandas as pd
import ast
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

def load_data(csv_file, module_file, iteration_number):
    ADMC = list(pd.read_csv(csv_file, nrows=0).columns)
    with open(module_file, 'r') as file:
        contents = file.read()
        module = ast.literal_eval(contents)
    myList_pop = module[iteration_number]
    myList = [x for x in ADMC if x not in myList_pop]
    N_in = len(myList) - 1
    df = pd.read_csv(csv_file, usecols=myList)
    y_csv = df["DX"]
    x_csv = df.drop('DX', axis=1)
    x = x_csv.iloc[:,]
    y = y_csv.to_numpy()
    X = x.values.reshape(x.shape[0], x.shape[1], 1)
    return X, y, N_in, module

# Model
kernel_size = 5
filters = 64
N_hidden1 = 32
N_hidden2 = 16
N_hidden3 = 8
N_out = 2

def create_model(N_in):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters, kernel_size, activation="relu", input_shape=(N_in, 1)))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dense(N_in, activation="relu"))
    model.add(tf.keras.layers.Dense(N_hidden1, activation="relu"))
    model.add(tf.keras.layers.Dense(N_hidden2, activation="relu"))
    model.add(tf.keras.layers.Dense(N_hidden3, activation="relu"))
    model.add(tf.keras.layers.Dense(N_out, activation='softmax', name="dense_e"))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def calculate_feature_importance(X, y, feature_names):
    rf = RandomForestClassifier()
    rf.fit(X, y)
    importances = {}
    for i, name in enumerate(feature_names):
        importances[name] = rf.feature_importances_[i]
    return importances

def calculate_overall_accuracy(X, y):
    kfold = KFold(n_splits=5, shuffle=True)
    acc_per_fold = []
    N_in = X.shape[1]

    for train_index, test_index in kfold.split(X):
        X_train1, X_test = X[train_index], X[test_index]
        y_train1, y_test = y[train_index], y[test_index]
        X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.25, random_state=1)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model = create_model(N_in)
        history = model.fit(X_train, y_train, batch_size=32, epochs=200, verbose=0, validation_data=(X_val, y_val))
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=-1)
        accuracy = accuracy_score(y_test, y_pred)
        acc_per_fold.append(accuracy)
    
    return np.mean(acc_per_fold)

def get_feature_value(feature, results_dict):
    for key in results_dict:
        if feature in key:
            return results_dict[key]
    return 0
        
def main(csv_file, module_file):
    results = {}
    _, _, _, module = load_data(csv_file, module_file, 0)
    for i in range(len(module)):
        X, y, N_in, _ = load_data(csv_file, module_file, i)
        kfold = KFold(n_splits=5, shuffle=True)
        acc_per_fold = []

        for train_index, test_index in kfold.split(X):
            X_train1, X_test = X[train_index], X[test_index]
            y_train1, y_test = y[train_index], y[test_index]
            X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.25, random_state=1)

            model = create_model(N_in)
            history = model.fit(X_train, y_train, batch_size=32, epochs=200, verbose=0, validation_data=(X_val, y_val))
            y_pred_prob = model.predict(X_test)
            y_pred = np.argmax(y_pred_prob, axis=-1)
            accuracy = accuracy_score(y_test, y_pred)
            acc_per_fold.append(accuracy)

        results[str(module[i])] = np.mean(acc_per_fold)
    return results

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python c-SWAT.py <path_to_csv> <path_to_module_file1> <path_to_module_file2>")
        sys.exit(1)

    csv_file = sys.argv[1]
    module_file1 = sys.argv[2]
    module_file2 = sys.argv[3]
    
    df = pd.read_csv(csv_file)
    y = df["DX"].to_numpy()
    X = df.drop('DX', axis=1).values
    feature_importances = calculate_feature_importance(X, y, df.columns[1:]) 
    y_value = calculate_overall_accuracy(X, y)
    
    print("Running for first module file...")
    results1 = main(csv_file, module_file1)
    print("Running for second module file...")
    results2 = main(csv_file, module_file2)
    
    value1_list = [y_value - results1[feature_group] for feature_group in results1]
    value2_list = []

    for feature_group in results1:
        group_diff = sum([(y_value - get_feature_value(feature, results2)) for feature in ast.literal_eval(feature_group)]) / len(ast.literal_eval(feature_group))
        value2_list.append(group_diff)

    value1_min, value1_max = min(value1_list), max(value1_list)
    value2_min, value2_max = min(value2_list), max(value2_list)

    value1_normalized = [(v - value1_min) / (value1_max - value1_min) for v in value1_list]
    value2_normalized = [(v - value2_min) / (value2_max - value2_min) for v in value2_list]

    for idx, feature_group in enumerate(results1):
        for feature in ast.literal_eval(feature_group):
            final_value = feature_importances[feature] + 0.5 * (value1_normalized[idx] + value2_normalized[idx])#print(f"{feature},{feature_importances[feature]:.6f},{value1_list[idx]:.6f},{value2_list[idx]:.6f},{value1_normalized[idx]:.6f},{value2_normalized[idx]:.6f},{final_value:.6f}")
            print(f"{feature},{final_value:.6f}")