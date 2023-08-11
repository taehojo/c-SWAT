import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
sys.path.append('../')
from script.model import create_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def load_data(csv_file):
    ADMC = list(pd.read_csv(csv_file, nrows=0).columns)
    N_in = len(ADMC) - 1
    df = pd.read_csv(csv_file, usecols=ADMC)
    y_csv = df["DX"]
    x_csv = df.drop('DX', axis=1)
    x = x_csv.iloc[:,]
    y = y_csv.to_numpy()
    X = x.values.reshape(x.shape[0], x.shape[1], 1)
    return X, y, N_in

def compute_oob_with_cnn(X, y, excluded_feature_index, N_in, n_folds=5, n_batch_size=32, n_epochs=200):
    X_excluded = np.delete(X, excluded_feature_index, axis=1) if excluded_feature_index is not None else X
    
    kfold = KFold(n_splits=n_folds, shuffle=True)
    acc_per_fold = []

    for train_index, test_index in kfold.split(X_excluded):
        X_train1, X_test = X_excluded[train_index], X_excluded[test_index]
        y_train1, y_test = y[train_index], y[test_index]
        X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.25, random_state=1)

        model = create_model(N_in - 1 if excluded_feature_index is not None else N_in)
        history = model.fit(X_train, y_train, batch_size=n_batch_size, epochs=n_epochs, verbose=0, validation_data=(X_val, y_val))
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=-1)
        accuracy = accuracy_score(y_test, y_pred)
        acc_per_fold.append(accuracy)
    
    return np.mean(acc_per_fold)

def compute_feature_importance_with_rf(X, y):
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X[:, :, 0], y)
    return rf.feature_importances_

def compute_accuracy_with_rf(X, y):
    rf = RandomForestClassifier(n_estimators=100, oob_score=True)
    rf.fit(X[:, :, 0], y)
    return rf.oob_score_

def compute_accuracy_without_feature_with_rf(X, y, excluded_feature_index):
    X_excluded = np.delete(X[:, :, 0], excluded_feature_index, axis=1) if excluded_feature_index is not None else X[:,:,0]
    rf = RandomForestClassifier(n_estimators=100, oob_score=True)
    rf.fit(X_excluded, y)
    return rf.oob_score_

def main_final(csv_file, n_folds=5, n_batch_size=32, n_epochs=200):
    X, y, N_in = load_data(csv_file)
    
    # Compute and print RF feature importances
    rf_importances = compute_feature_importance_with_rf(X, y)
    feature_names = list(pd.read_csv(csv_file, nrows=0).columns)[1:]  # Excluding the first 'label' column
    print("Random Forest Feature Importances:")
    for feature, importance in zip(feature_names, rf_importances):
        print(f"{feature}: {importance:.5f}")
    print('-'*80)
    
    # Overall accuracy using CNN
    overall_acc_cnn = compute_oob_with_cnn(X, y, None, N_in, n_folds, n_batch_size, n_epochs)
    print(f"Overall accuracy (all features included) using CNN: {overall_acc_cnn}")
    
    # Overall accuracy using RF
    overall_acc_rf = compute_accuracy_with_rf(X, y)
    print(f"Overall accuracy (all features included) using RF: {overall_acc_rf}")
    print('-'*80)
    
    # Compute and print OOB scores for each feature using CNN and RF
    for i, feature in enumerate(feature_names):
        oob_cnn = overall_acc_cnn - compute_oob_with_cnn(X, y, i+1, N_in, n_folds, n_batch_size, n_epochs)  # i+1 because we're excluding the first column
        print(f"OOB for feature {feature} with CNN: {oob_cnn:.5f}")
        
        oob_rf = overall_acc_rf - compute_accuracy_without_feature_with_rf(X, y, i+1)
        print(f"OOB for feature {feature} with RF: {oob_rf:.5f}")
    print('-'*80)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_csv>")
        sys.exit(1)

    csv_file = sys.argv[1]
    main_final(csv_file)
