import sys
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_get_accuracy(selected_features):
    selected_features.append('DX')
    data_selected = data[selected_features]

    X = data_selected.drop(columns=['DX']).values
    y = data_selected['DX'].values

    accuracies = []
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        
        accuracies.append(acc)

    return sum(accuracies) / len(accuracies)

if __name__ == "__main__":
    csv_file = sys.argv[1]
    pis_txt = sys.argv[2]
    num_features = int(sys.argv[3])

    data = pd.read_csv(csv_file)
    with open(pis_txt, 'r') as f:
        all_features = [line.split(',')[0] for line in f.readlines()]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # with c-SWAT PIS
    selected_features_top = all_features[:num_features]
    avg_acc_top = train_and_get_accuracy(selected_features_top)
    print(f"\nUsed PIS top {num_features} features. Average accuracy over 5-folds: {avg_acc_top:.4f}")

    # no c-SWAT
    selected_features_random = random.sample(all_features, num_features)
    avg_acc_random = train_and_get_accuracy(selected_features_random)
    print(f"\nUsed {num_features} random features. Average accuracy over 5-folds: {avg_acc_random:.4f}")