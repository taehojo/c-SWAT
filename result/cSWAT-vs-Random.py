import sys
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_get_accuracy(selected_features, feature_source):
    selected_features.append('DX')
    data_selected = data[selected_features]

    X = data_selected.drop(columns=['DX']).values
    y = data_selected['DX'].values

    accuracies = []
    fold_count = 1
    print(f"\nUsing {feature_source} {len(selected_features) - 1} features.")
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        
        print(f"Accuracy for fold-{fold_count}: {acc:.4f}")
        accuracies.append(acc)
        fold_count += 1

    avg_acc = sum(accuracies) / len(accuracies)
    print(f"Average accuracy over 10-folds: {avg_acc:.4f}")
    return avg_acc

if __name__ == "__main__":
    csv_file = sys.argv[1]
    pis_txt = sys.argv[2]
    num_features = int(sys.argv[3])

    data = pd.read_csv(csv_file)
    with open(pis_txt, 'r') as f:
        all_features = [line.split(',')[0] for line in f.readlines()]

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # with c-SWAT PIS
    selected_features_top = all_features[:num_features]
    train_and_get_accuracy(selected_features_top, 'PIS top')

    # no c-SWAT
    selected_features_random = random.sample(all_features, num_features)
    train_and_get_accuracy(selected_features_random, 'random')