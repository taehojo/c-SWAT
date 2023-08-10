import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from script.model import create_model
from script.data_processing import load_data
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from script.model import create_model
from script.data_processing import load_data
from sklearn.model_selection import train_test_split

def main(csv_file, module_file, n_folds=5, n_batch_size=32, n_epochs=200):
    _, _, _, module = load_data(csv_file, module_file, 0)
    for i in range(len(module)):
        print("iteration_no=", i, " / loocv_module=", module[i])
        X, y, N_in, _ = load_data(csv_file, module_file, i)
        print(f"Number of input: {N_in}")

        kfold = KFold(n_splits=n_folds, shuffle=True)
        acc_per_fold = []
        fold_no = 1

        for train_index, test_index in kfold.split(X):
            print(f"> Fold {fold_no}: module_no={i}, Number of input: {N_in}")
            X_train1, X_test = X[train_index], X[test_index]
            y_train1, y_test = y[train_index], y[test_index]
            X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.25, random_state=1)

            model = create_model(N_in)
            history = model.fit(X_train, y_train, batch_size=n_batch_size, epochs=n_epochs, verbose=0, validation_data=(X_val, y_val))
            y_pred_prob = model.predict(X_test)
            y_pred = np.argmax(y_pred_prob, axis=-1)
            accuracy = accuracy_score(y_test, y_pred)

            acc_per_fold.append(accuracy)
            fold_no = fold_no + 1

        print(f'> average acc for interation_no={i}: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print('------------------------------------------------------------------------')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <path_to_csv> <path_to_module_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    module_file = sys.argv[2]
    main(csv_file, module_file)
