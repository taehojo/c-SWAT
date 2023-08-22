import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.metrics import auc

matplotlib.use('Agg')
sns.set_style("whitegrid")
sns.set_context("paper")

file_path_adni = sys.argv[1]
file_path_pis = sys.argv[2]
top_features_count = int(sys.argv[3])

adni_data = pd.read_csv(file_path_adni)
pis_data = pd.read_csv(file_path_pis, sep=",", names=["Feature_Name", "PIS_Value"])

top_features = pis_data.sort_values(by="PIS_Value", ascending=False).head(top_features_count)["Feature_Name"].tolist()
least_significant_features = pis_data.sort_values(by="PIS_Value", ascending=True).head(top_features_count)["Feature_Name"].tolist()

top_data = adni_data[['DX'] + top_features]
least_significant_data = adni_data[['DX'] + least_significant_features]

def create_model(N_in):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(64, 5, activation="relu", input_shape=(N_in, 1)))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dense(N_in, activation="relu"))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(8, activation="relu"))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

fixed_fpr_grid = np.linspace(0, 1, 100)
fixed_recall_grid = np.linspace(0, 1, 100)

def cross_validate(data, n_splits=5):
    X = data.drop(columns=['DX']).values
    y = data['DX'].values
    X = X.reshape(X.shape[0], X.shape[1], 1) 
    
    kf = StratifiedKFold(n_splits=n_splits)
    avg_fpr, avg_tpr, avg_precision, avg_recall = [], [], [], []
    
    aucs = []
    
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = create_model(X_train.shape[1])
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_data=(X_test, y_test))
        y_pred_probs = model.predict(X_test)

        aucs.append(roc_auc_score(y_test, y_pred_probs[:, 1]))

        fpr, tpr, _ = roc_curve(y_test, y_pred_probs[:, 1])
        precision, recall, _ = precision_recall_curve(y_test, y_pred_probs[:, 1])
        
        interpolated_tpr = np.interp(fixed_fpr_grid, fpr, tpr)
        
        interpolated_precision = np.interp(fixed_recall_grid, recall[::-1], precision[::-1])
        
        avg_fpr.append(fixed_fpr_grid)
        avg_tpr.append(interpolated_tpr)
        avg_precision.append(interpolated_precision)
        avg_recall.append(fixed_recall_grid)
    
    return aucs, avg_fpr, avg_tpr, avg_precision, avg_recall

top_metrics = cross_validate(top_data)
least_significant_metrics = cross_validate(least_significant_data)

avg_auc_top = np.mean(top_metrics[0])
avg_auc_least = np.mean(least_significant_metrics[0])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

# 1. ROC Curve
avg_fpr_top = fixed_fpr_grid  
avg_tpr_top = np.mean(np.array(top_metrics[2]), axis=0)
avg_auc_top = auc(avg_fpr_top, avg_tpr_top)  

avg_fpr_least = fixed_fpr_grid  
avg_tpr_least = np.mean(np.array(least_significant_metrics[2]), axis=0)
avg_auc_least = auc(avg_fpr_least, avg_tpr_least)  

axes[0].plot(avg_fpr_top, avg_tpr_top, label=f"Top Features (AUC = {avg_auc_top:.3f})", color="blue")
axes[0].plot(avg_fpr_least, avg_tpr_least, label=f"Least Significant Features (AUC = {avg_auc_least:.3f})", color="red", linestyle='--')

axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve Comparison')
axes[0].legend(loc="lower right")

# 2. PR Curve
avg_recall_top = fixed_recall_grid  
avg_precision_top = np.mean(np.array(top_metrics[3]), axis=0)
axes[1].plot(avg_recall_top, avg_precision_top, label=f"Top Features", color="blue")

avg_recall_least = fixed_recall_grid  
avg_precision_least = np.mean(np.array(least_significant_metrics[3]), axis=0)
axes[1].plot(avg_recall_least, avg_precision_least, label=f"Least Significant Features", color="red", linestyle='--')

axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve Comparison')
axes[1].legend(loc="lower left")

plt.tight_layout()

plt.savefig(f'png/top_features_{top_features_count}_auc_top_{avg_auc_top:.4f}_auc_least_{avg_auc_least:.4f}.png', dpi=300)

plt.close()

print(f"Top {top_features_count} Features AUC: {avg_auc_top:.4f}")
print(f"Least Significant {top_features_count} Features AUC: {avg_auc_least:.4f}")
