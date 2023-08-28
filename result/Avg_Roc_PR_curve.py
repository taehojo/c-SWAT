import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import random

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
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model

fixed_fpr_grid = np.linspace(0, 1, 100)
fixed_recall_grid = np.linspace(0, 1, 100)

def cross_validate(data, n_splits=10):
    X = data.drop(columns=['DX']).values
    y = data['DX'].values
    
    kf = StratifiedKFold(n_splits=n_splits)
    avg_fpr, avg_tpr, avg_precision, avg_recall = [], [], [], []
    
    aucs = []
    
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = create_model(X_train.shape[1])
        model.fit(X_train, y_train)
        y_pred_probs = model.predict_proba(X_test)[:, 1]

        aucs.append(roc_auc_score(y_test, y_pred_probs))

        fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_probs)
        
        interpolated_tpr = np.interp(fixed_fpr_grid, fpr, tpr)
        interpolated_precision = np.interp(fixed_recall_grid, recall[::-1], precision[::-1])
        
        avg_fpr.append(fixed_fpr_grid)
        avg_tpr.append(interpolated_tpr)
        avg_precision.append(interpolated_precision)
        avg_recall.append(fixed_recall_grid)
    
    return aucs, avg_fpr, avg_tpr, avg_precision, avg_recall

top_metrics = cross_validate(top_data)
least_significant_metrics = cross_validate(least_significant_data)

random_features = random.sample(pis_data["Feature_Name"].tolist(), top_features_count)
random_data = adni_data[['DX'] + random_features]

random_metrics = cross_validate(random_data)
avg_auc_random = np.mean(random_metrics[0])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

# 1. ROC Curve
avg_fpr_top = fixed_fpr_grid  
avg_tpr_top = np.mean(np.array(top_metrics[2]), axis=0)
avg_auc_top = auc(avg_fpr_top, avg_tpr_top)  

avg_fpr_least = fixed_fpr_grid  
avg_tpr_least = np.mean(np.array(least_significant_metrics[2]), axis=0)
avg_auc_least = auc(avg_fpr_least, avg_tpr_least)  

avg_fpr_random = fixed_fpr_grid  
avg_tpr_random = np.mean(np.array(random_metrics[2]), axis=0)
avg_auc_random = auc(avg_fpr_random, avg_tpr_random)

axes[0].plot(avg_fpr_top, avg_tpr_top, label=f"c-SWAT Top Features (AUC = {avg_auc_top:.3f})", color="blue")
axes[0].plot(avg_fpr_random, avg_tpr_random, label=f"Random Features (AUC = {avg_auc_random:.3f})", color="green", linestyle=':')
axes[0].plot(avg_fpr_least, avg_tpr_least, label=f"Least Associated Features (AUC = {avg_auc_least:.3f})", color="red", linestyle='--')

axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve Comparison')
axes[0].legend(loc="lower right")

# 2. PR Curve
avg_recall_top = fixed_recall_grid  
avg_precision_top = np.mean(np.array(top_metrics[3]), axis=0)
axes[1].plot(avg_recall_top, avg_precision_top, label=f"c-SWAT Top Features", color="blue")

avg_recall_random = fixed_recall_grid  
avg_precision_random = np.mean(np.array(random_metrics[3]), axis=0)
axes[1].plot(avg_recall_random, avg_precision_random, label=f"Random Features", color="green", linestyle=':')

avg_recall_least = fixed_recall_grid  
avg_precision_least = np.mean(np.array(least_significant_metrics[3]), axis=0)
axes[1].plot(avg_recall_least, avg_precision_least, label=f"Least Associated Features", color="red", linestyle='--')

axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve Comparison')
axes[1].legend(loc="lower left")

plt.tight_layout()

plt.savefig(f'png/top_features_{top_features_count}_auc_top_{avg_auc_top:.4f}_auc_least_{avg_auc_least:.4f}.png', dpi=300)

plt.close()

print(f"Top {top_features_count} Features AUC: {avg_auc_top:.4f}")
print(f"Least Associated {top_features_count} Features AUC: {avg_auc_least:.4f}")
print(f"Randomly Selected {top_features_count} Features AUC: {avg_auc_random:.4f}")
