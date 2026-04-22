import atexit
import os
import sys
from pathlib import Path

# Create local matplotliib cache to avoid errors if you dont have perms to write to default location
# Added this to ensure it can run on professors env too
matplotlib_config_dir = Path(__file__).resolve().parent / ".matplotlib-cache"
matplotlib_config_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# simple function to take all printed items and save to text file for easier readibility/referencing
class TeeOutput:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except ValueError:
                pass
output_file_path = Path(__file__).resolve().parent / "script_output.txt"
output_file_handle = output_file_path.open("w", encoding="utf-8")
original_stdout = sys.stdout
sys.stdout = TeeOutput(original_stdout, output_file_handle)

def close_output_file():
    sys.stdout = original_stdout
    output_file_handle.close()
atexit.register(close_output_file)


def show_or_close_plot():
    if "agg" in plt.get_backend().lower():
        plt.close()
    else:
        plt.show()

#####################################################
            # STEP 1: DATA EXPLORATION #
#####################################################
path = "./train/desc_rdk_train.csv"
df_desc = pd.read_csv(path)

print("\n*************DATA EXPLORATION*************")
print(df_desc.head())
print(df_desc.shape)
print(df_desc.info())
print(df_desc.describe())
print(df_desc.isna().sum().sort_values(ascending=False).head(20))
print(df_desc['Expt'].value_counts())

descriptor_features = df_desc.drop(columns=["CompName", "Expt"])
experiment_labels = df_desc["Expt"]

print(descriptor_features.shape)
print(experiment_labels.shape)
print("\n*************END OF DATA EXPLORATION*************")


#####################################################
            # STEP 2: DATA PREPROCESSING #
#####################################################
train_desc_ft, val_desc_ft, train_expt_labels, val_expt_labels = train_test_split(
    descriptor_features, experiment_labels,
    test_size=0.2,
    random_state=400,
    stratify=experiment_labels
)

variance = train_desc_ft.var()
non_constant = variance[variance > 0].index
train_desc_ft = train_desc_ft[non_constant]
val_desc_ft = val_desc_ft[non_constant]

print("\n*************DATA PREPROCESSING*************")
print(f"Features after removing constants: {len(non_constant)}")

print(f"Training set: {train_desc_ft.shape}")
print(f"Validation set: {val_desc_ft.shape}")
print(f"\nTraining class balance:\n{train_expt_labels.value_counts()}")
print(f"\nValidation class balance:\n{val_expt_labels.value_counts()}")

correlations = train_desc_ft.corrwith(train_expt_labels).abs()
corr_features = correlations[correlations >= 0.25].index.tolist()

print(f"Original number of features: {descriptor_features.shape[1]}")
print(f"Features after correlation filter: {len(corr_features)}")
print(f"\nSelected features: {corr_features}")

if len(corr_features) == 0:
    raise ValueError("Note: no features passed the correlation threshold")

filtered_train = train_desc_ft[corr_features]
filtered_val = val_desc_ft[corr_features]

print(f"Filtered training set shape: {filtered_train.shape}")
print(f"Filtered validation set shape: {filtered_val.shape}")

selected_corrs = correlations[correlations >= 0.25].sort_values(ascending=False)
print(selected_corrs)
print("\n*************END OF DATA PREPROCESSING*************")

#####################################################
    # STEP 3: DIMENSIONALITY REDUCTION
#####################################################
scaler = StandardScaler()
scaled_train = scaler.fit_transform(filtered_train)
scaled_val = scaler.transform(filtered_val)

pca = PCA(n_components=0.95, random_state=400)
pca_train = pca.fit_transform(scaled_train)
pca_val = pca.transform(scaled_val)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\n*************DIMENSONLIATY REDUCTION*************")
print("\nExplained variance by first 10 PCs:")
for i in range(min(10, len(cumulative_variance))):
    print(f"PC{i+1}: {cumulative_variance[i]:.3f}")
print(f"\nNumber of retained principal components: {pca.n_components_}")
print(f"PCA training set shape: {pca_train.shape}")
print(f"PCA validation set shape: {pca_val.shape}")
print("\n*************END OF DIMENSONLIATY REDUCTION*************")


#####################################################
# STEP 4: PIPELINE A — LOGISTIC REGRESSION (PCA)
#####################################################
log_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=400
)

log_model.fit(pca_train, train_expt_labels)

train_preds = log_model.predict(pca_train)
val_preds = log_model.predict(pca_val)

print("\n*************PIPELINE A: Logistic Regression (PCA)")
print(f"\nTraining Accuracy: {accuracy_score(train_expt_labels, train_preds):.3f}")
print(f"Validation Accuracy: {accuracy_score(val_expt_labels, val_preds):.3f}")
print("\nValidation Confusion Matrix:")
print(confusion_matrix(val_expt_labels, val_preds))
print("\nValidation Classification Report:")
print(classification_report(val_expt_labels, val_preds, digits=3))

pca_probs = log_model.predict_proba(pca_val)[:, 1]

fpr_pca, tpr_pca, _ = roc_curve(val_expt_labels, pca_probs)
auc_pca = roc_auc_score(val_expt_labels, pca_probs)

print(f"\nPCA ROC AUC: {auc_pca:.3f}")

plt.figure()
plt.plot(fpr_pca, tpr_pca, label=f"AUC = {auc_pca:.3f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - PCA Logistic Regression")
plt.legend()
show_or_close_plot()
print("\n*************END OF PIPELINE A: PCA*************")


#####################################################
# STEP 5: PIPELINE B — LASSO
####################################################
scaler_lasso = StandardScaler()

scaled_train_desc_ft = scaler_lasso.fit_transform(filtered_train)
scaled_val_desc_ft = scaler_lasso.transform(filtered_val)

lasso_model = LogisticRegression(
    solver="liblinear",
    l1_ratio=1.0,
    max_iter=1000,
    class_weight="balanced",
    random_state=400
)

lasso_model.fit(scaled_train_desc_ft, train_expt_labels)
lasso_coeff = lasso_model.coef_[0]

selected_ft_lasso = [
    feature for feature, coef in zip(filtered_train.columns, lasso_coeff)
    if coef != 0
]

lasso_import = pd.DataFrame({
    "Feature": filtered_train.columns,
    "Coefficient": lasso_coeff
})

lasso_import = lasso_import[lasso_import["Coefficient"] != 0]
lasso_import["AbsCoef"] = np.abs(lasso_import["Coefficient"])
lasso_import = lasso_import.sort_values(by="AbsCoef", ascending=False)

print("\nTop LASSO Features:")
print(lasso_import.head(10))

print("\nNumber of features selected by LASSO:", len(selected_ft_lasso))
print("\nSelected features:")
print(selected_ft_lasso)

lasso_train_preds = lasso_model.predict(scaled_train_desc_ft)
lasso_val_preds = lasso_model.predict(scaled_val_desc_ft)

print("\n*************PIPELINE B: LASSO (Logistic Regression)")
print(f"\nTraining Accuracy: {accuracy_score(train_expt_labels, lasso_train_preds):.3f}")
print(f"Validation Accuracy: {accuracy_score(val_expt_labels, lasso_val_preds):.3f}")
print("\nValidation Confusion Matrix:")
print(confusion_matrix(val_expt_labels, lasso_val_preds))
print("\nValidation Classification Report:")
print(classification_report(val_expt_labels, lasso_val_preds, digits=3))

lasso_probs = lasso_model.predict_proba(scaled_val_desc_ft)[:, 1]

fpr_lasso, tpr_lasso, thresholds_lasso = roc_curve(val_expt_labels, lasso_probs)
auc_lasso = roc_auc_score(val_expt_labels, lasso_probs)
print(f"\nLASSO ROC AUC: {auc_lasso:.3f}")

plt.figure()
plt.plot(fpr_lasso, tpr_lasso, label=f"AUC = {auc_lasso:.3f}")
plt.plot([0,1], [0,1], linestyle='--')  # random baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - LASSO Model (AUC = {auc_lasso:.3f})")
plt.legend()
show_or_close_plot()
print("\n*************END OF PIPELINE B: LASSO*************")


#####################################################
# STEP 6: PIPELINE C — RANDOM FOREST
#####################################################
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=400,
    class_weight="balanced"
)

rf_model.fit(filtered_train, train_expt_labels)

rf_train_preds = rf_model.predict(filtered_train)
rf_val_preds = rf_model.predict(filtered_val)

print("\n*************PIPELINE C: RANDOM FOREST*************")
print(f"\nTraining Accuracy: {accuracy_score(train_expt_labels, rf_train_preds):.3f}")
print(f"Validation Accuracy: {accuracy_score(val_expt_labels, rf_val_preds):.3f}")
print("\nValidation Confusion Matrix:")
print(confusion_matrix(val_expt_labels, rf_val_preds))
print("\nValidation Classification Report:")
print(classification_report(val_expt_labels, rf_val_preds, digits=3))

rf_probs = rf_model.predict_proba(filtered_val)[:, 1]

fpr_rf, tpr_rf, thresholds_rf = roc_curve(val_expt_labels, rf_probs)
auc_rf = roc_auc_score(val_expt_labels, rf_probs)

print(f"\nRandom Forest ROC AUC: {auc_rf:.3f}")

plt.figure()
plt.plot(fpr_rf, tpr_rf, label=f"AUC = {auc_rf:.3f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
show_or_close_plot()

rf_importances = rf_model.feature_importances_

rf_feature_importance = pd.DataFrame({
    "Feature": filtered_train.columns,
    "Importance": rf_importances
}).sort_values(by="Importance", ascending=False)
print("\nTop 10 Random Forest Important Features:")
print(rf_feature_importance.head(10))
print("\n*************END OF PIPELINE C: RANDOM FOREST*************")

#####################################################
# STEP 7: OVERALL MODEL COMPARISON
#####################################################

plt.figure()
plt.plot(fpr_pca, tpr_pca, label=f"PCA (AUC={auc_pca:.3f})")
plt.plot(fpr_lasso, tpr_lasso, label=f"LASSO (AUC={auc_lasso:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC={auc_rf:.3f})")
plt.plot([0,1], [0,1], linestyle='--', label="Random")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison Across Models")
plt.legend()
show_or_close_plot()

# Model Comparison Table
pca_report = classification_report(val_expt_labels, val_preds, output_dict=True)
lasso_report = classification_report(val_expt_labels, lasso_val_preds, output_dict=True)
rf_report = classification_report(val_expt_labels, rf_val_preds, output_dict=True)

comparison_df = pd.DataFrame({
    "Model": ["PCA", "LASSO", "Random Forest"],
    "AUC": [auc_pca, auc_lasso, auc_rf],
    "Accuracy": [
        accuracy_score(val_expt_labels, val_preds),
        accuracy_score(val_expt_labels, lasso_val_preds),
        accuracy_score(val_expt_labels, rf_val_preds)
    ],

    "Precision_0": [
        pca_report["0"]["precision"],
        lasso_report["0"]["precision"],
        rf_report["0"]["precision"]
    ],

    "Recall_0": [
        pca_report["0"]["recall"],
        lasso_report["0"]["recall"],
        rf_report["0"]["recall"]
    ],

    "Precision_1": [
        pca_report["1"]["precision"],
        lasso_report["1"]["precision"],
        rf_report["1"]["precision"]
    ],

    "Recall_1": [
        pca_report["1"]["recall"],
        lasso_report["1"]["recall"],
        rf_report["1"]["recall"]
    ]
})

comparison_df = comparison_df.round(2)
comparison_df.to_csv("model_comparison.csv", index=False)
print("\nMODEL COMPARISON TABLE:")
print(comparison_df)

#####################################################
# STEP 8: Use same preprocessing and RF model to
# train on training data + use external test set
#####################################################
