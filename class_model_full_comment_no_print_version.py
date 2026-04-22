import os
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


def show_or_close_plot():
    if "agg" in plt.get_backend().lower():
        plt.close()
    else:
        plt.show()

#####################################################
            # STEP 1: DATA EXPLORATION #
#####################################################
# Loading dataset from training directory
path = "./train/desc_rdk_train.csv"
df_desc = pd.read_csv(path)


# Inspecting data: first few rows, shape & number of unique vals
# print(df_desc.head())
# print(df_desc.shape)
# print(df_desc.info())
# print(df_desc.describe())
# print(df_desc.isna().sum().sort_values(ascending=False).head(20))
# print(df_desc['Expt'].value_counts())

# There are 1,465 molecules with 121 columns
# Expt column has 1,132 molecules in Class 1 and 333 molecules in Class 0
# 77% of data is Class 1, 23% of data is Class 0
# no nulls

descriptor_features = df_desc.drop(columns=["CompName", "Expt"])
experiment_labels = df_desc["Expt"]

# print(descriptor_features.shape) 
# print(experiment_labels.shape) 

# 1465 molecules, 119 descriptor features (after dropping CompName and Expt)
# 1465 molecules, 1 column (Expt)

#####################################################
            # STEP 2: DATA PREPROCESSING #
#####################################################
train_desc_ft, val_desc_ft, train_expt_labels, val_expt_labels = train_test_split(
    descriptor_features, experiment_labels, 
    test_size=0.2, 
    random_state=42,# For reproducibility
    stratify=experiment_labels
)

# Compute variance on training set only
variance = train_desc_ft.var()

# Keep only features with non-zero variance
non_constant = variance[variance > 0].index

# Apply to both train and validation sets
train_desc_ft = train_desc_ft[non_constant]
val_desc_ft = val_desc_ft[non_constant]

# print(f"Features after removing constants: {len(non_constant)}")

# print(f"Training set: {train_desc_ft.shape}")
# 1172 molecules, 119 descriptor features
# print(f"Validation set: {val_desc_ft.shape}")
# 293 molecules, 119 descriptor features
# print(f"\nTraining class balance:\n{train_expt_labels.value_counts()}")
# Training set: 1172 molecules. 906 in Class 1 and 266 in Class 0 (77% Class 1, 23% Class 0)
# print(f"\nValidation class balance:\n{val_expt_labels.value_counts()}")
# Validation set: 293 molecules. 226 in Class 1 and 67 in Class 0 (77% Class 1, 23% Class 0)

# Only grab features whose correlation magnitude is >= 0.25 with the target variabl
correlations = train_desc_ft.corrwith(train_expt_labels).abs()
corr_features = correlations[correlations >= 0.25].index.tolist()

# print(f"Original number of features: {descriptor_features.shape[1]}")
# print(f"Features after correlation filter: {len(corr_features)}")
# print(f"\nSelected features: {corr_features}")

# Original number of features: 119
# Features after correlation filter: 58

# If no features meet the correlation threshold, raise an error
if len(corr_features) == 0:
    raise ValueError("Note: no features passed the correlation threshold")

# Apply the same selected features to training and validation sets
filtered_train = train_desc_ft[corr_features]
filtered_val = val_desc_ft[corr_features]

# print(f"Filtered training set shape: {filtered_train.shape}")
# print(f"Filtered validation set shape: {filtered_val.shape}")

# filtered training set shape: (1172, 58)
# filtered validation set shape: (293, 58)

# View selected features and their correlation values with the target variable
selected_corrs = correlations[correlations >= 0.25].sort_values(ascending=False)
# print(selected_corrs)

# Top 5 features with correlation >= 0.25:
 # NumLipinskiHBA = 0.598396
 # MQN21 = 0.598347
 # TPSA = 0.593393
 # NumHeteroatoms = 0.578047
 # MQN20 = 0.570963

# The strongest predictors are chemically interpretable and align with known BBB permeability factors, such as hydrogen bond acceptors (NumLipinskiHBA), polar surface area (TPSA), and heteroatom count (NumHeteroatoms). This suggests that the model may be capturing meaningful chemical properties related to BBB permeability rules.
# There are 58 features, but many are highly correlated with each other, so we will consider PCA for dimensionality reduction in the next step.

filtered_train = filtered_train.drop(columns=["NumLipinskiHBA"])
filtered_val = filtered_val.drop(columns=["NumLipinskiHBA"])

#####################################################
    # STEP 3: DIMENSIONALITY REDUCTION
#####################################################
scaler = StandardScaler()
scaled_train = scaler.fit_transform(filtered_train)
scaled_val = scaler.transform(filtered_val)

pca = PCA(n_components=0.95, random_state=42)
pca_train = pca.fit_transform(scaled_train)
pca_val = pca.transform(scaled_val)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# print("\nExplained variance by first 10 PCs:")
# for i in range(min(10, len(cumulative_variance))):
#   print(f"PC{i+1}: {cumulative_variance[i]:.3f}")

# print(f"\nNumber of retained principal components: {pca.n_components_}")

# print(f"PCA training set shape: {pca_train.shape}")
# print(f"PCA validation set shape: {pca_val.shape}")

# The descriptor space was reduced from 119 original descriptors to 15 principal components while retaining approximately 95% of the variance.

#####################################################
# STEP 4: PIPELINE A — LOGISTIC REGRESSION (PCA)
#####################################################
log_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

# Train the model on the PCA-transformed training data
log_model.fit(pca_train, train_expt_labels)

# Make predictions on training and validation sets
train_preds = log_model.predict(pca_train)
val_preds = log_model.predict(pca_val)

# Evaluate the model's performance using:
    # accuracy
    # confusion matrix
    # classification report
# print("\n*************PIPELINE A: Logistic Regression (PCA)")

# print(f"\nTraining Accuracy: {accuracy_score(train_expt_labels, train_preds):.3f}")
# print(f"Validation Accuracy: {accuracy_score(val_expt_labels, val_preds):.3f}")

# Training Accuracy score: 0.875
# Validation Accuracy score: 0.863
# Based on the training accuracy score and the validation accuracy score:
    # The model is generalizing well
    # The model is not overfitting due to the close scores
    # The model is not underfitting due to the relatively high scores
    # The model overall learned relevant data patterns

#print("\nValidation Confusion Matrix:")
#print(confusion_matrix(val_expt_labels, val_preds))

# Based on the confusion matrix:
    # True Negatives: 49
    # False Positives: 18
    # False Negatives: 22
    # True Positives: 204
    # From the output of the confusion matrix, the model is performing better at correctly identifying Class 1 (BBB permeable) molecules than Class 0 (non-permeable). 
    # The model has a higher number of true positives (204) compared to true negatives (49), indicating it is more effective at predicting the positive class.
    # There are still a significant number of false positives (18) and false negatives (22), suggesting there can be some improvement in distinguishing between the two classes, especially for Class 0.

#print("\nValidation Classification Report:")
#print(classification_report(val_expt_labels, val_preds, digits=3))

# Based on the classification report:
    # Precision for Class 0: 0.690
    # Recall for Class 0: 0.731
    # F1-score for Class 0: 0.710
    # Class 0 Interpretation: 
        # Performs ok for the minority class
        # recall > 0.7 means it correctly identifies most of the non-permeable molecules
        # precision < 0.7 means that some of the molecules it predicts as non-permeable are actually permeable

    # Precision for Class 1: 0.919
    # Recall for Class 1: 0.903
    # F1-score for Class 1: 0.911
        # Class 1 Interpretation:
            # Performs well for the majority class
            # recall > 0.9 means it correctly identifies most of the permeable molecules
            # precision > 0.9 means that most of the molecules it predicts as permeable are actually permeable

    # Macro vs. Weighted Avg Interpretation:
        # Slightly biased towards class 1, but overall well

pca_probs = log_model.predict_proba(pca_val)[:, 1]

fpr_pca, tpr_pca, _ = roc_curve(val_expt_labels, pca_probs)
auc_pca = roc_auc_score(val_expt_labels, pca_probs)

#print(f"\nPCA ROC AUC: {auc_pca:.3f}")

plt.figure()
plt.plot(fpr_pca, tpr_pca, label=f"AUC = {auc_pca:.3f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - PCA Logistic Regression")
plt.legend()
show_or_close_plot()


#####################################################
# OVERALL INTERPRETATION OF PIPELINE A:
#####################################################
    # The model performs better on Class 1 than Class 0, with higher precision, recall, and F1-score for the positive class.
    
    # Weaknesses of PCA Pipeline:
        # 1. Loss of Interpretability:
            # PCA transforms the original descriptors into principal components, which are linear combinations of features.
            # This makes it difficult to interpret which specific chemical descriptors are driving model predictions.
            #  How to address: LASSO-based models can address this limitation by selecting a subset of original features, allowing for more direct interpretability.

        # 2. Linear Model Limitation:
            # Logistic regression assumes a linear relationship between the transformed features and the log-odds of the outcome.
            # This may limit the model’s ability to capture more complex, non-linear relationships between molecular descriptors and BBB permeability.
            # How to address: Non-linear models such as Support Vector Machines (SVM) or Random Forests can better capture these complex relationships.

        # 3. Reduced Performance on Minority Class (Class 0):
            # The model performs worse on Class 0 compared to Class 1 - class imbalance most likely
            # This is reflected in lower precision, recall, and F1-score for the minority class.
            # This issue can be addressed by:
                # applying LASSO to better isolate informative features for the minority class
                # or using models like Random Forest that are more robust to class imbalance.

        # 4. Potential Loss of Discriminative Information:
            # PCA retains components based on variance, not predictive power.
            # As a result, some components that explain high variance may not be the most informative for classification.
            # LASSO, in contrast, selects features based on their contribution to prediction performance.

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
    random_state=42
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

#print("\nTop LASSO Features:")
#print(lasso_import.head(10))
# Shows more hydrogen bond donors -> worse BBB penetration
# Larger/heavier molecules -> worse BBB penetration
# With the strongest positive predictors, we can tell that structural features are important

#print("\nNumber of features selected by LASSO:", len(selected_ft_lasso))
#print("\nSelected features:")
#print(selected_ft_lasso)
# Number of features selected by LASSO: 45

lasso_train_preds = lasso_model.predict(scaled_train_desc_ft)
lasso_val_preds = lasso_model.predict(scaled_val_desc_ft)

# Evaluation
#print("\n*************PIPELINE B: LASSO (Logistic Regression)")

#print(f"\nTraining Accuracy: {accuracy_score(train_expt_labels, lasso_train_preds):.3f}")
#print(f"Validation Accuracy: {accuracy_score(val_expt_labels, lasso_val_preds):.3f}")
# Training Accuracy: 0.899
# Validation Accuracy: 0.884

#print("\nValidation Confusion Matrix:")
#print(confusion_matrix(val_expt_labels, lasso_val_preds))
# Based on the confusion matrix:
    # True Negatives: 55
    # False Positives: 12
    # False Negatives: 22
    # True Positives: 204
    # Interpretation of matrix: LASSO performs better than PCA in correctly identifying Class 0 molecules (55 true negatives vs. 49 in PCA) and has fewer false positives (12 vs. 18 in PCA).      # Same number of false negatives (22) and true positives (204) as the PCA model.

#print("\nValidation Classification Report:")
#print(classification_report(val_expt_labels, lasso_val_preds, digits=3))
# Based on the classification report:
    # Precision for Class 0: 0.714
    # Recall for Class 0: 0.821
    # F1-score for Class 0: 0.764
    # Class 0 Interpretation:
        # Much better at learning minority class (recall 0.821 vs. 0.731 in PCA)
        # Precision also improved (0.714 vs. 0.690 in PCA), meaning fewer false positives for Class 0
        # F1score slightly improved (0.764 vs. 0.710 in PCA)
    # Precision for Class 1: 0.944
    # Recall for Class 1: 0.903
    # F1-score for Class 1: 0.923
    # Class 1 Interpretation:
        # Slightly better than PCA for Class 1 as well (precision 0.944 vs. 0.919 in PCA)
        # Recall is the same as PCA (0.903)
        # F1-score improved (0.923 vs. 0.911 in PCA)
    
    # Macro vs. Weighted Avg Interpretation:
        # Slightly biased towards Class 1

# Grab probabilities for ROC/AUC analysis to evaluate model's ability to discriminate between classes across different thresholds
lasso_probs = lasso_model.predict_proba(scaled_val_desc_ft)[:, 1]

# ROC / AUC
fpr_lasso, tpr_lasso, thresholds_lasso = roc_curve(val_expt_labels, lasso_probs)
auc_lasso = roc_auc_score(val_expt_labels, lasso_probs)
#print(f"\nLASSO ROC AUC: {auc_lasso:.3f}")

# Plot ROC curve
plt.figure()
plt.plot(fpr_lasso, tpr_lasso, label=f"AUC = {auc_lasso:.3f}")
plt.plot([0,1], [0,1], linestyle='--')  # random baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - LASSO Model (AUC = {auc_lasso:.3f})")
plt.legend()
show_or_close_plot()

#####################################################
# OVERALL INTERPRETATION OF PIPELINE B: LASSO
#####################################################
# Strengths of LASSO Pipeline:
    # 1. Improved Interpretability:
        # The model retains original chemical descriptors, enabling direct interpretation of feature importance.
    
    # 2. Embedded Feature Selection:
        # LASSO performs automatic feature selection by shrinking less important coefficients to zero,
        # reducing model complexity while maintaining predictive performance.
    
    # 3. Strong Performance on Both Classes:
        # The model achieves high recall for both classes, including the minority class (Class 0),
        # indicating effective detection of non-permeable molecules.
    
    # 4. Good Generalization:
        # The small gap between training and validation accuracy suggests the model is not overfitting.

# Limitations of LASSO Pipeline:
    # 1. Linear Model Assumption:
        # The model assumes a linear relationship between descriptors and the log-odds of BBB permeability,
        # which may limit its ability to capture complex, non-linear relationships.
    
    # 2. Partial Feature Reduction:
        # Although LASSO reduces the number of features, a relatively large number of descriptors (45)
        # are still retained, which may indicate that many features contribute to the prediction.
    
    # 3. Sensitivity to Regularization Strength:
        # The number of selected features depends on the regularization parameter, and different values
        # may lead to different subsets of features.

#####################################################
# STEP 6: PIPELINE C — RANDOM FOREST
#####################################################
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(filtered_train, train_expt_labels)

rf_train_preds = rf_model.predict(filtered_train)
rf_val_preds = rf_model.predict(filtered_val)

#print("\n*************PIPELINE C: RANDOM FOREST")

#print(f"\nTraining Accuracy: {accuracy_score(train_expt_labels, rf_train_preds):.3f}")
#print(f"Validation Accuracy: {accuracy_score(val_expt_labels, rf_val_preds):.3f}")
# Training Accuracy: 0.998
# Validation Accuracy: 0.887
# From these values we can tell there is overfitting as training accuracy is high and validation is lower

#print("\nValidation Confusion Matrix:")
#print(confusion_matrix(val_expt_labels, rf_val_preds))
# Based on the confusion matrix:
    # True Negatives: 43
    # False Positives: 24
    # False Negatives: 9
    # True Positives: 217
    # Strong at detecting class 1 but weak with class 0

#print("\nValidation Classification Report:")
#print(classification_report(val_expt_labels, rf_val_preds, digits=3))
# Based on the classification report:
    # Class 0 precision: 0.827
    # Class 0 recall: 0.642 - misses ~35% of non-permeable molecules
    # Class 0 F1-score: 0.723
    # Class 1 precision: 0.900
    # Class 1 recall: 0.960 - strong at identifying permeable molecules
    # Class 1 F1-score: 0.929 

# ROC / AUC
rf_probs = rf_model.predict_proba(filtered_val)[:, 1]

fpr_rf, tpr_rf, thresholds_rf = roc_curve(val_expt_labels, rf_probs)
auc_rf = roc_auc_score(val_expt_labels, rf_probs)

#print(f"\nRandom Forest ROC AUC: {auc_rf:.3f}")

plt.figure()
plt.plot(fpr_rf, tpr_rf, label=f"AUC = {auc_rf:.3f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
show_or_close_plot()

# From the plot, we can see that the model has strong discriminative ability (AUC 0.913)
# There is a strong separation between the classes, but the model is still overfitting.

rf_importances = rf_model.feature_importances_

rf_feature_importance = pd.DataFrame({
    "Feature": filtered_train.columns,
    "Importance": rf_importances
}).sort_values(by="Importance", ascending=False)

#print("\nTop 10 Random Forest Important Features:")
#print(rf_feature_importance.head(10))
# TPSA (0.0889) is the strongest feature for polarity/hydrogen bonding. High polarity -> harder BBB penetration
# Lipophilicity driven by SlogP and slogp_VSA2 is a driver of BBB permaeability
# Molecular Structure like MQN21, MQN20 and MQN23 show importance
# From this we can overall tell that BBB permeability depends on polarity, lipophilicity and molecular structure

#####################################################
# OVERALL INTERPRETATION OF PIPELINE C: RANDOM FOREST
#####################################################
# Strengths of Random Forset Pipeline:
    # 1. Non-linear Modeling Capability: Can capture interactions between descriptors that linear models may miss
    # 2. Strong Performance on Class 1: High recall for permeable molecules
    # 3. Robust to scaled amount of features, so can use less deimnsionality reduction
    # 4. High AUC: Indicates strong discriminative ability between classes
# Limitations of Random Forest Pipeline:
    # 1. Overfitting: Very high training accuracy but lower validation
    # 2. Weaker perfomance on class 0, lower recall (0.642)
    # 3. Difficult to tell which descriptors are most important for prediction
    # 4. Sensitive to class imbalance depite class weights
    #5. Computationally expensive

#####################################################
# STEP 7: OVERALL MODEL COMPARISON
#####################################################

# combined ROC for comparison
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
# PCA metrics
pca_report = classification_report(val_expt_labels, val_preds, output_dict=True)
# LASSO metrics
lasso_report = classification_report(val_expt_labels, lasso_val_preds, output_dict=True)
# RF metrics
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
#print("\nMODEL COMPARISON TABLE:")
#print(comparison_df)

# Interpretation of Model Comparison:
    # We need to decide whether its better to predict non-permeable molecules or permeable molecules
    # Because of the context misclassifying non-permeable molecules is more costly. 
    # We want to prioritize recall for Class 0 in this case and find balance with AUC & Accuracy
    # LASSO has the best recall and only has .01 point difference in AUC and accuracy compared to RF, so LASSO is the best model.

#####################################################
# STEP 8: TRAIN MODEL
#####################################################
df_full_train = pd.read_csv("./train/desc_rdk_train.csv")

full_train_desc_ft = df_full_train.drop(columns=["CompName", "Expt"])
full_train_expt_label = df_full_train["Expt"]

# Applying same preprocessing as before
variance_full = full_train_desc_ft.var()
non_constant_full = variance_full[variance_full > 0].index
full_train_desc_ft = full_train_desc_ft[non_constant_full]
correlations_full = full_train_desc_ft.corrwith(full_train_expt_label).abs()
corr_features_full = correlations_full[correlations_full >= 0.25].index.tolist()

if len(corr_features_full) == 0:
    raise ValueError("No features passed the correlation threshold on full training data.")

full_train_desc_ft = full_train_desc_ft[corr_features_full]

if "NumLipinskiHBA" in full_train_desc_ft.columns:
    full_train_desc_ft = full_train_desc_ft.drop(columns=["NumLipinskiHBA"])

scaler_final = StandardScaler()
scaled_full_train_desc_ft = scaler_final.fit_transform(full_train_desc_ft)

# Training on full training data with LASSO
lasso_final = LogisticRegression(
    solver="liblinear",
    l1_ratio=1.0,
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

lasso_final.fit(scaled_full_train_desc_ft, full_train_expt_label)

# Using external test set
df_test = pd.read_csv("./external_test/desc_rdk_test.csv")
test_output = df_test.copy()
test_desc_ft = df_test.drop(columns=["CompName"])

# Apply same preprocessing to test set
test_desc_ft = test_desc_ft[non_constant_full]
test_desc_ft = test_desc_ft[corr_features_full]
if "NumLipinskiHBA" in test_desc_ft.columns:
    test_desc_ft = test_desc_ft.drop(columns=["NumLipinskiHBA"])
scaled_test_desc_ft = scaler_final.transform(test_desc_ft)

# Make predictions on test set
test_preds = lasso_final.predict(scaled_test_desc_ft)
test_output["Expt"] = test_preds

# Save-*****
test_output.to_csv("desc_rdk_test_predictions.csv", index=False)
#print(test_output.head())
print("\nScript Completed Run! :)")


# Look accurate, the model predictted 336 class 1 and 123 class 0
# That's about 73% class 1 and 27% class 0, which is similar to the training data distribution (77% class 1 and 23% class 0)
# Model isn't showing overt errors
