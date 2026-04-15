import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SKLDA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, f1_score
from scipy import stats

# --- CELL 1 ---

sns.set_style("whitegrid")
try:
    df = pd.read_csv('AswanData_weatherdata.csv')
except FileNotFoundError:
    print("Error: 'AswanData_weatherdata.csv' not found. Please ensure the file is correctly placed.")
    exit()

df = df.drop('Date', axis=1)
df['Solar_Class'] = pd.qcut(df['Solar(PV)'], q=3, labels=['Low', 'Medium', 'High'])
df = df.drop('Solar(PV)', axis=1)

X = df.drop('Solar_Class', axis=1)
y = df['Solar_Class']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
print(f"Target classes encoded: {le.classes_}")

# --- CELL 2 ---

print("\n--- Missing Values ---")
print(X.isnull().sum())
for col in X.columns:
    if X[col].isnull().any():
        X[col].fillna(X[col].mean(), inplace=True)

print("\n--- Descriptive Statistics ---")
print(X.describe())

print("\n--- Skewness and Kurtosis ---")
print(X.skew())
print(X.kurt())

# --- CELL 3 ---

print("\n--- Covariance Matrix ---")
print(X.cov())
print("\n--- Correlation Matrix ---")
corr_matrix = X.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

print("\n--- ANOVA Tests (Feature vs. Solar_Class) ---")
for col in X.columns:
    try:
        groups = [df[df['Solar_Class'] == name][col] for name in class_names]
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"ANOVA for {col}: F-stat={f_stat:.4f}, P-value={p_value:.4f}")
    except ValueError as e:
        print(f"Could not perform ANOVA for {col}: {e}")

# --- CELL 4 ---

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA Explained Variance (2 components): {pca.explained_variance_ratio_.sum():.4f}")

lda_sk = SKLDA(n_components=2)
X_lda = lda_sk.fit_transform(X_scaled, y_encoded)
print(f"LDA Discriminability components created: {X_lda.shape[1]}")

# --- CELL 5 ---

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

models = {
    'Naive Bayesian': GaussianNB(),
    'Decision Tree (Entropy)': DecisionTreeClassifier(criterion='entropy', random_state=42),
    'K-NN (Euclidean)': KNeighborsClassifier(n_neighbors=5, p=2),
    'K-NN (Manhattan)': KNeighborsClassifier(n_neighbors=5, p=1),
    'LDA Classifier': SKLDA(),
    'PCA + K-NN': KNeighborsClassifier(n_neighbors=5),
}
results = {}
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

print("\n--- 5. MODEL IMPLEMENTATION AND EVALUATION ---")

# --- CELL 6 ---

for name, model in models.items():
    print(f"\n--- Model: {name} ---")

    if 'PCA' in name:
        X_train_model = PCA(n_components=2).fit_transform(X_train)
        X_test_model = PCA(n_components=2).fit(X_train).transform(X_test)
        cv_scores = cross_val_score(model, X_train_model, y_train, cv=kfold, scoring='accuracy')
    elif 'LDA Classifier' in name:
        X_train_model = SKLDA(n_components=2).fit_transform(X_train, y_train)
        X_test_model = SKLDA(n_components=2).fit(X_train, y_train).transform(X_test)
        cv_scores = cross_val_score(model, X_train_model, y_train, cv=kfold, scoring='accuracy')
    else:
        X_train_model = X_train
        X_test_model = X_test
        cv_scores = cross_val_score(model, X_train_model, y_train, cv=kfold, scoring='accuracy')

    results[name] = {'CV_Accuracy': cv_scores.mean()}
    print(f"K-fold Cross-Validation Avg Accuracy: {cv_scores.mean():.4f}")

    model.fit(X_train_model, y_train)
    y_pred = model.predict(X_test_model)

    acc = accuracy_score(y_test, y_pred)
    err_rate = 1 - acc
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    results[name].update({
        'Test_Accuracy': acc,
        'Error_Rate': err_rate,
        'Precision': report['macro avg']['precision'],
        'Recall': report['macro avg']['recall'],
        'F1_Score': f1,
    })

    print(f"Test Accuracy: {acc:.4f}, Error Rate: {err_rate:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    results[name]['Confusion_Matrix'] = cm
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {name}')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()

    train_acc = model.score(X_train_model, y_train)
    print(f"Training Accuracy: {train_acc:.4f}")
    if (train_acc - acc) > 0.1:
        results[name]['Overfit_Analysis'] = "Potential Overfitting (Train Acc >> Test Acc)"
    elif acc < 0.6:
        results[name]['Overfit_Analysis'] = "Potential Underfitting (Low Acc)"
    else:
        results[name]['Overfit_Analysis'] = "Balanced (Train Acc ≈ Test Acc)"
    print(f"Overfitting Analysis: {results[name]['Overfit_Analysis']}")

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_model)
        plt.figure(figsize=(7, 7))
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC curve {class_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (One-vs-Rest) for {name}')
        plt.legend(loc="lower right")
        plt.show()

# --- CELL 7: Bayesian-like Probabilistic Prediction using GaussianNB (Sklearn) ---

print("\n--- Probabilistic Model using Gaussian Naive Bayes (Sklearn) ---")
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred_nb = gnb.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

# --- FIXED SAMPLE PREDICTION ---
# Use mean of each feature column
sample_input = X_train.mean(axis=0).to_numpy().reshape(1, -1)

print("\nSample Input (mean of features):")
print(sample_input)

prob_pred = gnb.predict_proba(sample_input)
print("\nPredicted probabilities for sample input:", prob_pred)



# --- CELL 8: Final Comparison Chart ---

comparison_df = pd.DataFrame({k: v for k, v in results.items() if 'Test_Accuracy' in v}).T[['Test_Accuracy', 'F1_Score', 'Error_Rate']]
comparison_df = comparison_df.dropna()

print("\n--- FINAL MODEL COMPARISON TABLE ---")
print(comparison_df.sort_values(by='Test_Accuracy', ascending=False))

plt.figure(figsize=(10, 6))
comparison_df['Test_Accuracy'].sort_values().plot(kind='barh', color='skyblue')
plt.title('Comparison of Model Test Accuracies')
plt.xlabel('Test Accuracy')
plt.show()