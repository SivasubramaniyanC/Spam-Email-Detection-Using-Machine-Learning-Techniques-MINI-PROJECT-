import pandas as pd 
import numpy as np 
import nltk 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.metrics import roc_curve, auc, f1_score, classification_report, 
mean_absolute_error, mean_squared_error, r2_score, confusion_matrix 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 
import random 
warnings.filterwarnings('ignore') 
nltk.download('punkt') 
nltk.download('punkt_tab') 
nltk.download('stopwords') 
np.random.seed(42) 
random.seed(42) 
# Feature extraction functions 
def classify_attachments(attachments): 
if pd.isna(attachments) or attachments == '': 
return 0 
harmful_extensions = ['.exe', '.bat', '.ps1', '.vbs', '.wsf', '.js'] 
attachments = [att.strip() for att in attachments.split(',')] 
return 1 if any(att in harmful_extensions for att in attachments) else 0 
def classify_urls(num_urls): 
if pd.isna(num_urls): 
return 0 
num_urls = float(num_urls) 
if num_urls > 3: 
return 2 
elif num_urls > 0: 
return 1 
return 0 
def extract_protocol_features(protocols): 
if pd.isna(protocols) or protocols == '': 
return 0 
protocols = [p.strip() for p in protocols.split(',')] 
risky_protocols = ['ftp', 'mailto'] 
return 1 if any(p in risky_protocols for p in protocols) else 0 
def preprocess_text(text): 
if pd.isna(text): 
return '' 
tokens = word_tokenize(text.lower()) 
stop_words = set(stopwords.words('english')) 
tokens = [t for t in tokens if t.isalpha() and t not in stop_words] 
return ' '.join(tokens) 
def detect_brands(text): 
brands = ['amazon', 'paypal', 'microsoft', 'apple', 'google', 'facebook'] 
return 1 if any(brand in text.lower() for brand in brands) else 0 
# Load and preprocess dataset 
df = pd.read_csv('spam10000_balanced_generated_1.csv') 
df['email_text'] = df['email_text'].apply(preprocess_text) 
df['attachment_risk'] = df['attachments'].apply(classify_attachments) 
df['num_urls'] = df['num_urls'].apply(lambda x: float(x) + np.random.normal(0, 0.5) if 
not pd.isna(x) else 0) 
df['url_risk'] = df['num_urls'].apply(classify_urls) 
df['protocol_risk'] = df['protocols'].apply(extract_protocol_features) 
df['header_risk'] = df['headers'].str.contains('@yahoo.com|@hotmail.com', case=False, 
na=False).astype(int) 
df['brand_risk'] = df['email_text'].apply(detect_brands) 
# Add regression target 
df['risk_score'] = df['num_urls'].clip(0, 5) + df['attachment_risk'] * 2 + df['url_risk'] * 
2 + df['protocol_risk'] + df['brand_risk'] 
# Label noise 
df['label'] = df['label'].map({'spam': 1, 'ham': 0}) 
label_noise = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05]) 
df['label'] = df['label'] ^ label_noise 
# Feature extraction 
tfidf = TfidfVectorizer(max_features=200, min_df=5) 
email_features = tfidf.fit_transform(df['email_text']).toarray() 
other_features = df[['attachment_risk', 'url_risk', 'protocol_risk', 'header_risk', 
'brand_risk']].values 
X = np.hstack([email_features, other_features]) 
y = df['label'].values 
y_reg = df['risk_score'].values 
# Add noise to features 
X = X + np.random.normal(0, 0.1, X.shape) 
# Split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, 
test_size=0.2, random_state=42) 
# Scale features 
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 
X_train_reg_scaled = scaler.fit_transform(X_train_reg) 
X_test_reg_scaled = scaler.transform(X_test_reg) 
# Train models 
rf = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=20, 
random_state=42) 
rf.fit(X_train_scaled, y_train) 
rfr = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42) 
rfr.fit(X_train_reg_scaled, y_train_reg) 
# Classification evaluation 
y_pred_rf = rf.predict(X_test_scaled) 
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1] 
print("Random Forest Classification Report:") 
print(classification_report(y_test, y_pred_rf, target_names=['Ham', 'Spam'])) 
# Confusion Matrix 
cm = confusion_matrix(y_test, y_pred_rf) 
plt.figure(figsize=(8, 6)) 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], 
yticklabels=['Ham', 'Spam']) 
plt.title('Confusion Matrix - Random Forest') 
plt.xlabel('Predicted') 
plt.ylabel('Actual') 
plt.show() 
# Regression evaluation - Random Forest 
y_pred_reg_rf = rfr.predict(X_test_reg_scaled) 
print("\nRandom Forest Regression Metrics:") 
print(f"Regression MAE: {mean_absolute_error(y_test_reg, y_pred_reg_rf):.4f}") 
print(f"Regression MSE: {mean_squared_error(y_test_reg, y_pred_reg_rf):.4f}") 
print(f"Regression R2: {r2_score(y_test_reg, y_pred_reg_rf):.4f}") 
# Cross-validation for Random Forest Classifier 
cv_scores_rf = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='f1') 
print(f"\nRandom 
Forest 
{cv_scores_rf.std():.4f}") 
# F1-Score for Random Forest 
CV F1 Scores: {cv_scores_rf.mean():.4f} ± 
print(f"F1 Score (Random Forest): {f1_score(y_test, y_pred_rf):.4f}") 
# Feature importance 
importances = rf.feature_importances_ 
feature_names = tfidf.get_feature_names_out().tolist() + ['attachment_risk', 'url_risk', 
'protocol_risk', 'header_risk', 'brand_risk'] 
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}) 
importance_df = importance_df.sort_values('Importance', ascending=False).head(10) 
plt.figure(figsize=(8, 6)) 
sns.barplot(data=importance_df, x='Importance', y='Feature') 
plt.title('Top 10 Feature Importances') 
plt.show() 
# ROC Curve 
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf) 
roc_auc_rf = auc(fpr_rf, tpr_rf) 
plt.figure(figsize=(8, 6)) 
plt.plot(fpr_rf, tpr_rf, label=f'RF ROC Curve (AUC = {roc_auc_rf:.2f})') 
plt.plot([0, 1], [0, 1], 'k--') 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve') 
plt.legend() 
plt.show() 
# PCA Visualization 
pca = PCA(n_components=2) 
X_pca = pca.fit_transform(X_train_scaled) 
pca_df = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Label': y_train}) 
plt.figure(figsize=(8, 6)) 
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Label', palette=['blue', 'red'], 
alpha=0.5) 
plt.title('PCA Visualization of Email Features') 
plt.legend(['Ham', 'Spam']) 
plt.show()
