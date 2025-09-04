import pandas as pd 
import numpy as np 
import nltk 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.preprocessing import StandardScaler 
import email 
from email import policy 
import re 
import warnings 
import random 
import os 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
from io import StringIO 
# Note: 'Papa' and 'loadFileData' are assumed to be environment-specific. 
# If not available, we'll load the CSV directly for local execution. 
try: 
import Papa 
from Papa import loadFileData 
except ImportError: 
loadFileData = None 
warnings.filterwarnings('ignore') 
nltk.download('punkt') 
nltk.download('punkt_tab') 
nltk.download('stopwords') 
np.random.seed(42) 
random.seed(42) 
# Feature extraction functions 
def classify_attachments(attachments): 
if not attachments or pd.isna(attachments): 
return 0 
harmful_extensions = ['.exe', '.bat', '.ps1', '.vbs', '.wsf', '.js'] 
return 1 if any(ext.lower() in str(attachments) for ext in harmful_extensions) else 0 
def classify_urls(text): 
if not text or pd.isna(text): 
return 0 
urls 
= re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA
F][0-9a-fA-F]))+', str(text)) 
num_urls = len(urls) 
if num_urls > 3: 
return 3 
elif num_urls > 0: 
return 2 
return 0 
def extract_protocol_features(protocols): 
if not protocols or pd.isna(protocols): 
return 0 
protocols = str(protocols).lower() 
return 1 if 'ftp' in protocols or 'mailto' in protocols else 0 
def preprocess_text(text): 
if not text or pd.isna(text): 
return '' 
tokens = word_tokenize(str(text).lower()) 
stop_words = set(stopwords.words('english')) 
tokens = [t for t in tokens if t.isalpha() and t not in stop_words] 
return ' '.join(tokens) 
def detect_brands(text): 
if not text or pd.isna(text): 
return 0 
brands = ['amazon', 'paypal', 'microsoft', 'apple', 'google', 'facebook'] 
return 1 if any(brand in str(text).lower() for brand in brands) else 0 
def extract_header_risk(headers): 
if not headers or pd.isna(headers): 
return 0 
risky_domains = ['@yahoo.com', '@hotmail.com'] 
return 1 if any(domain in str(headers).lower() for domain in risky_domains) else 0 
# Function to parse .eml file 
def parse_eml_file(file_path): 
with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: 
msg = email.message_from_file(f, policy=policy.default) 
# Extract email text 
email_text = '' 
if msg.is_multipart(): 
for part in msg.walk(): 
if part.get_content_type() == 'text/plain': 
email_text += part.get_payload(decode=True).decode(errors='ignore') 
else: 
email_text = msg.get_payload(decode=True).decode(errors='ignore') 
# Extract attachments 
attachments = [] 
for part in msg.walk(): 
if part.get_content_disposition() == 'attachment': 
filename = part.get_filename() 
if filename: 
attachments.append(filename) 
# Extract headers 
headers = [str(msg.get(h, '')) for h in ['From', 'To', 'Reply-To']] 
# Count URLs for regression 
num_urls = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0
9a-fA-F][0-9a-fA-F]))+', email_text)) 
return { 
'email_text': preprocess_text(email_text), 
'attachments': ','.join(attachments) if attachments else '', 
'headers': ','.join(headers), 
'num_urls': num_urls 
} 
# Function to train classification and regression models 
def train_models(): 
# Load dataset 
csv_path = "spam10000_balanced_generated_1.csv" 
if loadFileData and os.path.exists(csv_path): 
csv_data = loadFileData(csv_path) 
df = pd.read_csv(StringIO(csv_data)) 
else: 
try: 
df = pd.read_csv(csv_path) 
except FileNotFoundError: 
print(f"Error: Dataset file '{csv_path}' not found.") 
return None, None, None, None 
# Handle missing values 
df['email_text'] = df['email_text'].fillna('') 
df['attachments'] = df['attachments'].fillna('') 
df['headers'] = df['headers'].fillna('') 
df['protocols'] = df['protocols'].fillna('') 
df['num_urls'] = df['num_urls'].fillna(0).astype(float) 
# Convert labels to binary 
df['label'] = df['label'].map({'spam': 1, 'ham': 0}) 
# Feature extraction 
df['attachment_risk'] = df['attachments'].apply(classify_attachments) 
df['url_risk'] = df['email_text'].apply(classify_urls) 
df['protocol_risk'] = df['protocols'].apply(extract_protocol_features) 
df['header_risk'] = df['headers'].apply(extract_header_risk) 
df['brand_risk'] = df['email_text'].apply(detect_brands) 
# Create risk score for regression 
df['risk_score'] = ( 
df['num_urls'].clip(0, 5) + 
df['attachment_risk'] * 2 + 
df['url_risk'] * 2 + 
df['protocol_risk'] + 
df['brand_risk']  
) 
# Prepare features 
tfidf = TfidfVectorizer(max_features=100, min_df=1) 
email_features = tfidf.fit_transform(df['email_text']).toarray() 
other_features = df[['attachment_risk', 'url_risk', 'protocol_risk', 'header_risk', 
'brand_risk']].values 
X = np.hstack([email_features, other_features]) 
y = df['label'].values 
y_reg = df['risk_score'].values 
# Scale features 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
# Train classification model 
rf_clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42) 
rf_clf.fit(X_scaled, y) 
# Train regression model 
rf_reg 
= 
random_state=42) 
RandomForestRegressor(n_estimators=50, 
rf_reg.fit(X_scaled, y_reg) 
return rf_clf, rf_reg, tfidf, scaler 
# Function to classify email and calculate risk level 
def classify_email(file_path, clf_model, reg_model, tfidf, scaler): 
if clf_model is None or reg_model is None: 
print("Error: Models not trained due to missing dataset.") 
return None 
email_data = parse_eml_file(file_path) 
# Extract features 
email_text = email_data['email_text'] 
attachments = email_data['attachments'] 
headers = email_data['headers'] 
num_urls = email_data['num_urls'] 
max_depth=5, 
attachment_risk = classify_attachments(attachments) 
url_risk = classify_urls(email_text) 
protocol_risk = extract_protocol_features(email_text) 
header_risk = extract_header_risk(headers) 
brand_risk = detect_brands(email_text) 
# Transform text features 
email_features = tfidf.transform([email_text]).toarray() 
other_features = np.array([[attachment_risk, url_risk, protocol_risk, header_risk, 
brand_risk]]) 
X = np.hstack([email_features, other_features]) 
# Scale features 
X_scaled = scaler.transform(X) 
# Classification prediction 
clf_prediction = clf_model.predict(X_scaled)[0] 
clf_probabilities = clf_model.predict_proba(X_scaled)[0] 
# Regression prediction 
reg_prediction = reg_model.predict(X_scaled)[0] 
# Calculate risk level (1-10) using regression score and feature presence 
risk_features = [attachment_risk, url_risk, protocol_risk, header_risk, brand_risk] 
feature_score = sum(2 if r > 0 else 0 for r in risk_features)  # Count features 
scaled_reg_score = min(max((reg_prediction / 10) * 5, 0), 7)  # Contribute up to 5 
points 
risk_level = min(round((feature_score * 2 )+ (scaled_reg_score * 2)), 10)  # Cap at 
return { 
'classification': 'spam' if risk_level > 4 else 'ham', 
'spam_probability': clf_probabilities[1], 
'risk_score': reg_prediction, 
'risk_level': risk_level, 
'features': { 
'attachment_risk': attachment_risk, 
'url_risk': url_risk, 
'protocol_risk': protocol_risk, 
'header_risk': header_risk, 
'brand_risk': brand_risk, 
'num_urls': num_urls 
} 
} 
# Main execution 
if __name__ == "__main__": 
# Train models 
clf_model, reg_model, tfidf, scaler = train_models() 
# Example usage: classify an .eml file 
eml_file_path = input("Enter the path to the .eml file: ") 
if os.path.exists(eml_file_path): 
result = classify_email(eml_file_path, clf_model, reg_model, tfidf, scaler) 
if result: 
print(f"Classification: {result['classification']}") 
print(f"Risk Level (1-10): {result['risk_level']}") 
print("Contributing Features:") 
for feature, value in result['features'].items(): 
print(f"  {feature}: {value}") 
else: 
print("File not found!")
