import os 
import re 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from email import policy 
from email.parser import BytesParser 
from bs4 import BeautifulSoup 
from urllib.parse import urlparse 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report 
from sklearn.preprocessing import LabelEncoder 
from sklearn.decomposition import PCA 
from sklearn.tree import plot_tree 
 
def extract_email_details(eml_file): 
    with open(eml_file, "rb") as f: 
        msg = BytesParser(policy=policy.default).parse(f) 
     
    email_text = "" 
    hyperlinks = 0 
    urls = 0 
    attachments = [] 
    protocols = set() 
    headers = {"From": msg["From"], "To": msg["To"], "Date": msg["Date"]} 
     
    dangerous_extensions = {".exe", ".bat", ".vbs", ".js", ".ps1", ".wsf"} 
    spam_flag = False 
 
     
    if msg.is_multipart(): 
        for part in msg.walk(): 
            content_type = part.get_content_type() 
            content_disposition = part.get("Content-Disposition", "") 
             
            if "attachment" in content_disposition: 
                filename = part.get_filename() 
                if filename: 
                    attachments.append(filename) 
                    if any(filename.lower().endswith(ext) for ext in dangerous_extensions): 
                        spam_flag = True 
            elif content_type == "text/plain": 
                email_text += part.get_payload(decode=True).decode(errors='ignore').strip() + 
" " 
            elif content_type == "text/html": 
                html_content = part.get_payload(decode=True).decode(errors='ignore').strip() 
                soup = BeautifulSoup(html_content, "html.parser") 
                for a in soup.find_all("a", href=True): 
                    hyperlinks += 1 
                    urls += 1 
                    url = urlparse(a['href']) 
                    if url.scheme: 
                        protocols.add(url.scheme) 
                email_text += soup.get_text(separator=" ", strip=True) + " " 
    else: 
        email_text = msg.get_payload(decode=True).decode(errors='ignore').strip() 
     
    if urls > 12: 
        spam_flag = True 
     
    return { 
        "Headers": headers, 
        "Text": email_text,  
 
        "Number of Hyperlinks": hyperlinks, 
        "Number of URLs": urls, 
        "Attachments": len(attachments), 
        "Protocols": list(protocols), 
        "Spam Flag": spam_flag 
    } 
def load_dataset(csv_file): 
    df = pd.read_csv(csv_file).dropna() 
    le = LabelEncoder() 
    df['label'] = le.fit_transform(df['label'])  # Spam = 1, Ham = 0 
     
    return df, le 
csv_file = "spam8000.csv"  # Update with actual file path 
df, label_encoder = load_dataset(csv_file) 
df.head() 
def extract_features(df): 
    df['Text Length'] = df['email_text'].apply(len) 
    df['Hyperlinks'] = df['email_text'].apply(lambda x: len(re.findall(r'https?://\S+', x))) 
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000) 
    text_features = vectorizer.fit_transform(df['email_text']).toarray() 
    feature_df = pd.DataFrame(text_features) 
    df = df.reset_index(drop=True).join(feature_df) 
    df.columns = df.columns.astype(str) 
    return df.drop(columns=['email_text']), vectorizer 
 
 
def train_model(df, vectorizer): 
    X = df.drop(columns=['label']) 
    y = df['label'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    model = RandomForestClassifier(n_estimators=100, random_state=42) 
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)
      print("\nClassification Report:") 
    print(classification_report(y_test, y_pred, digits=4)) 
     
    y_prob = model.predict_proba(X_test)[:, 1] 
    fpr, tpr, _ = roc_curve(y_test, y_prob) 
    roc_auc = auc(fpr, tpr) 
    plt.figure() 
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f}') 
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title('Receiver Operating Characteristic (ROC) Curve') 
    plt.legend(loc='lower right') 
    plt.show() 
     
    pca = PCA(n_components=2) 
    reduced_X = pca.fit_transform(X) 
    plt.figure(figsize=(8, 6)) 
    plt.scatter(reduced_X[:, 0], reduced_X[:, 1], c=y, cmap='coolwarm', alpha=0.7) 
    plt.xlabel('PCA Component 1') 
    plt.ylabel('PCA Component 2') 
    plt.title('PCA Visualization') 
    plt.show() 
     
    plt.figure(figsize=(20, 10), dpi=150)  # Increased figure size for better spacing 
    plot_tree(model.estimators_[0], filled=True, feature_names=list(X.columns), 
class_names=['Ham', 'Spam'], max_depth=4, fontsize=10, proportion=True, 
rounded=True, impurity=False) 
    plt.show() 
     
    return model, X.columns.astype(str), vectorizer
def main(): 
    dataset_file = "spam8000.csv" 
    email_file = "sample.eml" 
     
    if os.path.exists(dataset_file): 
        df, label_encoder = load_dataset(dataset_file) 
        df, tfidf_vectorizer = extract_features(df) 
        spam_model, feature_names, tfidf_vectorizer = train_model(df, tfidf_vectorizer) 
    else: 
        print("Dataset not found! Train model first.") 
        return 
     
    if os.path.exists(email_file): 
        email_data = extract_email_details(email_file) 
        print("\nEmail Details:") 
        print(f"Headers: {email_data['Headers']}") 
        print(f"Number of Hyperlinks: {email_data['Number of Hyperlinks']}") 
        print(f"Number of URLs: {email_data['Number of URLs']}") 
        print(f"Number of Attachments: {email_data['Attachments']}") 
        print(f"Detected Protocols: {', '.join(email_data['Protocols']) if 
email_data['Protocols'] else 'None'}") 
        print(f"\nEmail Classification: {'Spam' if email_data['Spam Flag'] else 'Not Spam'}") 
    else: 
        print("Email file not found!") 
 
if __name__ == "__main__": 
    main() 
