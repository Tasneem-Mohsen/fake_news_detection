import os
import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Helpers
def remove_punct(text):
    return "".join([char for char in text if char not in string.punctuation])

def count_punct_words(text):
    words = text.split()
    if len(words) == 0:
        return 0
    punct_count = sum(1 for char in text if char in string.punctuation)
    return round(punct_count / len(words), 3) * 100

def count_cap_words(text):
    words = text.split()
    if len(words) == 0:
        return 0
    cap_count = sum(1 for char in text if char.isupper())
    return round(cap_count / len(words), 3) * 100

# Stopwords شائعة موسعة
basic_stopwords = set([
    "the", "is", "in", "and", "to", "a", "of", "that", "it", "for", "on", "with", "as", "at", "this", 
    "be", "by", "are", "was", "from", "an", "or", "but", "if", "not", "we", "you", "they", "he", "she", 
    "them", "his", "her", "its", "our", "their", "have", "has", "had", "do", "does", "did", "will", 
    "would", "can", "could", "should", "may", "might", "been", "being", "up", "down", "out", "over", 
    "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", 
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "only", "own", 
    "same", "so", "than", "too", "very", "just", "now", "into", "about", "between", "after", "before",
    "because", "while", "during", "what", "which", "who", "whom", "these", "those", "i", "me", "my", 
    "myself", "your", "yours", "yourself", "yourselves", "him", "himself", "hers", "herself", "itself", 
    "ours", "ourselves", "themselves", "am", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", 
    "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn"
])

def basic_tokenizer(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    return [word for word in tokens if word not in basic_stopwords]

def main():
   
    print(">>> Script started...") 

    print("Loading data...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "../data/merged_news.csv")
    df = pd.read_csv(csv_path)

    

    print(">>> Script finished successfully.") 

    
    df['text_nopunc'] = df['text'].apply(lambda x: remove_punct(str(x).lower()))
    df['title_nopunc'] = df['title'].apply(lambda x: remove_punct(str(x).lower()))

    df['text_tokens'] = df['text_nopunc'].apply(basic_tokenizer)
    df['title_tokens'] = df['title_nopunc'].apply(basic_tokenizer)

    df['text_lemmatized'] = df['text_tokens'].apply(lambda x: " ".join(x))
    df['title_lemmatized'] = df['title_tokens'].apply(lambda x: " ".join(x))

    # Vectorization
    tfidf = TfidfVectorizer(ngram_range=(1, 1), max_features=5000)
    tfidftitle = TfidfVectorizer(ngram_range=(1, 1), max_features=5000)

    text_tfidf = tfidf.fit_transform(df['text_lemmatized'])
    title_tfidf = tfidftitle.fit_transform(df['title_lemmatized'])

    # Feature Engineering
    df['body_len'] = df['text'].apply(lambda x: len(str(x)) - str(x).count(' '))
    df['title_len'] = df['title'].apply(lambda x: len(str(x)) - str(x).count(' '))
    df['punct_per_word%'] = df['text'].apply(count_punct_words)
    df['punct_per_word%title'] = df['title'].apply(count_punct_words)
    df['cap_per_word%'] = df['text'].apply(count_cap_words)
    df['cap_per_word%title'] = df['title'].apply(count_cap_words)

    num_vars = ['body_len', 'title_len', 'punct_per_word%', 'punct_per_word%title', 'cap_per_word%', 'cap_per_word%title']
    scaler = MinMaxScaler()
    df[num_vars] = scaler.fit_transform(df[num_vars])

    text_tfidf_df = pd.DataFrame(text_tfidf.toarray(), columns=tfidf.get_feature_names_out())
    title_tfidf_df = pd.DataFrame(title_tfidf.toarray(), columns=tfidftitle.get_feature_names_out())

    X = pd.concat([df[num_vars], text_tfidf_df, title_tfidf_df], axis=1)
    y = df['label']

    # Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    # Evaluation
    y_pred = rf_model.predict(X_test)
    print("Model Score:", rf_model.score(X_test, y_test))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print("Done!")

if __name__ == "__main__":
    main()