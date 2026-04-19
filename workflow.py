import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from prefect import task, flow
from utils import clean_text
import joblib

@task
def load_and_preprocess():
    # File load 
    df = pd.read_csv("C:/Users/hp/Downloads/reviews_badminton/data.csv")
    df.columns = df.columns.str.strip() 
    
    #  Column Names 
    text_col = 'Review text' 
    rating_col = 'Ratings'
    
    print(f"Using column '{text_col}' for text and '{rating_col}' for ratings.")

    # Cleaning process
    df['cleaned_review'] = df[text_col].apply(clean_text)
    
    # Sentiment Labeling (Ratings column )
    # 
    df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')
    df = df.dropna(subset=[rating_col]) 
    
    df['sentiment'] = df[rating_col].apply(lambda x: 1 if x >= 4 else 0)
    
    return df

@task
def train_model(df):
    mlflow.set_experiment("Sentiment_Analysis_Flipkart")
    
    with mlflow.start_run(run_name="RandomForest_Baseline"):
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42
        )
        
        tfidf = TfidfVectorizer(max_features=2000)
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        
        n_estimators = 100
        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X_train_tfidf, y_train)
        
        preds = model.predict(X_test_tfidf)
        f1 = f1_score(y_test, preds)
        
        # MLflow Logging
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model", registered_model_name="FlipkartSentimentModel")
        
        # Save locally for Streamlit
        joblib.dump(model, "best_model.pkl")
        joblib.dump(tfidf, "tfidf_vectorizer.pkl")
        
        print(f"Model trained with F1 Score: {f1}")

@flow(name="Sentiment_Analysis_Workflow")
def main_flow():
    data = load_and_preprocess()
    train_model(data)

if __name__ == "__main__":
    main_flow()