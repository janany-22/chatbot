import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import spacy
from PyPDF2 import PdfReader
import docx
import nltk
from nltk.tokenize import word_tokenize
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from sklearn.preprocessing import StandardScaler

# Download NLTK data (run once)
nltk.download('punkt')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Store history of models and data
models = {}
data_summaries = {}

# File parsing functions for different file types
def parse_file(file_path):
    print(f"Attempting to parse file: {file_path}")
    if not os.path.exists(file_path):
        print(f"File not found at: {file_path}")
        return None
    
    file_path = os.path.normpath(file_path)
    file_extension = os.path.splitext(file_path)[1].lower()
    print(f"Detected file extension: {file_extension}")
    data = None
    
    try:
        if file_extension == ".csv":
            data = pd.read_csv(file_path, delimiter=',')
        elif file_extension == ".xlsx":
            data = pd.read_excel(file_path)
        elif file_extension == ".pdf":
            pdf_reader = PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            data = pd.DataFrame({"text": [text]})
        elif file_extension == ".docx":
            doc = docx.Document(file_path)
            text = " ".join([para.text for para in doc.paragraphs])
            data = pd.DataFrame({"text": [text]})
        elif file_extension in [".txt", ".text"]:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            data = pd.DataFrame({"text": [text]})
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        print(f"Successfully parsed file into DataFrame with shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error parsing file: {e}")
        return None

# Preprocess data
def preprocess_data(data):
    if data is not None and not data.empty:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mean())
        
        text_cols = data.select_dtypes(include=['object']).columns
        for col in text_cols:
            data[col] = data[col].astype(str).apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
        
        return data
    return None

# Train ML model for numerical data
def train_numerical_model(data, file_id):
    if data is None or data.empty:
        print(f"No data to train model for file {file_id}")
        return None
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print(f"No numerical columns found in file {file_id}. Skipping numerical model training.")
        return None
    
    target = 'Closing Balance' if 'Closing Balance' in numeric_cols else numeric_cols[0]
    feature_cols = [col for col in numeric_cols if col != target and col != 'Chq./Ref.No.']
    if not feature_cols:
        print(f"Insufficient features for training with target {target} in file {file_id}.")
        return None
    
    X = data[feature_cols]
    y = np.log1p(data[target])  # Log transform target to handle large values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LGBMRegressor()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model trained for file {file_id} with MSE: {mse}")
    
    model_path = f"model_{file_id}.joblib"
    joblib.dump(model, model_path)
    
    return model, {"features": feature_cols, "target": target, "mse": mse, "model_path": model_path}

# Train text model using TF-IDF for similarity
def train_text_model(data, file_id):
    if data is None or data.empty or "text" not in data.columns:
        return None
    
    text_data = data["text"].iloc[0]
    text_path = f"text_{file_id}.json"
    with open(text_path, 'w', encoding='utf-8') as f:
        json.dump({"text_content": text_data}, f)
    
    return {"text_content": text_data}, text_path

# Store trained model and data summary
def store_model_and_summary(model, data, file_id, is_text=False):
    if model and data is not None:
        if is_text:
            model_data, model_path = model
        else:
            model_data, metadata = model
            model_path = metadata["model_path"]
        models[file_id] = {"model_path": model_path, "is_text": is_text}
        summary = {
            "columns": data.columns.tolist(),
            "shape": data.shape,
            "summary_stats": data.describe(include='all').to_dict()  # Include all columns
        }
        data_summaries[file_id] = summary
        summary_path = f"summary_{file_id}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f)
        return f"Model and summary stored for file {file_id} at {model_path} and {summary_path}"
    return None

# Load model
def load_model(file_id):
    if file_id in models:
        model_data = models[file_id]
        if model_data["is_text"]:
            with open(model_data["model_path"], 'r', encoding='utf-8') as f:
                text_data = json.load(f)
            return {"text_content": text_data["text_content"]}
        else:
            return joblib.load(model_data["model_path"])
    return None

# Parse intent and query
def parse_intent_and_query(user_input):
    doc = nlp(user_input.lower())
    intent = "general"
    query = user_input
    
    if any(token.text in ["hello", "hi", "hey"] for token in doc):
        intent = "greeting"
    elif "upload" in query.lower() and any(token.text in ["train", "upload", "file"] for token in doc):
        intent = "file_processing"
    elif any(token.text in ["what", "average", "sum", "mean"] for token in doc):
        intent = "question"
    print(f"Detected intent: {intent}, Query: {query}")  # Debug print
    return intent, query

# Fetch answer
def fetch_answer(intent, query):
    print(f"Entering fetch_answer with intent: {intent}, query: {query}")  # Debug print
    if intent == "greeting":
        return "Hello! How can I assist you today?"
    
    elif intent == "question":
        if not models or not data_summaries:
            return "Please upload a file first to train the model."
        
        # Use the most recent file_id if available
        file_id = max(models.keys()) if models else None
        if file_id is None:
            return "No file has been uploaded yet."
        
        model_data = models[file_id]
        summary = data_summaries[file_id]
        model = load_model(file_id)
        print(f"Model data: {model_data}, Summary: {summary}")  # Debug print
        
        if model_data["is_text"]:
            text_content = model["text_content"]
            tfidf = TfidfVectorizer().fit_transform([text_content, query])
            similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            if similarity > 0.3:
                doc = nlp(text_content)
                query_doc = nlp(query)
                for token in query_doc:
                    for ent in doc.ents:
                        if token.text.lower() in ent.text.lower():
                            return f"Based on the file, {token.text} relates to: {ent.text}"
                return f"The file contains information similar to your query (similarity: {similarity:.2f})."
            return "No relevant information found in the text."
        
        else:
            print(f"Numerical model loaded: {model}")  # Debug print
            all_cols = summary["columns"]
            query_tokens = nlp(query.lower())
            print(f"All Columns: {all_cols}")  # Debug print
            
            # Detect aggregate operation and column
            aggregate = None
            column_name = None
            for token in query_tokens:
                if token.text in ["sum", "average", "mean"]:
                    aggregate = token.text
                for col in all_cols:
                    if token.text in col.lower().replace('.', '').replace('/', '').replace(' ', ''):
                        column_name = col
                        break
                if column_name:
                    break
            
            if column_name and column_name in summary["summary_stats"]:
                stats = summary["summary_stats"][column_name]
                if aggregate == "sum":
                    return f"Sum of {column_name}: {stats['count'] * stats['mean']:.2f}"
                elif aggregate in ["average", "mean"]:
                    return f"Average of {column_name}: {stats['mean']:.2f}"
                else:
                    return f"Summary for {column_name}: mean={stats['mean']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}"
            elif column_name and column_name not in summary["summary_stats"]:
                # Handle text or date columns
                sample_value = stats.get("top") if "top" in stats else "N/A"
                unique_count = len(stats.get("unique", [])) if "unique" in stats else 0
                return f"Sample value for {column_name}: {sample_value}, Unique values: {unique_count}"
            else:
                # Default to target prediction if no specific column or aggregate
                target = summary.get("metadata", {}).get("target") if "metadata" in summary else None
                numeric_cols = summary.get("metadata", {}).get("features") if "metadata" in summary else []
                if target and numeric_cols:
                    input_data = [summary["summary_stats"].get(col, {}).get("mean", 0) for col in numeric_cols]
                    if input_data and len(input_data) == len(numeric_cols):
                        input_data = np.array(input_data).reshape(1, -1)
                        scaler = StandardScaler()
                        input_data = scaler.fit_transform(input_data)
                        prediction = model.predict(input_data)
                        prediction = np.expm1(prediction[0])
                        return f"Predicted {target} for your query: {prediction:.2f}"
                return "Please provide a question related to the data columns (e.g., 'what is the closing balance' or 'average of deposit amt')."

# Main chatbot loop
def chatbot():
    file_id = 0
    
    while True:
        user_input = input("You: ")
        
        intent, query = parse_intent_and_query(user_input)
        
        if intent == "file_processing" and "upload" in query.lower():
            file_path = input("Please enter the file path: ")
            data = parse_file(file_path)
            if data is not None:
                data = preprocess_data(data)
                print("Data columns and types:\n", data.dtypes)
                if "text" in data.columns or data.select_dtypes(include=[np.number]).empty:
                    model, model_path = train_text_model(data, file_id)
                    is_text = True
                else:
                    model, metadata = train_numerical_model(data, file_id)
                    is_text = False
                if model:
                    store_model_and_summary((model, model_path) if is_text else (model, metadata), data, file_id, is_text)
                    print(f"File {file_id} processed successfully.")
                    file_id += 1
                else:
                    print("Failed to train model.")
        
        elif intent == "question":
            answer = fetch_answer(intent, query)
            print(f"Bot: {answer}")

if __name__ == "__main__":
    print("Welcome to the Chatbot! Upload a file or ask a question.")
    chatbot()