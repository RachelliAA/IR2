import pandas as pd
import re
import nltk
from transformers import BertTokenizer, BertModel
import torch
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# Function to clean text
def clean_text(text):
    # Remove non-English words and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Remove stop words
    words = text.split()
    cleaned_words = [word.lower() for word in words if word.lower() not in stop_words]
    return ' '.join(cleaned_words)


# Function to split text into chunks of 512 tokens
def split_into_chunks(text, max_length=512):
    tokens = tokenizer.tokenize(text)
    for i in range(0, len(tokens), max_length):
        yield ' '.join(tokens[i:i + max_length])


# Function to process text with BERT
def process_with_bert(text):
    chunks = list(split_into_chunks(text))
    embeddings = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Get the [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
        embeddings.append(cls_embedding)

    # Aggregate embeddings (e.g., take the mean)
    aggregated_embedding = torch.mean(torch.stack(embeddings), dim=0)
    return aggregated_embedding


# Read Excel file
file_path = "posts_first_targil.xlsx"
excel_data = pd.ExcelFile(file_path)

# Process each sheet
results = {}
for sheet_name in excel_data.sheet_names:
    df = excel_data.parse(sheet_name)

    # Ensure required columns exist
    if "Body Text" in df.columns and "title" in df.columns:
        # Combine and clean text
        df['combined_text'] = (df['title'].fillna('') + " " + df['Body Text'].fillna('')).apply(clean_text)

        # Apply BERT
        df['bert_embedding'] = df['combined_text'].apply(process_with_bert)
        results[sheet_name] = df

# Save results to a new Excel file
with pd.ExcelWriter("processed_results.xlsx") as writer:
    for sheet_name, result_df in results.items():
        result_df.to_excel(writer, sheet_name=sheet_name, index=False)
