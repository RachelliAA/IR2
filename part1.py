merging
# import pandas as pd
#
# # Upload your Excel file
# file_path = 'tfidf_structured_matrix1.xlsx'  # Replace with the path to your Excel file
#
# # Read all sheets into a dictionary
# sheets = pd.read_excel(file_path, sheet_name=None)
#
# # Get a set of all unique column names from all tabs
# all_columns = set()
# for sheet in sheets.values():
#     all_columns.update(sheet.columns)
#
# # Combine all sheets, filling missing columns with zeros
# combined_df = pd.DataFrame(columns=sorted(all_columns))  # Create a DataFrame with all possible columns
# for sheet_name, df in sheets.items():
#     # Align columns with the combined DataFrame, fill missing with zeros
#     df = df.reindex(columns=combined_df.columns, fill_value=0)
#     combined_df = pd.concat([combined_df, df], ignore_index=True)
#
# # Save the combined sheet to a new Excel file
# output_path = 'combined_file_with_all_columns.xlsx'  # Replace with the desired output path
# combined_df.to_excel(output_path, index=False)
#
# print(f"Combined sheet with all columns saved to {output_path}")
################## combined the tabs to 1 tab #####################################
# import pandas as pd
# from sklearn.cluster import MiniBatchKMeans
# from sklearn.preprocessing import StandardScaler
# from scipy.sparse import csr_matrix
#
# # Load the combined file
# file_path = 'combined_file_with_all_columns.xlsx'  # Replace with your combined file path
# data = pd.read_excel(file_path)
#
# # Preprocess the data: Ensure numerical values and handle missing data
# data = data.fillna(0)  # Replace NaN with 0 or use other imputation methods
# data_numeric = data.select_dtypes(include=['number'])  # Select only numerical columns
#
# # Convert to sparse matrix (if necessary, for large datasets)
# data_sparse = csr_matrix(data_numeric.values)
#
# # Apply MiniBatchKMeans clustering (this can handle sparse matrices)
# kmeans = MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=100)
# data['Cluster'] = kmeans.fit_predict(data_sparse)
#
# # Save the clustered data to a new file
# output_path = 'clustered_data_sparse.xlsx'  # Replace with the desired output path
# data.to_excel(output_path, index=False)
#
# print(f"Clustered data saved to {output_path}")
############# kmeans clustering worked ##################


#doesnt work
#
# import pandas as pd
# from rank_bm25 import BM25Okapi
# import string
#
#
# # Function to preprocess text
# def preprocess(text):
#     return text.lower().translate(str.maketrans('', '', string.punctuation)).split()
#
#
# # Load the Excel file
# file_path = 'posts_first_targil.xlsx'
# sheets = pd.read_excel(file_path, sheet_name=None)
#
# # Dictionary to store results
# results = {}
#
# # Iterate over each sheet
# for sheet_name, df in sheets.items():
#     combined_text = df['title'].astype(str) + ' ' + df['Body Text'].astype(str)
#     documents = combined_text.tolist()
#     tokenized_docs = [preprocess(doc) for doc in documents]
#     bm25 = BM25Okapi(tokenized_docs)
#
#     # Create a list of terms from the BM25 index
#     terms = bm25.idf.keys()
#     columns = ['Document'] + list(terms)
#     scores_df = pd.DataFrame(columns=columns)
#
#     print(f"Sheet: {sheet_name}")
#     print(f"Number of documents: {len(documents)}")
#     print(f"Number of terms: {len(terms)}")
#
#     for i, doc in enumerate(tokenized_docs):
#         scores = bm25.get_scores(doc)
#         row_data = [documents[i]] + list(scores)
#         if len(row_data) == len(columns):  # Ensure correct number of columns
#             scores_df.loc[i] = row_data
#         else:
#             print(
#                 f"Skipping row {i} in sheet {sheet_name} due to mismatch in columns. Expected {len(columns)}, got {len(row_data)}.")
#
#     results[sheet_name] = scores_df
#
# # Save the results to a new Excel file
# with pd.ExcelWriter('results.xlsx') as writer:
#     for sheet_name, result_df in results.items():
#         result_df.to_excel(writer, sheet_name=sheet_name, index=False)
#
# print("TF-IDF with BM25Okapi results saved to results.xlsx")
#
# import pandas as pd
# import re
# import nltk
# from transformers import BertTokenizer, BertModel
# import torch
# from nltk.corpus import stopwords
#
# # Download stopwords if not already downloaded
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))
#
# # Load BERT tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
#
#
# # Function to clean text
# def clean_text(text):
#     # Remove non-English words and numbers
#     text = re.sub(r'[^A-Za-z\s]', '', text)
#     # Remove stop words
#     words = text.split()
#     cleaned_words = [word.lower() for word in words if word.lower() not in stop_words]
#     return ' '.join(cleaned_words)
#
#
# # Function to split text into chunks of 512 tokens
# def split_into_chunks(text, max_length=512):
#     tokens = tokenizer.tokenize(text)
#     for i in range(0, len(tokens), max_length):
#         yield ' '.join(tokens[i:i + max_length])
#
#
# # Function to process text with BERT
# def process_with_bert(text):
#     chunks = list(split_into_chunks(text))
#     embeddings = []
#
#     for chunk in chunks:
#         inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         # Get the [CLS] token embedding (first token)
#         cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
#         embeddings.append(cls_embedding)
#
#     # Aggregate embeddings (e.g., take the mean)
#     aggregated_embedding = torch.mean(torch.stack(embeddings), dim=0)
#     return aggregated_embedding
#
#
# # Read Excel file
# file_path = "posts_first_targil.xlsx"
# excel_data = pd.ExcelFile(file_path)
#
# # Process each sheet
# results = {}
# for sheet_name in excel_data.sheet_names:
#     df = excel_data.parse(sheet_name)
#
#     # Ensure required columns exist
#     if "Body Text" in df.columns and "Title" in df.columns:
#         # Combine and clean text
#         df['combined_text'] = (df['Title'].fillna('') + " " + df['Body Text'].fillna('')).apply(clean_text)
#
#         # Apply BERT
#         df['bert_embedding'] = df['combined_text'].apply(process_with_bert)
#         results[sheet_name] = df
#
# # Save results to a new Excel file
# with pd.ExcelWriter("processed_results.xlsx") as writer:
#     for sheet_name, result_df in results.items():
#         result_df.to_excel(writer, sheet_name=sheet_name, index=False)
#


import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.sparse import csr_matrix
import umap
import matplotlib.pyplot as plt

# Load the combined file
file_path = 'combined_file_with_all_columns.xlsx'  # Replace with your combined file path
data = pd.read_excel(file_path)

# Assuming 'TrueLabel' column exists in your dataset for evaluation
true_labels = data['TrueLabel'] if 'TrueLabel' in data.columns else None

# Preprocess the data: Ensure numerical values and handle missing data
data = data.fillna(0)  # Replace NaN with 0 or use other imputation methods
data_numeric = data.select_dtypes(include=['number'])  # Select only numerical columns

# Convert to sparse matrix (if necessary, for large datasets)
data_sparse = csr_matrix(data_numeric.values)

# Apply MiniBatchKMeans clustering (this can handle sparse matrices)
kmeans = MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=100)
data['Cluster'] = kmeans.fit_predict(data_sparse)

# If true labels are available, evaluate performance
if true_labels is not None:
    precision = precision_score(true_labels, data['Cluster'], average='macro')
    recall = recall_score(true_labels, data['Cluster'], average='macro')
    f1 = f1_score(true_labels, data['Cluster'], average='macro')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
else:
    print("True labels not found. Skipping evaluation.")

# Dimensionality reduction using UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
reduced_data = reducer.fit_transform(data_numeric.values)

# Create a scatter plot of the clusters
plt.figure(figsize=(10, 7))
for cluster in range(4):  # Adjust for your number of clusters
    cluster_data = reduced_data[data['Cluster'] == cluster]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}', alpha=0.7)

plt.title('Clusters Visualization using UMAP')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.legend()
plt.grid(True)
plt.show()

# Save the clustered data to a new file
output_path = 'clustered_data_umap.xlsx'  # Replace with the desired output path
data.to_excel(output_path, index=False)

print(f"Clustered data saved to {output_path}")