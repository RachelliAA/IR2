
import copy
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from textblob import Word

import pandas as pd
import scipy.sparse
import re


# DOSNT TAKE  CARE OF  'HELLO' ''''
import re

import re


def clean_text(text):
    if isinstance(text, str):  # Ensure the input is a string
        # Add spaces around special characters (like ?!, but not within words)
        cleaned_text = re.sub(r'(?<!\w)([.,;:!?(){}[\]<>/"\-“”])|([.,;:!?(){}[\]<>/"\-“”])(?!\w)', r' \1\2 ', text)

        # Add spaces around ‘ and ’ but only if they're not inside a word
        cleaned_text = re.sub(r"(?<!\w)[‘’]|[‘’](?!\w)", r" \g<0> ", cleaned_text)

        # Remove extra spaces that may have been created
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        return cleaned_text
    else:
        return text  # Return the original text if it's not a string


# Function to preprocess, remove punctuation and numbers, and lemmatize text using spaCy
def preprocess_and_lemmatize(text):
    if isinstance(text, str):
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Process the text with spaCy
        doc = nlp(text)
        # Lemmatize and remove stopwords, ensure only alphabetic tokens are kept
        lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
        return lemmatized_text
    return text


# Function to preprocess, remove punctuation and numbers, and lemmatize text using spaCy
def preprocess_and_lemmatize(text):
    if isinstance(text, str):
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Process the text with spaCy
        doc = nlp(text)
        # Lemmatize and remove stopwords, ensure only alphabetic tokens are kept
        lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
        return lemmatized_text


    return ""

# --------------------------------MAIN--------------------------------------------------------------

# Replace 'your_file.xlsx' with the path to your Excel file
file_path = 'posts_first_targil.xlsx'

# Load the Excel file
excel_data = pd.ExcelFile(file_path)

# List of sheet names
sheet_names = excel_data.sheet_names  # This will give you the names of the sheets

# Dictionary to store each sheet's data as a dataframe
sheets_data = {}

# Read each sheet into a separate dataframe
for sheet in sheet_names:
    sheets_data[sheet] = pd.read_excel(file_path, sheet_name=sheet)

# Now sheets_data contains a dataframe for each sheet
# Access each dataframe by sheet name, for example:
sheet1_data = sheets_data[sheet_names[0]]  # First sheet
sheet2_data = sheets_data[sheet_names[1]]  # Second sheet, etc.

# Display the first few rows of the first sheet as a sample
print(sheet1_data.head())

cleaned_sheets_data = {}

# Read each sheet into a separate dataframe and clean the "Body Text", 'title' column
for sheet in sheet_names:
    df = sheets_data[sheet]

    # Apply clean_text function to the 'Body Text' and 'title' columns
    if 'Body Text' in df.columns:
        df['Body Text'] = df['Body Text'].apply(clean_text)
    if 'title' in df.columns:
        df['title'] = df['title'].apply(clean_text)

    # Store the cleaned dataframe
    cleaned_sheets_data[sheet] = df

# To verify, print the first few rows of the 'Body Text' column from the first sheet
# print(cleaned_sheets_data[sheet_names[0]]['Body Text'])
print(cleaned_sheets_data[sheet_names[0]].head())

# saves the cleaned (made spaces) text to a file
# Define the path where you want to save the cleaned Excel file
output_file_path = 'cleaned_posts_first_targil1.xlsx'

# Create a new Excel writer object
with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
    # Loop through the cleaned_sheets_data dictionary and save each sheet to the Excel file
    for sheet_name, df in cleaned_sheets_data.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)


print(f"Cleaned data has been saved to {output_file_path}")
"""
########now we have clean from punctuation#########


# Deep copy of dictionary
lemma_sheets_data = copy.deepcopy(cleaned_sheets_data)

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Define the path to the saved Excel file
input_file_path = 'cleaned_posts_first_targil1.xlsx'

# Read the Excel file back into a dictionary of DataFrames
cleaned_sheets_data = pd.read_excel(input_file_path, sheet_name=None)


print(f"Data has been loaded from {input_file_path}")

# Process each sheet
for sheet_name, df in cleaned_sheets_data.items():
    if 'Body Text' in df.columns and 'title' in df.columns:
        # Replace NaN with an empty string before processing
        df['Body Text'] = df['Body Text'].fillna("")
        df['title'] = df['title'].fillna("")

        # Perform lemmatization on each entry in the 'Body Text' and 'title' columns
        df['Body Text'] = df['Body Text'].apply(preprocess_and_lemmatize)
        df['title'] = df['title'].apply(preprocess_and_lemmatize)

        # Combine the two columns
        df['Combined Text'] = df['title'] + " " + df['Body Text']

        # Save the cleaned DataFrame
        cleaned_file_path = 'lemma_posts_first_targil1.xlsx'
        with pd.ExcelWriter(cleaned_file_path, engine='openpyxl') as writer:
            for sheet_name, df in cleaned_sheets_data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Cleaned data has been saved to {cleaned_file_path}")

#####now we have a file with lemmas

#########start of tfidf########################
# does tfidf on the lemma and saves to file
input_file_path = 'lemma_posts_first_targil1.xlsx'

# Read the Excel file back into a dictionary of DataFrames
cleaned_sheets_data = pd.read_excel(input_file_path, sheet_name=None)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', min_df=5)

# Prepare TF-IDF results per sheet
tfidf_results = {}

for sheet_name, df in cleaned_sheets_data.items():
    if 'Combined Text' in df.columns:
        combined_texts = df['Combined Text'].tolist()

        # Compute TF-IDF
        tfidf_matrix = vectorizer.fit_transform(combined_texts)

        # Convert to dense array
        dense_tfidf_matrix = tfidf_matrix.toarray()

        # Convert to DataFrame
        tfidf_df = pd.DataFrame(dense_tfidf_matrix, columns=vectorizer.get_feature_names_out())

        # Save this TF-IDF DataFrame to the results dictionary
        tfidf_results[sheet_name] = tfidf_df
    else:
        print(f"'Combined Text' column not found in sheet: {sheet_name}")

# Save the TF-IDF matrices to a multi-sheet Excel file
output_excel_path = 'tfidf_structured_matrix1.xlsx'
with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
    for sheet_name, tfidf_df in tfidf_results.items():
        tfidf_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"TF-IDF matrices have been saved to {output_excel_path}")



###########TFIDF on the cleaned data ############################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Define the path to the cleaned Excel file
input_file_path = 'cleaned_posts_first_targil1.xlsx'

# Read the Excel file into a dictionary of DataFrames
cleaned_sheets_data = pd.read_excel(input_file_path, sheet_name=None)

print(f"Data has been loaded from {input_file_path}")

# Iterate through each sheet and print the first few rows of each
for sheet_name, df in cleaned_sheets_data.items():
    print(f"First few rows of sheet: {sheet_name}")
    print(df.head())  # Show the first few rows of the current sheet
    print("\n" + "-"*50 + "\n")  # Separator between sheets


# Initialize the TF-IDF Vectorizer (no need for lemmatization)
vectorizer = TfidfVectorizer(stop_words='english', min_df=5)

# Prepare TF-IDF results per sheet
tfidf_results = {}

# Process each sheet and apply TF-IDF
for sheet_name, df in cleaned_sheets_data.items():
    print(sheet_name)
    if 'Body Text' in df.columns and 'title' in df.columns:
        # Replace NaN with an empty string before processing
        df['Body Text'] = df['Body Text'].fillna("")
        df['title'] = df['title'].fillna("")

        # Combine the 'Body Text' and 'title' columns
        df['Combined Text'] = df['title'] + " " + df['Body Text']

        # Compute TF-IDF for the combined text
        combined_texts = df['Combined Text'].tolist()
        tfidf_matrix = vectorizer.fit_transform(combined_texts)

        # Convert the TF-IDF matrix to a dense array
        dense_tfidf_matrix = tfidf_matrix.toarray()

        # Convert the dense matrix to a DataFrame
        tfidf_df = pd.DataFrame(dense_tfidf_matrix, columns=vectorizer.get_feature_names_out())

        # Save the TF-IDF DataFrame to the results dictionary
        tfidf_results[sheet_name] = tfidf_df
    else:
        print(f"'Body Text' or 'title' column not found in sheet: {sheet_name}")

# Save the TF-IDF matrices to a multi-sheet Excel file
output_excel_path = 'tfidf_matrix_no_lemma.xlsx'
with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
    for sheet_name, tfidf_df in tfidf_results.items():
        tfidf_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"TF-IDF matrices have been saved to {output_excel_path}")



########################### finished part A ###################################

#################adding lables#########################33
import pandas as pd


def add_labels_to_excel(input_file_path, output_file_path):
    # Adds a 'Label' column to each sheet in the input Excel file and saves the result to the output Excel file.
    #
    # Args:
    # input_file_path (str): The path to the input Excel file.
    # output_file_path (str): The path to the output Excel file where the updated file will be saved.

    # Read the Excel file with multiple sheets
    excel_data = pd.read_excel(input_file_path, sheet_name=None)

    # Define a function to assign labels (1 or 0) based on a simple rule
    def assign_label(row):
        # Convert all values to numeric, errors will be set as 0
        row = pd.to_numeric(row, errors='coerce').fillna(0)

        # Example condition: if the sum of TF-IDF values in the row is greater than a threshold, label it as 1, otherwise 0
        if row.sum() > 1.9:  # You can adjust this threshold as needed
            return 1
        else:
            return 0

    # Iterate over each sheet in the Excel data and apply the label function
    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
        for sheet_name, df in excel_data.items():
            # Apply the label function to each row and create a new 'Label' column
            df['Label'] = df.apply(assign_label, axis=1)

            # Ensure at least one sheet has visible data
            if not df.empty:
                # Write the updated DataFrame to the new Excel file
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Labels have been added and saved to {output_file_path}")


# Example of how to call the function
input_file = 'tfidf_matrix_no_lemma.xlsx'
output_file = 'cleaned_file_with_labels.xlsx'

add_labels_to_excel(input_file, output_file)

# Example of how to call the function
input_file = 'tfidf_structured_matrix1.xlsx'
output_file = 'lemma_file_with_labels.xlsx'

add_labels_to_excel(input_file, output_file)


####################################################################
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif


def calculate_metrics(input_file_path, output_file_path):

    # Calculates Information Gain and Chi-squared Statistic for each feature in the input Excel file
    # with respect to the 'Label' column in each sheet.
    #
    # Args:
    # input_file_path (str): The path to the input Excel file.
    # output_file_path (str): The path to the output Excel file where the results will be saved.

    # Read the Excel file with multiple sheets
    excel_data = pd.read_excel(input_file_path, sheet_name=None)

    # Dictionary to store results for each sheet
    results = {}

    for sheet_name, df in excel_data.items():
        # Ensure the dataframe is not empty and has a 'Label' column
        if 'Label' in df.columns:
            # Separate features (X) and labels (y)
            X = df.drop(columns=['Label'])
            y = df['Label']

            # Calculate Information Gain (mutual information)
            ig = mutual_info_classif(X, y, discrete_features=False)

            # Calculate Chi-squared Statistic
            chi2_stat, p_values = chi2(X, y)

            # Store the results
            results[sheet_name] = pd.DataFrame({
                'Feature': X.columns,
                'Information Gain': ig,
                'Chi-squared Statistic': chi2_stat,
                'P-value': p_values
            })

    # Write results to a new Excel file with one sheet per input sheet
    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
        for sheet_name, result_df in results.items():
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Metrics calculated and saved to {output_file_path}")


# # Example of how to call the function
# input_file = 'cleaned_file_with_labels.xlsx'  # Replace with your input file name
# output_file = 'claened_metrics_results.xlsx'  # Replace with your output file name
#
# calculate_metrics(input_file, output_file)


# Example of how to call the function
input_file = 'lemma_file_with_labels.xlsx'  # Replace with your input file name
output_file = 'lemma_metrics_results.xlsx'  # Replace with your output file name

calculate_metrics(input_file, output_file)

"""

