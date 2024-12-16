#combines csv files together
# import pandas as pd
#
# # List of CSV file paths
# csv_files = ["file1.csv", "file2.csv", "file3.csv", "file4.csv"]
#
# # Read and combine all CSV files
# combined_df = pd.concat([pd.read_csv(file) for file in csv_files])
#
# # Reset the index (optional, to make it consistent)
# combined_df.reset_index(drop=True, inplace=True)
#
# # Save the combined data to a new CSV file
# combined_df.to_csv("combined_file.csv", index=False)
#
# print("Files combined successfully into 'combined_file.csv'")

#gets a list of all the file names
import os

# Specify the root directory containing the subfolders
root_dir = "C:\\Users\\rache\\Desktop\\איחזור מידע\\hw2\\IR-Newspapers-files"

# List to store file names and their relative paths
csv_files = []

# Walk through the directory and its subdirectories
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".csv"):  # Check for CSV files
            # Construct relative path
            relative_path = os.path.relpath(os.path.join(subdir, file), start=root_dir)
            csv_files.append({"name": file, "relative_path": relative_path})

# Print all CSV files and their relative paths
for csv_file in csv_files:
    print(f"Name: {csv_file['name']}, Path: {csv_file['relative_path']}")

