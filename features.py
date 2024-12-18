# def delete_first_word_in_rows(input_file, output_file):
#     """
#     Deletes the first word in every row of a text file and writes the modified content to a new file.
#
#     Parameters:
#     input_file (str): Path to the input text file.
#     output_file (str): Path to the output text file to save the modified content.
#     """
#     try:
#         with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
#             for line in infile:
#                 # Split the line into words
#                 words = line.split()
#                 # Skip the first word and join the rest
#                 new_line = ' '.join(words[1:]) + '\n' if words else '\n'
#                 # Write the modified line to the output file
#                 outfile.write(new_line)
#         print(f"Updated content written to {output_file}")
#     except Exception as e:
#         print(f"An error occurred: {e}")
#
# # Example usage
# delete_first_word_in_rows('features.txt', 'output.txt')

import pandas as pd

# Define the file paths
text_file_path = "output.txt"  # Replace with your text file path
excel_file_path = "features.xlsx"  # Desired Excel file output path

# Read the text file and process it
data = []
with open(text_file_path, 'r') as file:
    for line in file:
        # Split the line into two words
        words = line.strip().split(maxsplit=1)
        if len(words) == 2:  # Ensure there are exactly two words
            data.append(words)

# Create a DataFrame
df = pd.DataFrame(data, columns=["First Word", "Second Word"])

# Write to Excel
df.to_excel(excel_file_path, index=False)

print(f"Data has been written to {excel_file_path}")

