from collections import Counter

def filter_and_sort_words(file_path):
    """
    Reads a text file, filters words starting with 'dim', counts their occurrences,
    and sorts them by frequency in descending order.

    :param file_path: Path to the text file.
    :return: A list of tuples (word, count), sorted by count in descending order.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read and split words
            words = file.read().split()


        # Filter words starting with "dim" (case-insensitive)
        filtered_words = [word for word in words if word.lower().startswith('dim') or word.lower().startswith('row')]

        # Count occurrences
        word_counts = Counter(filtered_words)

        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        return sorted_words
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []


words=filter_and_sort_words('features.txt')
print(words)

