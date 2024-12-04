import re
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def luhn_summarization(text, summary_length=3, threshold=0.5):
    """
    Summarizes a given text using Luhn's heuristic method.

    Parameters:
        text (str): The input text to summarize.
        summary_length (int): Number of sentences to include in the summary.
        threshold (float): Frequency threshold for considering significant words.

    Returns:
        str: The summarized text.
    """
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    word_frequencies = Counter(filtered_words)
    max_frequency = max(word_frequencies.values())
    
    for word in word_frequencies:
        word_frequencies[word] /= max_frequency

    significant_words = {word for word, freq in word_frequencies.items() if freq >= threshold}

    sentence_scores = {}
    for sentence in sentences:
        sentence_words = word_tokenize(sentence.lower())
        significant_count = sum(1 for word in sentence_words if word in significant_words)
        sentence_length = len(sentence_words)
        if sentence_length > 0:
            score = significant_count ** 2 / sentence_length
            sentence_scores[sentence] = score

    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:summary_length]
    summary = " ".join(summarized_sentences)

    return summary

if __name__ == "__main__":
    file_path = input("Enter the path to the text file: ")
    summary_length = int(input("Enter the number of sentences for the summary: "))
    
    try:
        with open("blog_post.txt", 'r') as file:
            input_text = file.read()
            summary = luhn_summarization(input_text, summary_length=summary_length)
            print("\nSummary:")
            print(summary)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
