

import os
import re
import string
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    print("Downloading NLTK data...")
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")


class EmailPreprocessor:
    

    def __init__(self):
        
        self.stop_words = set(stopwords.words("english"))
        self.vocabulary = {}  # word -> index mapping
        self.vocab_size = 0

    def clean_text(self, text):
        
        
        text = text.lower()

        
        text = re.sub(r"subject:\s*", "", text)
        text = re.sub(r"from:\s*.*?\n", "", text)
        text = re.sub(r"to:\s*.*?\n", "", text)
        text = re.sub(r"date:\s*.*?\n", "", text)
        text = re.sub(r"forwarded by.*?-{10,}", "", text, flags=re.DOTALL)
        text = re.sub(r"-{5,}.*?-{5,}", "", text, flags=re.DOTALL)

        
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

        
        text = re.sub(r"\S+@\S+\.\S+", "", text)

        
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^\w\s]", " ", text)

        
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def tokenize_text(self, text):
        
        tokens = word_tokenize(text)

        
        tokens = [
            token
            for token in tokens
            if token not in self.stop_words
            and len(token) > 2  
            and token.isalpha()
        ]  

        return tokens

    def preprocess_email(self, email_text):
       
        cleaned_text = self.clean_text(email_text)
        tokens = self.tokenize_text(cleaned_text)
        return tokens

    def build_vocabulary(self, dataset_paths, min_frequency=2):
        
        word_counts = Counter()
        total_emails = 0

        for category in ["ham", "spam"]:
            folder_path = dataset_paths[f"train_{category}"]
            if not os.path.exists(folder_path):
                print(f"Warning: Path not found - {folder_path}")
                continue

            
            txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
            print(f"Processing {len(txt_files)} {category} emails...")

            for filename in txt_files:
                filepath = os.path.join(folder_path, filename)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
                        email_content = file.read()
                        tokens = self.preprocess_email(email_content)
                        word_counts.update(tokens)
                        total_emails += 1
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

        
        filtered_words = [
            word for word, count in word_counts.items() if count >= min_frequency
        ]

        
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(filtered_words))}
        self.vocab_size = len(self.vocabulary)

        print(f"Vocabulary built:")
        print(f"  - Total emails processed: {total_emails}")
        print(f"  - Total unique words: {len(word_counts)}")
        print(f"  - Words with freq >= {min_frequency}: {self.vocab_size}")
        print(f"  - Sample words: {list(self.vocabulary.keys())[:10]}")

        return self.vocabulary


def test_preprocessor():
    
    print("Testing Email Preprocessor...")

    # Sample email text
    sample_email = """
    Subject: Get Rich Quick!!!
    From: spammer@fake.com
    To: victim@email.com
    
    Dear Sir/Madam,
    
    You have WON $1,000,000!!! Click here: http://fake-site.com
    Send your bank details to claim@fake.com
    
    Best regards,
    The Lottery Team
    """

    preprocessor = EmailPreprocessor()

    print("\nOriginal text:")
    print(sample_email)

    cleaned = preprocessor.clean_text(sample_email)
    print(f"\nCleaned text:\n{cleaned}")

    tokens = preprocessor.tokenize_text(cleaned)
    print(f"\nTokens: {tokens}")

    full_tokens = preprocessor.preprocess_email(sample_email)
    print(f"\nFull preprocessing result: {full_tokens}")


if __name__ == "__main__":
    test_preprocessor()
