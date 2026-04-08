import nltk
import string
from nltk.corpus import stopwords

nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    words = text.split()

    words = [w for w in words if w not in stopwords.words('english')]
    words = [w for w in words if w not in string.punctuation]

    return " ".join(words)