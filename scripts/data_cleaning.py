import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer=WordNetLemmatizer()

nltk.download("stopwords")
stop_words=set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text=text.lower()
    text=re.sub(r"http\S+|www\S+|https\S+","",text,flags=re.MULTILINE)
    text=re.sub(r"@\w+","",text)
    text=re.sub(r"#","",text)
    text=re.sub(r"\d+","",text)

    text=text.translate(str.maketrans("","",string.punctuation))

    tokens=word_tokenize(text)

    cleaned_tokens=[
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    text=" ".join(cleaned_tokens)
    
    return text.strip()




