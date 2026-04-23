import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

text = input("Enter your text: ")

tokens = word_tokenize(text)
print("\nTokens:", tokens)
print("\nWord Count:", len(tokens))

stemmer = PorterStemmer()
stems = [stemmer.stem(word) for word in tokens]
print("\nStemming:", stems)

lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(word) for word in tokens]
print("\nLemmatization:", lemmas)

blob = TextBlob(text)
print("\nSentiment Polarity:", blob.sentiment.polarity)

if blob.sentiment.polarity > 0:
    print("Positive Text")
elif blob.sentiment.polarity < 0:
    print("Negative Text")
else:
    print("Neutral Text")
