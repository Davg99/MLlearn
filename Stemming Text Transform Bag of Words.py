#Text Transform

#Input Text > normalize syntax in Preprocess, use stemmer and stop words to apply a cleaner lexicon > vectorize the cleaned sentence(s) into bag of words  

#Natural Language Toolkit (NLTK) works with human language data 
#applying in statistical natural language processing (NLP). 
#text processing libraries for tokenization, parsing, classification, stemming, tagging and semantic reasoning.
#-q = quiet
!pip -q install nltk

# Prepare cleaning functions
# re provides regular expression support
import re, string
import nltk

#SnowballStemmer is a Python language library. 
from nltk.stem import SnowballStemmer

# input your text
text = [‘ ’]

print(text)

# define your stop words
#stop words eliminate unimportant words, allowing focus on important words.
stop_words = ["a", "an", "the", "this", "that", "is", "it", "to", "and", "in", "of"]

# define your stemmer
#Stemming reduces a word to its base or root form to improve the accuracy of NLP models.
stemmer = SnowballStemmer('english')

# define your preprocessing tooling
def preProcessText(text):
    # lowercase and strip leading/trailing white space
    text = text.lower().strip()
    
    # remove HTML tags
    text = re.compile('<.*?>').sub('', text)
    
    # remove punctuation
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    
    # remove extra white space
    text = re.sub('\s+', ' ', text)
    
    return text
        
#define your lexicon function: information (semantic, grammatical) about individual words or word strings.
def lexiconProcess(text, stop_words, stemmer):
    filtered_sentence = []
    words = text.split(" ")
    for w in words:
        if w not in stop_words:
            filtered_sentence.append(stemmer.stem(w))
    text = " ".join(filtered_sentence)
    
    return text

# define cleaned sentence function
def cleanSentence(text, stop_words, stemmer):
    return lexiconProcess(preProcessText(text), stop_words, stemmer)

# Prepare vectorizer 
from sklearn.feature_extraction.text import CountVectorizer
# define the vectorizer function
textvectorizer = CountVectorizer(binary=True, max_features = 200 ) 
# can also limit vocabulary size here, with max_features

print(len(text))

# Clean up the text
text_cleaned = [cleanSentence(item, stop_words, stemmer) for item in text]

print(len(text_cleaned))
print(text_cleaned)

# Vectorize the cleaned text
text_vectorized = textvectorizer.fit_transform(text_cleaned)

print('Vocabulary: \n', textvectorizer.vocabulary_)
print(len(textvectorizer.vocabulary_))
print('Bag of Words Binary Features: \n', text_vectorized.toarray())

print(text_vectorized.shape)
