import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist

# Preprocessing function
def preprocess_text(text):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = [word.lower() for sentence in sentences for word in word_tokenize(sentence)]
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.casefold() not in stop_words]
    
    # Stem the words
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return words

# TF-IDF calculation function
def calculate_tf_idf(text):
    # Calculate term frequency (TF)
    word_freq = FreqDist(text)
    tf_scores = {word: word_freq[word] / len(text) for word in word_freq.keys()}
    
    # Calculate inverse document frequency (IDF)
    sentences = sent_tokenize(' '.join(text))
    document_freq = {}
    for word in word_freq.keys():
        document_freq[word] = sum(1 for sent in sentences if word in word_tokenize(sent))
    
    # Calculate TF-IDF scores
    tf_idf_scores = {word: tf_scores[word] * document_freq[word] for word in word_freq.keys()}
    
    return tf_idf_scores

# Summarization function
def summarize_text(text, num_sentences=3):
    # Preprocess the text
    words = preprocess_text(text)
    
    # Calculate TF-IDF scores
    tf_idf_scores = calculate_tf_idf(words)
    
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Calculate sentence scores based on TF-IDF scores
    sentence_scores = {}
    for sentence in sentences:
        sentence_words = word_tokenize(sentence.lower())
        sentence_score = sum(tf_idf_scores[word] for word in sentence_words if word in tf_idf_scores.keys())
        sentence_scores[sentence] = sentence_score
    
    # Sort the sentences based on scores and select the top sentences
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    top_sentences = [sentence for (sentence, score) in sorted_sentences[:num_sentences]]
    
    # Join the top sentences to form the summary
    summary = " ".join(top_sentences)
    
    return summary

# Get user input
input_text = input("Enter the text you want to summarize:\n")

# Summarize the text
summary = summarize_text(input_text)

# Print the summary
print("Summary:")
print(summary)
