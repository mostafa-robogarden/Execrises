#import nltk
#nltk.download()
import csv
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
from collections import Counter
import os

MAX_LINES = 100
DIR = 'dataset'
def readFileIntoLists(filename):
    #lines = []
    full_filename = os.path.join(DIR, filename)
    with open(full_filename, encoding="utf8") as csvFile:
        positive = []
        negative = []
        reader = csv.reader(csvFile)
        data = list(reader)
        for row in data[1:]:
            if row[1] == '0':
                positive.append(row[0])
            else:
                negative.append(row[0])
        return positive, negative
def createLexicon(pos: list[str], neg: list[str], lemmatizer: WordNetLemmatizer):
    lexicon = []
    for _list in [pos[:MAX_LINES], neg[:MAX_LINES]]:
        for sentence in _list:
            _words = word_tokenize(sentence.lower())
            lexicon += list(_words)
        lexicon = [lemmatizer.lemmatize(l) for l in lexicon]
        word_counts = Counter(lexicon)
        result_lexicon = []
        for word in word_counts:
            if 1000 > word_counts[word] > 50:
                result_lexicon.append(word)
    return result_lexicon
def sampleHandling(sample: list[str], lexicon, classification, lemmatizer: WordNetLemmatizer):
    featureset = []
    for line in sample[:MAX_LINES]:
        _current_words = word_tokenize(line.lower())
        _current_words = [lemmatizer.lemmatize(l) for l in _current_words]
        features = np.zeros(len(lexicon))
        for word in _current_words:
            if word.lower() in lexicon:
                index = lexicon.index(word.lower())
                features[index] += 1
            features = list(features)
            featureset.append([features, classification])
    return featureset
def processData(filename):
    pos, neg = readFileIntoLists(filename)
    lemmatizer = WordNetLemmatizer()
    lexicon = createLexicon(pos, neg, lemmatizer)
    features = []
    features += sampleHandling(pos, lexicon, [1, 0], lemmatizer)
    features += sampleHandling(neg, lexicon, [0, 1], lemmatizer)
    random.shuffle(features)
    return features

if __name__ == '__main__':
    processData('Train.csv')