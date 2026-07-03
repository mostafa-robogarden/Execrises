#import nltk
#nltk.download()
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
from collections import Counter
import pandas as pd
import os
import pickle

MAX_LINES = 100
DIR = 'dataset/yahoo_answers_csv'
def readFileIntoLists(filename):
    df = pd.read_csv(os.path.join(filename, 'train.csv'), header=None)
    print('---------------read the dataset--------------')
    df.rename(columns={0: 'labels', 1: 'question', 2: 'description', 3: 'answer'}, inplace=True)
    df.drop(columns=['description', 'answer'], inplace = True)
    df.rename(columns = {0: 'label', 1: 'question'}, inplace = True)
    data = df.to_dict(orient='records')
    print('---------------cleaning the data--------------')
    for i in range(len(data)):
        question: str = data[i]['question']
        question = question.replace('?', '')
        question = question.replace("n't", ' not')
        question = question.replace("'s", ' is')
        question = question.replace("'re", ' are')
        question = question.replace(',', '')
        data[i]['question'] = question
    return data
def featureLabelSplit(data):
    X = []
    y = []
    for features, labels in data:
         X.append(features)
         y.append(labels)
    X = np.array(X)
    return X, y
def createLexicon(data, lemmatizer: WordNetLemmatizer):
    lexicon = []
    print('---------------creating the lexicon--------------')
    for row in data[:MAX_LINES]:
        question = row['question']
        _words = word_tokenize(question.lower())
        lexicon += list(_words)
    lexicon = [lemmatizer.lemmatize(l) for l in lexicon]
    print('---------------counting words--------------')
    word_counts = Counter(lexicon)
    result_lexicon = []
    question_heads = ['what', 'why', 'how', 'when', 'where', 'which']
    for count in word_counts:
        if word_counts[count] < 20 and word_counts[count] not in question_heads:
            result_lexicon.append(count)
    print('---------------removed frequent unnecessary words--------------')
    return result_lexicon
def sampleHandling(sample, lexicon, lemmatizer: WordNetLemmatizer):
    featureset = []
    print('---------------creating features--------------')
    for line in sample[:MAX_LINES]:
        classification = [0] * 10
        label = line['labels']
        classification[int(label) - 1] = 1
        line = line['question']
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
    data = readFileIntoLists(filename)
    lemmatizer = WordNetLemmatizer()
    lexicon = createLexicon(data, lemmatizer)
    features = []
    features += sampleHandling(data, lexicon, lemmatizer)
    print('---------------shuffling the datat--------------')
    random.shuffle(features)
    print('---------------saving the data to file--------------')
    features, labels = featureLabelSplit(features)
    with open('features.pickle', 'wb') as file:
        pickle.dump(features, file)
    with open('labels.pickle', 'wb') as file:
        pickle.dump(labels, file)
    print('---------------printing a sample--------------')
    print(features[0])
    return features


processData(DIR)