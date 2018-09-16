import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

# nltk.download()

lemmatizer = WordNetLemmatizer()
nr_lines = 10000000


def create_lexicon(pos, neg):
    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for line in contents[:nr_lines]:
                # create a list of the words in the line
                all_words = word_tokenize(line.lower())
                # concat with the existing ones
                lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(word) for word in lexicon]
    # vector de frecventa pt cuvinte
    w_counts = Counter(lexicon)

    l2 = []

    for w in w_counts:
        if 1000 > w_counts[w]:
            # avoid common words like prepositions
            l2.append(w)

    print(len(l2))
    return l2


def handle_sample(sample, lexicon, classification):
    feature_set = []
    """
    [
    [[0 1 0 1 1 0],[1 0]] sentence with word 1,3,4 from lexicon appearing with positive label
    [...]
    ]
    """
    with open(sample, 'r') as f:
        contents = f.readlines()
        for line in contents[:nr_lines]:
            current_words = word_tokenize(line.lower())
            current_words = [lemmatizer.lemmatize(word) for word in current_words]

            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            feature_set.append([list(features), classification])

    """"""
    return feature_set


def create_feature_set_and_label(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    dataset = []
    dataset += handle_sample('positive.txt', lexicon, [1, 0])
    dataset += handle_sample('negative.txt', lexicon, [0, 1])
    random.shuffle(dataset)

    dataset = np.array(dataset)
    print(dataset.shape)
    """does tf.argmax([output]) == tf.argmax([1,0] or [0,1] given as expectation above)?"""
    testing_size = int(test_size * len(dataset))
    """take the first column aka features from all examples"""
    train_x = list(dataset[:, 0][:-testing_size])
    train_y = list(dataset[:, 1][:-testing_size])
    test_x = list(dataset[:, 0][-testing_size:])
    test_y = list(dataset[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_set_and_label('positive.txt', 'negative.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)
# create_lexicon("./negative.txt", "./positive.txt")
