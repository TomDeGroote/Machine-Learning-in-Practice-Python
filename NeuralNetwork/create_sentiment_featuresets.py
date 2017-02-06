import os
from nltk.tokenize import word_tokenize
import numpy as np
import random
# Pickle helps to save progress, so you don't need to start from the beginning every time
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer

# Tp download the natural language toolkit, do:
# import nltk
# nltk.download()

# Define the lemmatizer
lemmatizer = WordNetLemmatizer()
# Set the number of lines used (make smaller if you want a smaller data size)
hm_lines = 100000


# Function that creates a lexicon given a file with positive examples and a file with negative examples
def create_lexicon(pos, neg):
    # Initialise the lexicon
    lexicon = []
    with open(pos, 'r') as f:
        # Read all lines in the pos file
        contents = f.readlines()
        # For every line tokenize the words of the line and add them to the lexicon
        for l in contents:
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    # Same principle as for the pos file above
    with open(neg, 'r') as f:
        contents = f.readlines()
        for l in contents:
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    # Lemmatize all words in our lexicon
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    # Count the number of occurrences of every word in our lexicon
    w_counts = Counter(lexicon)

    # Initialise a new lexicon where we remove very rare and very common words like "a", "or", "and"
    l2 = []
    for w in w_counts:
        # If a given word shows up more than 50 times but less than a 1000 add them to our filtered lexicon
        # These values should be tweaked to fit your data set (e.g. based on %) but this is a simple example.
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    # Show the length of this new lexicon
    print("Initial lexicon length", len(lexicon), "Filtered lexicon length:", len(l2))
    return l2


# Function that maps the occurrence of a word in a sample to the zeros array representing all words
# in a lexicon by turning them "on" (changing them to 1)
def sample_handling(sample, lexicon, classification):
    # Initialise our feature set
    featureset = []

    with open(sample, 'r') as f:
        # Read all lines in the sample file
        contents = f.readlines()
        # Only handle the first hm_line nr of lines
        for l in contents[:hm_lines]:
            # Tokenize the words in the line
            current_words = word_tokenize(l.lower())
            # Lemmatize the words from the line
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            # Initialize the features with a zer array
            features = np.zeros(len(lexicon))
            # For every word in our lines' words turn the appropriate element in the features hot
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            # Add the features of this line to our feature set with the given classification
            features = list(features)
            featureset.append([features, classification])
    return featureset


# Function to create training and test sets
def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    # Use the create_lexicon function to generate a lexicon for the pos and neg files
    lexicon = create_lexicon(pos, neg)

    # Create features based on all pos and neg features
    features = []
    features += sample_handling(pos, lexicon, [1, 0])
    features += sample_handling(neg, lexicon, [0, 1])
    random.shuffle(features)
    features = np.array(features)

    # Calculate the number of testing examples
    testing_size = int(test_size*len(features))

    # Generate training and testing data sets (x and y)
    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y


# Runner
if __name__ == '__main__':
    # Our root directory of this file (In the Neural Network folder)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels(ROOT_DIR + '/../data/pos.txt',
                                                                      ROOT_DIR + '/../data/neg.txt')
    # Write results away with pickle
    with open(ROOT_DIR + '/../pickle_dump/sentiment_set.pickle', 'wb') as f:
        pickle.dump(([train_x, train_y, test_x, test_y]), f)
