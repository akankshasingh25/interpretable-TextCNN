import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

class Interpretability_Threshold():
    """Find the interpretability threshold for a CNN model"""

    def __init__(self):
        self.percentile = []
        self.trend_f1_score = []

    def get_top_percentile_oflayer(self, arr, top_percentile):
        """Compute the threshold value for the top percentile of an array"""
        return np.percentile(np.abs(arr), 100 - top_percentile * 100)

    def apply_thresholding(self, arr, thresholded_values):
        """Apply a threshold to an array"""
        return np.where(np.abs(arr) >= thresholded_values, arr, 0)

    # Define a function to calculate the F1 Score from predicted probabilities
    def get_class_predictions(self, model, X, y):
        """Find class labels for class probabilities predicted by a model"""
        y_prob = model.predict(X, verbose = 0)
        y_prob = y_prob[:, 0]   # probabilities

        precision, recall, thresholds = precision_recall_curve(y, y_prob)
        # convert to f1 score
        fscore = (2 * precision * recall) / (precision + recall)
        # locate the index of the largest f1 score
        ix = np.argmax(fscore)
        decision_boundary = fscore[ix]
        
        # find class labels
        y_pred = []
        for i in range(len(y_prob)):
            if y_prob[i] > decision_boundary:
                y_pred.append(1)
            else: 
                y_pred.append(0)
            
        return y_pred

    def compute_f1_score(self, model, X, y):
        """Compute the F1-score for the development dataset"""
        y_pred = self.get_class_predictions(model, X, y)
        f1 = f1_score(y, y_pred)
        return f1

    def algorithm(self, model, X, Y, epsilon, top_percentile):
        # Initialize F1-scores to be equal
        f1_dev = self.compute_f1_score(model, X_dev, y_dev)
        f1_dev_prime = f1_dev

        while f1_dev_prime >= f1_dev + epsilon:

            f1_dev = f1_dev_prime
            
            self.trend_f1_score.append(f1_dev_prime)
            self.percentile.append(top_percentile)

            # Compute threshold for each layer
            for layer in model.layers:
                if 'conv' in layer.name:    # apply only to convolutional layers
                    thresholded_weights = self.get_top_percentile_oflayer(benchmark_dictionary["{}".format(layer.name)]["weights"], top_percentile)
                    thresholded_bias = self.get_top_percentile_oflayer(benchmark_dictionary["{}".format(layer.name)]["bias"], top_percentile)
                    
                    layer.set_weights([self.apply_thresholding(layer.get_weights()[0], thresholded_weights),
                                    self.apply_thresholding(layer.get_weights()[1], thresholded_bias)])
            
            # Compute F1-score with thresholded model
            f1_dev_prime = self.compute_f1_score(model, X_dev, y_dev)
    
            if f1_dev_prime >= f1_dev + epsilon:
                top_percentile -= 0.01
            
            if top_percentile <= 0.05:
                break
    
        return self.percentile, self.trend_f1_score
    
# read the dataset
df = pd.read_csv(r'C:\Users\akank\Dropbox\My PC (LAPTOP-NQ9H8NTJ)\Documents\Sem 8\Project\code\dataset.csv')

X = df['tokenized_text'].values
y = df['label'].values

X_train, X_test , y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 42)

max_words = 20000   # maximum number of words to be used in the vocabulary
max_length = 200    # maximum length of the input sequences

# tokenize the text data
tokenizer = Tokenizer(num_words= max_words)
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1  # vocabulary size based on the tokenizer

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# pad the sequences to a fixed length
X_train = pad_sequences(X_train, maxlen = max_length, padding='post')
X_test = pad_sequences(X_test, maxlen = max_length, padding='post')
# Define the development dataset dev = X X Y as above
X_dev = X_test
y_dev = y_test

# Define the parameter ε as described
epsilon = -0.01

# Define the top percentile for discarding n-grams
top_percentile = 0.99

# Define the CNN model trained for the task at hand
model = keras.models.load_model(r'C:\Users\akank\Dropbox\My PC (LAPTOP-NQ9H8NTJ)\Documents\Sem 8\Project\code\TextCNN.h5')

calc_interpret_thresh = Interpretability_Threshold()
benchmark_f1_score = calc_interpret_thresh.compute_f1_score(model, X_test, y_test)
print(benchmark_f1_score)

benchmark_dictionary = {}
layer_weights = []
layer_bias = []

for layer in model.layers:
    if 'conv' in layer.name:    # apply only to convolutional layers
        layer_weights = layer.get_weights()[0]
        layer_bias = layer.get_weights()[1]
        benchmark_dictionary.setdefault("{}".format(layer.name),{})
        benchmark_dictionary["{}".format(layer.name)]["weights"] = layer_weights
        benchmark_dictionary["{}".format(layer.name)]["bias"] = layer_bias
        layer_weights = []
        layer_bias = []
percentile, trend_f1_score = calc_interpret_thresh.algorithm(model, X_dev, y_dev, epsilon, top_percentile)

thresholded_f1_score = calc_interpret_thresh.compute_f1_score(model, X_test, y_test)
print(thresholded_f1_score)

thresholded_dictionary = {}
layer_weights = []
layer_bias = []

for layer in model.layers:
    if 'conv' in layer.name:    # apply only to convolutional layers
        layer_weights = layer.get_weights()[0]
        layer_bias = layer.get_weights()[1]
        thresholded_dictionary.setdefault("{}".format(layer.name),{})
        thresholded_dictionary["{}".format(layer.name)]["weights"] = layer_weights
        thresholded_dictionary["{}".format(layer.name)]["bias"] = layer_bias
        layer_weights = []
        layer_bias = []
# Plotting loss vs. epoch function
plt.plot(percentile, trend_f1_score)
plt.xlabel("Top Percentile")
plt.ylabel("F1 Score")
plt.title("F1 for different top percentile")
plt.show()
plt.savefig("F1 Score")