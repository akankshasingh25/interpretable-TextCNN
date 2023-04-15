import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

##################################################################
#####                       1. ALGORITHM                     #####
##################################################################

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
    def get_class_predictions(self, model, X_set, y_set):
        # X_set and y_set can be training, development, testing, evaluation or validation sets
        """Find class labels for class probabilities predicted by a model"""
        y_prob = model.predict(X_set, verbose = 0)
        y_prob = y_prob[:, 0]   # probabilities

        precision, recall, thresholds = precision_recall_curve(y_set, y_prob)
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

    def compute_f1_score(self, model, X_set, y_set):
        """Compute the F1-score"""
        y_pred = self.get_class_predictions(model, X_set, y_set)
        f1 = f1_score(y_set, y_pred)
        return f1

    def algorithm(self, model, X_set, y_set, epsilon, top_percentile):
        """Find the interpretability threshold for a CNN model"""

        # Initialize F1-scores to be equal
        f1_dev = self.compute_f1_score(model, X_set, y_set)
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
            f1_dev_prime = self.compute_f1_score(model, X_set, y_set)
    
            if f1_dev_prime >= f1_dev + epsilon:
                top_percentile -= 0.01
            
            if top_percentile <= 0.2:
                break
    
        return self.percentile, self.trend_f1_score

##################################################################
#####                  2. DATA PREPARATION                   #####
##################################################################
 
# read the dataset
df = pd.read_csv(r'C:\Users\akank\Dropbox\My PC (LAPTOP-NQ9H8NTJ)\Documents\Sem 8\Project\code\dataset.csv')

X = df['tokenized_text'].values
y = df['label'].values

# Define the development dataset dev = X X Y as above
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size = 0.20, random_state= 42)

max_words = 20000   # maximum number of words to be used in the vocabulary
max_length = 200    # maximum length of the input sequences

# tokenize the text data
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1  # vocabulary size based on the tokenizer

X_train = tokenizer.texts_to_sequences(X_train)
X_dev = tokenizer.texts_to_sequences(X_dev)

# pad the sequences to a fixed length
X_train = pad_sequences(X_train, maxlen = max_length, padding='post')
X_dev = pad_sequences(X_dev, maxlen = max_length, padding='post')

##################################################################
#####            3. INTERPRETABILITY THRESHOLD               #####
##################################################################

# Define the parameter ε as described
epsilon = -0.01

# Define the top percentile for discarding n-grams
top_percentile = 0.99

# Define the CNN model trained for the task at hand
model = keras.models.load_model(r'C:\Users\akank\Dropbox\My PC (LAPTOP-NQ9H8NTJ)\Documents\Sem 8\Project\code\TextCNN.h5')

calc_interpret_thresh = Interpretability_Threshold()
benchmark_f1_score = calc_interpret_thresh.compute_f1_score(model, X_dev, y_dev)
print("The benchmark F1 Score is {}".format(benchmark_f1_score))

# f1_score = calc_interpret_thresh.compute_f1_score(model, X_train, y_train)
# print("The F1 Score is {}".format(f1_score))

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

thresholded_f1_score = calc_interpret_thresh.compute_f1_score(model, X_dev, y_dev)
print("The thresholded F1 Score is {}".format(thresholded_f1_score))

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

##################################################################
#####                        4. PLOTTING                     #####
##################################################################

# Plotting loss vs. epoch function
plt.plot(percentile, trend_f1_score)
plt.xlabel("Top Percentile")
plt.ylabel("F1 Score")
plt.title("F1 for different top percentile")
plt.show()
plt.savefig("F1 Score")
plt.figure(figsize=(6, 6))

# Plotting learned weights
weights = benchmark_dictionary["conv1d_1"]["weights"]
plt.subplot(1,2,1)
plt.imshow(weights[1])
plt.title("Learned Weights")
plt.xlabel("learned kernel")
plt.ylabel("filters")

weights = thresholded_dictionary["conv1d_1"]["weights"]
plt.subplot(1,2,2)
plt.imshow(weights[1])
plt.title("Thresholded Weights")
plt.xlabel("learned kernel")
plt.ylabel("filters")

plt.savefig("Fig1")
from keras import backend as K

# Plotting n-grams activation in filters
# Load the sentence of interest from the corpus and preprocess it
sentence = X[23]
tokenizer.fit_on_texts(sentence)
sentence = tokenizer.texts_to_sequences(sentence)
preprocessed_sentence = pad_sequences(sentence, maxlen = max_length, padding='post')

# Get the weights of the convolutional layer
conv_weights = model.layers[2].get_weights()[0]

# Get the output of the first convolutional layer for the input sentence
get_activations = K.function([model.layers[0].input], [model.layers[2].output])
activations = get_activations(np.array([preprocessed_sentence]))[0]
activations = activations[0]
print(activations.shape)

n_grams = conv_weights.shape[0]
num_filters = conv_weights.shape[2]

"""Plotting"""

##################################################################
#####            3. INTERPRETABILITY METRICES               #####
##################################################################

# # This implementation takes as input the interpretability array, which has shape (K, C, T) 
# # and represents the interpretability scores for each of the K regions, 
# # each of the C classes, 
# # and each of the T annotations. 

# # It also takes the R array, which has shape (K, M) and represents the set of instances that are relevant,
# # the y array, which has shape (M,) and represents the true labels of the instances, 
# # and the An array, which has shape (T, M) and represents the set of instances that have each annotation.

# # The implementation loops over all possible values of k, c, and t, 
# # and computes the numerator and denominator of the interpretability score for each triplet using nested loops. 
# # The numerator is the number of instances that belong to region k, have true label c, and have annotation t. 
# # The denominator is the number of instances that either belong to region k or have annotation t, and have true label c.
# # Finally, the interpretability score is computed by dividing the numerator by the denominator, and is stored in the interpretability array. 
# # The function returns the updated interpretability array.

# def compute_interpretability(interpretability, R, y, An):
#     M = R.shape[0]
#     for k in range(interpretability.shape[0]):
#         for c in range(interpretability.shape[1]):
#             for t in range(interpretability.shape[2]):
#                 numerator = 0
#                 denominator = 0
#                 for i in range(M):
#                     if y[i] == c:
#                         if R[k][i]:
#                             if An[t][i]:
#                                 numerator += 1
#                             denominator += 1
#                         elif An[t][i]:
#                             denominator += 1
#                 interpretability[k][c][t] = numerator / denominator
#     return interpretability

# # Note that this code assumes that R_k and An_t are numpy arrays of size M. 

# interpretability_kct = (np.sum([(np.sum((y_train == c) & An_t[x_i])) for x_i in range(M) if np.sum((y_train==c) & An_t[x_i]) > 0])) / (np.sum([(np.sum((y_train==c) | An_t[x_i])) for x_i in range(M) if np.sum((y_train==c) | An_t[x_i]) > 0])))
# # R_k is a list of numpy arrays where each numpy array represents the relevant for a given example x_i, 
# # and y is the corresponding list of ground truth labels

# R_k = []
# interpretability_k_t = np.sum([np.intersect1d(R_k[x_i], An_t[x_i]).size for x_i in range(M)]) / np.sum([np.union1d(R_k[x_i], An_t[x_i]).size for x_i in range(M)])
# # An_t, M, and c before using this code.
# interpretability_k_c = sum([len([x for x in R_k_x if y_i == c]) for R_k_x, y_i in zip(R_k, y)]) / sum([len(R_k_x) for R_k_x in R_k])
# y_true = ...
# y_pred = ...
# num_classes = 2

# TP_k = np.sum([np.sum((y_true == c) & (y_pred == c)) for c in range(num_classes)])
# TN_k = np.sum([np.sum((y_true != c) & (y_pred != c)) for c in range(num_classes)])
# FP_k = np.sum([np.sum((y_true != c) & (y_pred == c)) for c in range(num_classes)])
# FN_k = np.sum([np.sum((y_true == c) & (y_pred != c)) for c in range(num_classes)])
# accuracy_k = (TP_k + TN_k) / (TP_k + TN_k + FP_k + FN_k)
# precision_k = TP_k / (TP_k + FP_k)
# recall_k = TP_k / (TP_k + FN_k)
# F1_k = 2 * ((precision_k * recall_k) / (precision_k + recall_k))
