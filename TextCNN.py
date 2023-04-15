import pandas as pd
import numpy as np
from matplotlib import pyplot
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, MaxPooling1D, Input, concatenate, Dense, Dropout, GlobalMaxPooling1D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import plot_model
# read the dataset
df = pd.read_csv(r'C:\Users\akank\Dropbox\My PC (LAPTOP-NQ9H8NTJ)\Documents\Sem 8\Project\code\dataset.csv')

X = df['tokenized_text'].values
y = df['label'].values

##################################################################
#####                  1. DATA PREPARATION                   #####
##################################################################

X_train, X_test , y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 42)
max_words = 20000   # maximum number of words to be used in the vocabulary
max_length = 200    # maximum length of the input sequences
embedding_dim = 300

# tokenize the text data
tokenizer = Tokenizer(num_words= max_words)
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1  # vocabulary size based on the tokenizer

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# pad the sequences to a fixed length
X_train = pad_sequences(X_train, maxlen = max_length, padding='post')
X_test = pad_sequences(X_test, maxlen = max_length, padding='post')


# load the pre-trained embeddings using Gensim downloader
word_vectors = api.load('word2vec-google-news-300')

# create an embedding matrix for the vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < vocab_size and i < max_words and word in word_vectors:
        embedding_matrix[i] = word_vectors[word]

##################################################################
#####                        2. TextCNN                      #####
##################################################################
input = Input(shape=(None,), dtype="int64")
embedding_layer = Embedding(
    vocab_size,
    embedding_dim,
    weights=[embedding_matrix],
    input_length= max_length,
    trainable=False
)
embedded_sequence = embedding_layer(input)
conv_3 = Conv1D(100, 3, activation="relu")(embedded_sequence)
conv_4 = Conv1D(100, 4, activation="relu")(embedded_sequence)
conv_5 = Conv1D(100, 5, activation="relu")(embedded_sequence)
maxpool_3 = MaxPooling1D(3)(conv_3)
maxpool_4 = MaxPooling1D(4)(conv_4)
maxpool_5 = MaxPooling1D(5)(conv_5)
concatenated = concatenate([maxpool_3, maxpool_4, maxpool_5], axis=1)
fc = Dense(150, activation='relu')(concatenated)
dropout = Dropout(0.5)(fc)
pool = GlobalMaxPooling1D()(dropout)
output = Dense(1, activation='sigmoid')(pool)
model = Model(input, output)

model.summary()

# compile the model using Adam optimizer and categorical cross-entropy loss
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# %load_ext tensorboard
# # rm -rf logs

##################################################################
#####                        3. TRAINING                     #####
##################################################################

log_folder = 'logs'
callbacks = [
            EarlyStopping(patience = 10),
            TensorBoard(log_dir=log_folder)
            ]
num_epochs = 25
# train the model using mini-batches of size 100
history = model.fit(X_train, y_train, 
                    batch_size=100, 
                    epochs = num_epochs, 
                    validation_data=(X_test, y_test),
                    callbacks=callbacks)
# evaluate the model
_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

##################################################################
#####                        4. PLOTTING                     #####
##################################################################

# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()

# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()

##################################################################
#####                  3. SAVING AND LOADING                 #####
##################################################################
# save model and architecture to single file
model.save("TextCNN.h5")
print("Saved model to disk")

# load model
model = load_model('TextCNN.h5')
plot_model(model, to_file='model.png')

# evaluate loaded model on test data
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
score = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
