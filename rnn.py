from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

"Prediction of whether IMDB reviews are positive"

###################################################################################################
# Build a model
###################################################################################################

# Load and preprocess the IMDB dataset
max_features = 10000  # Number of words to consider as features
maxlen = 500  # Cut texts after this number of words (among top max_features most common words)

# Load the dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
"""
Each word in a corpus of film reviews is represented by number, such that a single review can 
be represented by a vector of numbers, and a training set etc. is an array of these vectors
"""

###################################################################################################
# Explore the reviews
###################################################################################################

# Load the word index
word_index = imdb.get_word_index()

# Reverse the word index to get a mapping from integers to words
reverse_word_index = {value: key for key, value in word_index.items()}

# Function to decode a review from integer sequence to text
def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

# Decode the first review in the training set
print("Decoded review:")
print(decode_review(x_train[0]))

# Optionally, you can also print the original encoded review and its label
print("\nEncoded review:")
print(x_train[0])
print("\nReview label (0 = negative, 1 = positive):")
print(y_train[0])

###################################################################################################
# Build a model
###################################################################################################

# Pad sequences with zeros to ensure equal length
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Build the RNN model
model = models.Sequential()
model.add(layers.Embedding(max_features, 128, input_length=maxlen))
"""
*Embedding layer* converts input integers into dense vectors of size of 128. Consider a randomly 
initalized set of vectors to represent the 5-token phrase "I love machine learning!"

[[ 0.01, 0.02, 0.03], # I
 [ 0.04, 0.05, 0.06], # love
 [ 0.07, 0.08, 0.09], # machine
 [ 0.10, 0.11, 0.12], # learning
 [ 0.13, 0.14, 0.15]] # !
 
The sentence "I love learning" would be represented by:

[[ 0.01, 0.02, 0.03], # I
 [ 0.04, 0.05, 0.06], # love
 [ 0.10, 0.11, 0.12]] # learning
 
*Advantages*:
1. *Dimensionality reduction:* a sentence does not need a one-hot encoded vector the size of the 
whole vocabulary
2. *Semantic meaning:* words with similar meanings or used in simiar context are mapped to 
nearby points in the embedding space
3. *Learning* during the training process as the model optimises against the task, capturing the 
syntactic and semantic properties

*Training* happens through adjusting embedding vectors by backpropogation, such that the 
embedding matrix is updated to improve the model's performance
"""

model.add(layers.SimpleRNN(128, return_sequences=False))
"""
*Simple RNN layer* processes the input sequence, capturing temporal dependencies
"""

model.add(layers.Dense(1, activation='sigmoid'))
"""
*Dense layer* is fully connected with a single neuron and sigmoid activation function, to output 
the probability of a review being positive
"""

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Display the model architecture
model.summary()

# Train the model
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# Plot training and validation accuracy and loss
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
"""
Loss = accuracy, as measured according to the problem, e.g:
- Regression -- MSE
- Classification -- Cross-Entropy

Note that after 6 epochs, training loss decreases and validation loss increases, which indicates
overfitting
"""

"""
1. Use more data (by collection or augmentation)
2. Regularise by penalising the loss function to constrain complexity
3. Regularise by penalising weight values
4. Regularise by random dropout of input units
5. Reduce model complexity, i.e. fewer layers and/or units per layer
6. Early stopping, when validation loss stops improving
7. Batch normalisation: normalise inputs of each layer, to improve training stability
8. Cross-validation
"""