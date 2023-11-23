import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

# Define the vocabulary of circuit components
vocab = ['RES', 'LED', 'POT', 'LINE']

# Create a mapping from components to indices
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for i, word in enumerate(vocab)}

# Generate random training data
data = []
for _ in range(1000):
    circuit_length = np.random.randint(5, 20)
    circuit = np.random.choice(vocab, circuit_length)
    data.append(circuit)

# Convert data to indices
indexed_data = [[word_to_index[word] for word in circuit] for circuit in data]

# Create training sequences
input_sequences = []
output_sequences = []
for circuit in indexed_data:
    for i in range(1, len(circuit)):
        input_sequences.append(circuit[:i])
        output_sequences.append(circuit[i])

# Pad sequences for consistent input size
max_sequence_length = max(map(len, input_sequences))
padded_input_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    input_sequences, maxlen=max_sequence_length, padding='pre'
)

# Convert to numpy arrays
x_train = np.array(padded_input_sequences)
y_train = np.array(output_sequences)

# Build a simple LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(vocab), output_dim=10, input_length=max_sequence_length))
model.add(LSTM(100))
model.add(Dense(len(vocab), activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Generate a sequence using the trained model
def generate_sequence(model, seed_sequence, max_length=10):
    generated_sequence = seed_sequence.copy()
    for _ in range(max_length):
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            [generated_sequence], maxlen=max_sequence_length, padding='pre'
        )
        predicted_index = np.argmax(model.predict(padded_sequence), axis=-1)
        generated_sequence.append(predicted_index[0])
    generated_words = [index_to_word[index] for index in generated_sequence]
    return generated_words

# Generate a sequence starting from a random seed
seed_sequence = [np.random.randint(len(vocab))]
generated_sequence = generate_sequence(model, seed_sequence)
print("Generated Circuit:", generated_sequence)
