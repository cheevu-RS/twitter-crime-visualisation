import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from preprocess_tweets import samples, labels
from tensorflow.keras.layers import Embedding
from tensorflow.keras import layers
from tensorflow.keras import regularizers

# Shuffle the data
seed = 1337
rng = np.random.RandomState(seed)
rng.shuffle(samples)
rng = np.random.RandomState(seed)
rng.shuffle(labels)

# Extract a training & validation split
validation_split = 0.2
num_validation_samples = int(validation_split * len(samples))
train_samples = samples[:-num_validation_samples]
val_samples = samples[-num_validation_samples:]
train_labels = labels[:-num_validation_samples]
val_labels = labels[-num_validation_samples:]


vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200)
text_ds = tf.data.Dataset.from_tensor_slices(train_samples).batch(128)
vectorizer.adapt(text_ds)
print(vectorizer.get_vocabulary()[:5])

voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

path_to_glove_file = os.path.join(
    os.curdir, "../datasets/glove.twitter.27B.100d.txt"
)

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

num_tokens = len(voc) + 2
embedding_dim = 100
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

class_names = ["NonCrime", "Crime"]

embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=True,
)

int_sequences_input = keras.Input(shape=(None,), dtype="int64")
embedded_sequences = embedding_layer(int_sequences_input)

x = layers.Conv1D(128, 3, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(embedded_sequences)
x = layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True))(x)
x = layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
x = layers.Dense(128, activation="relu",     
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5))(x)
x = layers.Dropout(0.5)(x)
preds = layers.Dense(len(class_names)+1, activation="softmax")(x)

model = keras.Model(int_sequences_input, preds)
model.summary()

x_train = vectorizer(np.array([[s] for s in train_samples])).numpy()
x_val = vectorizer(np.array([[s] for s in val_samples])).numpy()

y_train = np.array(train_labels)
y_val = np.array(val_labels)

#model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
#              optimizer=keras.optimizers.Adam(1e-4),
#              metrics=['accuracy'])
model.compile(
	    loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc"]
	    )


history = model.fit(x_train, y_train, epochs=20,
                    validation_data=(x_val,y_val),
                    validation_steps=30)

test_loss, test_acc = model.evaluate(x_val,y_val)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

model.save("tweet_crime_non_crime_model")
