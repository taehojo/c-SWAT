import tensorflow as tf

kernel_size = 5
filters = 64
N_hidden1 = 32
N_hidden2 = 16
N_hidden3 = 8
N_out = 2

def create_model(N_in):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters, kernel_size, activation="relu", input_shape=(N_in, 1)))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dense(N_in, activation="relu"))
    model.add(tf.keras.layers.Dense(N_hidden1, activation="relu"))
    model.add(tf.keras.layers.Dense(N_hidden2, activation="relu"))
    model.add(tf.keras.layers.Dense(N_hidden3, activation="relu"))
    model.add(tf.keras.layers.Dense(N_out, activation='softmax', name="dense_e"))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model