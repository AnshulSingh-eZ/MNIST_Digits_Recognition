import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(X_train[0])
print(Y_train[0])
print(X_train.shape)

X_train, X_test = X_train/255.0, X_test/255.0 

model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(784, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
optimizer = 'adam'
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
data = model.fit(X_train, Y_train, epochs=10)
model.save("adam_digit_rec.keras")
print("Model successfully saved!!")
pd.DataFrame(data.history).to_csv("adam_digit_rec.csv", index=False)
print("Data saved successfully!!")

model.evaluate(X_test, Y_test)
