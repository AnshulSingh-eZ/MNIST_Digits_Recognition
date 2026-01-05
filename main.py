import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(X_train[0])
print(Y_train[0])
print(X_train.shape)

X_train, X_test = X_train/255.0, X_test/255.0 

model = Sequential([
    Conv2D(32, (2,2), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (2,2), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
optimizer = 'adam'
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
data = model.fit(X_train, Y_train, epochs=5)
model.save("adam_digit_rec.keras")
print("Model successfully saved!!")
pd.DataFrame(data.history).to_csv("adam_digit_rec.csv", index=False)
print("Data saved successfully!!")

model.evaluate(X_test, Y_test)