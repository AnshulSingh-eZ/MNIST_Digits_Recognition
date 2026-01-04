import cv2
from tensorflow.keras.models import load_model

img = cv2.imread("1.jpeg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28,28))
img = 255-img
img = img/255.0
img = img.reshape(1,28,28)

print(img)

model1 = load_model('adam_digit_rec.keras')
model2 = load_model('sgd_1_digit_rec.keras')
pred1 = model1.predict(img)
pred2 = model2.predict(img)

a = pred1.argmax()
b = pred2.argmax()
print(a, b)
