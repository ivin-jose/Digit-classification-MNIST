import tensorflow as tf
from keras import *
import cv2
import numpy as np
from keras.models import load_model


model = load_model('handwritten_model.h5')

# Preparing the input image
path = "./datasets/three.png"
input_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
input_image = cv2.resize(input_image, (28, 28))
input_image = input_image.astype('float32') / 255.0
input_image = np.reshape(input_image, (1, 28*28))

# Making predictions
prediction = model.predict(input_image)
predicted_class = np.argmax(prediction)

print(f"Predicted class: {predicted_class}")
