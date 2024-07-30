import numpy as np
from tensorflow import keras
import keras
from keras import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.layers import Dense
np.random.seed(0)
import cv2
# Dataset

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print("x-train : ", x_train.shape)
# print("y-train : ", y_train.shape)
# print("x-test : ", x_test.shape)
# print("y-test : ", y_test.shape)

# Visualize examples

num_classes = 10
f, ax = plt.subplots(1, num_classes, figsize=(20, 20))

for i in range(0, num_classes):
  sample = x_train[y_train == i][0]
  ax[i].imshow(sample, cmap='gray')
  ax[i].set_title(f"Label : {i}", fontsize=16)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# print(x_train.shape)

# Normalization

x_train = x_train / 255.0
x_test = x_test / 255.0

#  Flattening the array to 1 dimension (Reshapeing)

x_train= x_train.reshape(x_train.shape[0], -1) # Meaning (60000, 28*28)
# print(x_train.shape) # output (60000, 784)


# ----------------------------------------------------------------------------------------

'''# Model creation 98.21% accuracy'''

model = Sequential([
    Dense(256, activation='relu', input_shape=(784,)),
    Dense(512, activation='relu'),
    layers.Dropout( 0.25),
    Dense(512, activation='relu'),
    layers.Dropout( 0.25),
    Dense(num_classes, activation='softmax')
])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# model.summary()

batch_size = 512 # 512 images at a time in network
epochs = 10

model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)


# ----------------------------------------------------------------------------------------

'''Evaluate'''

x_test = x_test.reshape(x_test.shape[0], -1)
test_loss, test_acc = model.evaluate(x_test, y_test)
# print("test loss", test_loss)

# ----------------------------------------------------------------------------------------

''' Testing with an image from outside'''

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

# ----------------------------------------------------------------------------------------

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# ----------------------------------------------------------------------------------------



print(y_pred)
print(y_pred_classes)


# ----------------------------------------------------------------------------------------












'''# testing with an random image with in the x_test dataset'''

random_idx = np.random.randint(0, x_test.shape[0])
x_sample = x_test[random_idx]
y_true = np.argmax(y_test, axis=1)
y_sample_true = y_true[random_idx]
y_sample_pred_class = y_true[random_idx]

print(f"Predicted : {y_sample_pred_class} True : {y_sample_true}")
plt.title(f"Predicted : {y_sample_pred_class} True : {y_sample_true}")
plt.imshow(x_sample.reshape(28, 28), cmap='gray')

'''# graphical user presentaion of true value vs predicted value'''

confusion_mtx = confusion_matrix(y_true, y_pred_classes)

'''# plot'''
fig, ax = plt.subplots(figsize=(10, 15))
ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax, cmap='Greens')
ax.set_xlabel('Predcited Label')
ax.set_ylabel('True Label')
ax.set_title('confusion Matrix')
