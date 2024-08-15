import numpy as np
from tensorflow import keras
import keras
from keras import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras import layers
np.random.seed(0)
import cv2

# Dataset

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Converting to one-hot encoded format 

num_classe = 10
y_train = keras.utils.to_categorical(y_train, num_classe)
y_test = keras.utils.to_categorical(y_test, num_classe)
print(x_train.shape)

# Normalization

x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshaping to 3d model

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# ----------------------------------------------------------------------------------------

'''Model'''

model = Sequential([
    layers.Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    layers.AveragePooling2D(pool_size=(2, 2)),
    layers.Conv2D(16, (5, 5), activation='relu'),
    layers.AveragePooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# model.summary()

batch_size = 32 # 32 images at a time in network
epochs = 10

model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)


# ----------------------------------------------------------------------------------------

'''Evaluate'''

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test loss", test_loss)
print("Test accuracy", test_acc)

# ----------------------------------------------------------------------------------------

''' Testing with an image from outside'''

# Preparing the input image
path = "./datasets/a.png"
input_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
input_image = cv2.resize(input_image, (28, 28))
input_image = input_image.astype('float32') / 255.0
input_image = np.reshape(input_image, (1, 28, 28, 1))

# Making predictions
prediction = model.predict(input_image)
predicted_class = np.argmax(prediction)
print(f"Predicted class: {predicted_class}")

# ----------------------------------------------------------------------------------------

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# ----------------------------------------------------------------------------------------







model.save('handwritten_lenet_model.h5')



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
