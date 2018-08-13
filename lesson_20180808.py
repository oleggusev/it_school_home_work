from keras.datasets import mnist

#(X_train, y_train), (X_test, y_test) = mnist.load_data()

from scipy.signal import convolve2d
import numpy as np
import cv2
from matplotlib import pyplot as plt


def show_differences(kernel):
    convolved = convolve2d(image, kernel)
    fig = plt.figure(figsize=(15, 15))
    plt.subplot(121)
    plt.title('Original image')
    plt.axis('off')
    plt.imshow(image, cmap='gray')

    plt.subplot(122)
    plt.title('Convolved image')
    plt.axis('off')
    plt.imshow(convolved, cmap='gray')
    #  plt.show()
    return convolved


# image = cv2.imread('data/cats/cat.png')
# # converting the image to grayscale
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#
# # 1st
# kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9
# output = show_differences(kernel)
#
#
# # 2nd
# kernel = np.ones((8,8), np.float32)/64
# dx = show_differences(kernel)
#
#
# # 3d
# kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
# dx = show_differences(kernel)
#
#
# # 4th
# kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
# dy = show_differences(kernel)


# 5th
# mag = np.hypot(dx, dy)  # magnitude
# mag *= 255.0 / np.max(mag)  # normalize (Q&D)
#
# fig = plt.figure(figsize=(15, 15))
# plt.subplot(121)
# plt.title('Original image')
# plt.axis('off')
# plt.imshow(image, cmap='gray')
#
# plt.subplot(122)
# plt.title('Convoluted image with highlighted edges')
# plt.axis('off')
# plt.imshow(mag, cmap='gray')
# plt.show()

# https://www.cc.gatech.edu/~hays/compvision/proj6/deepNetVis.png

import numpy as np
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt

from keras import backend as K
K.set_image_dim_ordering('th')

(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0])
plt.show()

# преоброзование данных
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# print(Y_train)

model = Sequential()

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1, 28, 28)))
model.add(Convolution2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

res = model.fit(X_train, Y_train, batch_size=32, nb_epoch=1, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

