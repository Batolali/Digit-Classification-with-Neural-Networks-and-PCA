import os
import cv2
import numpy as np
from skimage.transform import resize
from sklearn.decomposition import PCA
from tensorflow import keras
from keras.applications.densenet import layers


bins = 9
pix_per_cell = (8,8)
cell_per_block = (2, 2)

img_train = []
class_train = []

img_test = []
class_test = []

n = 0

#The code defines a list of folders (folders)
# that contains the paths to the training and testing data.
#The code loops through each folder in folders, and for each folder,
# it loops through the 10 classes of digits (0-9).
#For each class of digits, the code reads in each image file and resizes it to a 64x32 pixel
# image using the resize function.
#We have created a zero matrix consisting of three dimensions: (length, width, and bin).

folders = ["Digit_training/", "Digit_testing/"]
for folder in folders:
    for classes in range(10):
        folder_path = folder + str(classes) + "/"
        for file in os.listdir(folder_path):
            image = cv2.imread(os.path.join(folder_path, file))
            img = resize(image, (64, 32))
#computes the gradient magnitude and orientation using the Sobel operator
            G_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            G_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

            magnitude = np.sqrt(G_x ** 2 + G_y ** 2)
            orientation = np.arctan2(G_y, G_x) * (180 / np.pi) % 180

            cells_num_x = int(img.shape[1] / pix_per_cell[1])
            cells_num_y = int(img.shape[0] / pix_per_cell[0])

            hist = np.zeros((cells_num_y, cells_num_x, bins))

            for i in range(cells_num_y):
                for j in range(cells_num_x):
                    # selects the subset of the magnitude and angle arrays that correspond to the current cell
                    magnitude_of_cell = magnitude[i * pix_per_cell[0]:(i + 1) * pix_per_cell[0],
                                     j * pix_per_cell[1]:(j + 1) * pix_per_cell[1]]
                    orientation_of_cell = orientation[i * pix_per_cell[0]:(i + 1) * pix_per_cell[0],
                                 j * pix_per_cell[1]:(j + 1) * pix_per_cell[1]]

                    # computes the histogram of oriented gradients for the current cell
                    hist[i, j] = np.histogram(orientation_of_cell, bins=bins, range=(0, 180), weights=magnitude_of_cell)[0]

            norm = np.zeros((cells_num_y - cell_per_block[0] + 1, cells_num_x - cell_per_block[1] + 1,
                     cell_per_block[0] * cell_per_block[1] * bins))
## in order to calculate the first block, and second block
## we want them to pass line by line, column and column the important things is that we pass them all
            for i in range(norm.shape[0]):
                for j in range(norm.shape[1]):
                    hist_of_block = hist[i:i + cell_per_block[0], j:j + cell_per_block[1], :].flatten()
                    norm[i, j] = hist_of_block / np.sqrt(np.sum(hist_of_block ** 2) + 1e-7)

            features = norm.flatten()
## if the folder is first add it on imag trening, if it is zero add it on image test.
            if n == 0:
                img_train.append(features)
                class_train.append(classes)
            else:
                img_test.append(features)
                class_test.append(classes)


    n=1


img_train = np.array(img_train)
class_train = np.array(class_train)
img_test = np.array(img_test)
class_test = np.array(class_test)


# *********************************** (B) ****************************
# Define three network architecture
# simple feedforward neural network
# model1  with one hidden layer of 64 units with a ReLU Function and output layer have 10 units
model1 = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(img_train[0].shape)),
    layers.Dense(10, activation='softmax')
])

# feedforward neural network
# model1  with one hidden layer of 64 and 128 units with a ReLU Function and output layer have 10 units

model2 = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(img_train[0].shape)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
# model1  with one hidden layer of 64,256, 128 units with a ReLU Function and output layer have 10 units

model3 = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(img_train[0].shape)),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile the models
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model1.fit(img_train, class_train, epochs=10, batch_size=32)
model2.fit(img_train, class_train, epochs=10, batch_size=32)
model3.fit(img_train, class_train, epochs=10, batch_size=32)

# Evaluate the first network architecture
test_loss1, test_acc1 = model1.evaluate(img_test, class_test)
test_loss2, test_acc2 = model2.evaluate(img_test, class_test)
test_loss3, test_acc3 = model3.evaluate(img_test, class_test)

print("model1  : Test accuracy =", test_acc1)
print("model2  : Test accuracy =", test_acc2)
print("model3  : Test accuracy =", test_acc3)


# ******************************* (C) *******************************
pca = PCA(n_components=0.5)
img_train = pca.fit_transform(img_train)
img_test = pca.transform(img_test)

# Define three network architecture
# simple feedforward neural network
model1_PCA = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(img_train[0].shape)),
    layers.Dense(10, activation='softmax')
])

# feedforward neural network
model2_PCA = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(img_train[0].shape)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model3_PCA = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(img_train[0].shape)),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile the models
model1_PCA.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2_PCA.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model3_PCA.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model1_PCA.fit(img_train, class_train, epochs=10, batch_size=32)
model2_PCA.fit(img_train, class_train, epochs=10, batch_size=32)
model3_PCA.fit(img_train, class_train, epochs=10, batch_size=32)

# Evaluate the first network architecture
test_loss1_PCA, test_acc1_PCA = model1_PCA.evaluate(img_test, class_test)
test_loss2_PCA, test_acc2_PCA = model2_PCA.evaluate(img_test, class_test)
test_loss3_PCA, test_acc3_PCA = model3_PCA.evaluate(img_test, class_test)

print("model1  : Test accuracy =", test_acc1_PCA)
print("model2  : Test accuracy =", test_acc2_PCA)
print("model3  : Test accuracy =", test_acc3_PCA)


#The test accuracy for each model before and after applying PCA is written to a
# file named "accuracy.txt".
with open("accuracy.txt", "w") as file:
    file.write("******************* Befor PCA ****************************** \n")
    file.write("Test accuracy with 3 different types of neural network architecture. \n")
    file.write("model1  :  accuracy = {}\n".format(test_acc1))
    file.write("model2  :  accuracy = {}\n".format(test_acc2))
    file.write("model3  :  accuracy = {}\n".format(test_acc3))

    file.write("******************* After 0.5 PCA ****************************** \n")
    file.write("Test accuracy with 3 different types of neural network architecture.\n")
    file.write("model1  :  accuracy = {}\n".format(test_acc1_PCA))
    file.write("model2  :  accuracy = {}\n".format(test_acc2_PCA))
    file.write("model3  :  accuracy = {}\n".format(test_acc3_PCA))

