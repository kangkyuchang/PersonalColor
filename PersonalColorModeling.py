import os
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import layers, models

trainCSVPath  = "dataset/train/_classes.csv"
trainImgDir = "dataset/train/image"
testCSVPath = "dataset/test/_classes.csv"
testImgDir = "dataset/test/image"

trainDataFrame = pd.read_csv(trainCSVPath)
testDataFrame = pd.read_csv(testCSVPath)

trainFilepaths = trainDataFrame["filename"].apply(lambda x:os.path.join(trainImgDir, x)).tolist()
trainLabels = trainDataFrame.iloc[:,1].values
testFilepaths = testDataFrame["filename"].apply(lambda x:os.path.join(testImgDir, x)).tolist()
testLabels = testDataFrame.iloc[:,1].values
validationFilepaths = testDataFrame["filename"].apply(lambda x:os.path.join(testImgDir, x)).tolist()
validationLabels = testDataFrame.iloc[:,1].values

def load_image_and_label(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [416, 416])
    img = img / 255.0
    return img, label

trainPathDs = tf.data.Dataset.from_tensor_slices((trainFilepaths, trainLabels))
trainImageLabelDs = trainPathDs.map(lambda path, label: load_image_and_label(path, label), num_parallel_calls=tf.data.experimental.AUTOTUNE)

trainDs = trainImageLabelDs.shuffle(buffer_size=len(trainFilepaths)).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

testPathDs = tf.data.Dataset.from_tensor_slices((testFilepaths, testLabels))
testImageLabelDs = testPathDs.map(lambda path, label: load_image_and_label(path, label), num_parallel_calls=tf.data.experimental.AUTOTUNE)

testDs = testImageLabelDs.shuffle(buffer_size=len(trainFilepaths)).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

validationPathDs = tf.data.Dataset.from_tensor_slices((validationFilepaths, validationLabels))
validationImageLabelDs = validationPathDs.map(lambda path, label: load_image_and_label(path, label), num_parallel_calls=tf.data.experimental.AUTOTUNE)

validationDs = validationImageLabelDs.shuffle(buffer_size=len(validationFilepaths)).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

trainXList = []
trainYList = []
for x, y in trainDs:
    trainXList.append(x)
    trainYList.append(y)

trainX = np.concatenate(trainXList, axis=0)
trainY = np.concatenate(trainYList, axis=0)

testXList = []
testYList = []
for x, y in trainDs:
    testXList.append(x)
    testYList.append(y)

testX = np.concatenate(testXList, axis=0)
testY = np.concatenate(testYList, axis=0)

validationXList = []
validationYList = []
for x, y in validationDs:
    validationXList.append(x)
    validationYList.append(y)

validationX = np.concatenate(validationXList, axis=0)
validationY = np.concatenate(validationYList, axis=0)

model = models.Sequential([
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(416, 416, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # 멀티라벨: binary_crossentropy, 멀티클래스: categorical_crossentropy
    metrics=['accuracy']
)
#
history = model.fit(
    trainX, trainY,
    epochs=10,
    batch_size=32,
    validation_data=(validationX, validationY)
)
model.save("my_model.h5")
