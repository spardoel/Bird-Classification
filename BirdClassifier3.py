import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.callbacks import TensorBoard,EarlyStopping
from keras.layers import Dropout



import pathlib

# define the directories for the train, test and validation datsets
train_dir = pathlib.Path('D:/Documents/Bird classification project/SubSet/train/')
test_dir =  pathlib.Path('D:/Documents/Bird classification project/SubSet/test2/')
valid_dir = pathlib.Path('D:/Documents/Bird classification project/SubSet/valid/')

image_count = len(list(train_dir.glob('*/*.jpg')))
#print(image_count)

#ANIANIAU = list(train_dir.glob('ANIANIAU/*'))
#img = PIL.Image.open(str(ANIANIAU[0])).show()

# Define parameters
BATCH_SIZE = 64
IMG_HEIGHT = 160
IMG_WIDTH = 160
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)


# create training dataset object
train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)
# Create the validation dataset
valid_ds = tf.keras.utils.image_dataset_from_directory(
  valid_dir,
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)
#
test_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  image_size=(IMG_HEIGHT, IMG_WIDTH))

class_names = train_ds.class_names
#print(class_names)

# Print the shape of the data
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# Configure the datasets for speed
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE).cache()
valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE).cache()

normalization_layer = layers.Rescaling(1./255)
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

# for image, _ in train_ds.take(1):
#   plt.figure(figsize=(10, 10))
#   first_image = image[0]
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
#     plt.imshow(augmented_image[0] / 255)
#     plt.axis('off')
#   plt.show()

# preprocessing to rescale the pixel values to [-1,1] for the pretrained model
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


#normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
#image_batch, labels_batch = next(iter(normalized_ds))
#first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
#print(np.min(first_image), np.max(first_image))


num_classes = len(class_names)

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

#base_model.summary()
image_batch, label_batch = next(iter(train_ds))
feature_batch = base_model(image_batch)
#print(f"Feature batch shape {feature_batch.shape}")

# truncate the model to remove the prediction layer
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
#print(f"Feature batch average shape{feature_batch_average.shape}")

#prediction_layer = tf.keras.layers.Dense(num_classes)
prediction_layer = tf.keras.Sequential([
  layers.Dense(64, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(64, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(num_classes)
])

prediction_batch = prediction_layer(feature_batch_average)

#print(f"Prediction batch shape {prediction_batch.shape}")


inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

initial_epochs = 20

loss0, accuracy0 = model.evaluate(valid_ds)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_ds,
                    epochs=initial_epochs,
                    validation_data=valid_ds)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()



y_pred = model.predict(test_ds)

score = tf.nn.softmax(y_pred)

for img_score in score:
  print(
      f"This image most likely belongs to {class_names[np.argmax(img_score)]} with a {100 * np.max(img_score):.2f} percent confidence."
  )

# https://www.tensorflow.org/tutorials/images/transfer_learning

model = Sequential([
  layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
  layers.RandomFlip('horizontal_and_vertical'),
  layers.RandomRotation(0.2),
  layers.Conv2D(8, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.Dropout(0.5),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.Dropout(0.5),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.Dropout(0.5),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

logdir = os.path.join("logs", "BC3_logs")
tensorboard = TensorBoard(log_dir=logdir)
early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=30)

epochs=60
model.fit(
  train_ds,
  validation_data=valid_ds,
  epochs=epochs,
  callbacks=[tensorboard,early_stop]
)

loss = pd.DataFrame(model.history.history)
plt.figure()
plt.plot(loss)
plt.show()

y_pred = model.predict(test_ds)

score = tf.nn.softmax(y_pred)

for img_score in score:
  print(
      f"This image most likely belongs to {class_names[np.argmax(img_score)]} with a {100 * np.max(img_score):.2f} percent confidence."
  )
