import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing import image
import numpy as np

train_dir = '/content/drive/MyDrive/APP DATA/maize/maize_disease/training_data'
test_dir = '/content/drive/MyDrive/APP DATA/maize/maize_disease/validation_data'

# Image dimensions
img_width, img_height = 150, 150

# Batch size for training
batch_size = 32

# Number of training epochs
num_epochs = 20

# Create data generators for training and testing data with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Create a CNN model
num_classes=7
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(512),
    Activation('relu'),
    Dropout(0.5),

    Dense(num_classes),
    Activation('softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=num_epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Accuracy:", test_accuracy)

# Make predictions on a new image with data augmentation
new_image_path = '/content/blight.jpg'
new_img = keras.preprocessing.image.load_img(
    new_image_path,
    target_size=(img_height, img_width)
)
new_img = keras.preprocessing.image.img_to_array(new_img)
new_img = np.expand_dims(new_img, axis=0)
new_img /= 255.0

predictions = model.predict(new_img)
class_index = np.argmax(predictions)

# Map class index to disease name (create a dictionary or list)
class_names = ['Blight', 'Gray Leaf Spot', 'Healthy','common rust','maize ear rot','maize fall armyworm','maize stem borer']  # Replace with your class names

predicted_disease = class_names[class_index]
print(f"Predicted disease: {predicted_disease}")

