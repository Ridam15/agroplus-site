# plant_disease_model.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import time

# Constants
IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 32
EPOCHS = 20
TRAIN_DATA_DIR = r'C:\Users\user\Downloads\dataset\train'
VALIDATION_DATA_DIR = r'C:\Users\user\Downloads\dataset\val'
MODEL_SAVE_PATH = r'C:\Users\user\Downloads\dataset\plant_disease_model.h5'

# Data Augmentation and Preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% of data for validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Define the Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu',
           input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model
start_time = time.time()  # Record the start time

history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    validation_data=val_data,
    validation_steps=val_data.samples // val_data.batch_size,
    epochs=EPOCHS
)

end_time = time.time()  # Record the end time
total_time = end_time - start_time  # Calculate total time
time_per_epoch = total_time / EPOCHS  # Calculate time per epoch

# Estimate remaining time
elapsed_epochs = len(history.history['accuracy'])
remaining_epochs = EPOCHS - elapsed_epochs
estimated_remaining_time = time_per_epoch * remaining_epochs

# Convert time to hours, minutes, and seconds
hours, remainder = divmod(estimated_remaining_time, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Total training time: {total_time // 60:.0f} minutes")
print(
    f"Estimated remaining time: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds")

# Save the Model
model.save(MODEL_SAVE_PATH)

# Plot Training History
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.tight_layout()
plt.show()
