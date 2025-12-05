import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

# ------------------------------
# CONFIGURATION
# ------------------------------
img_size = 160
batch_size = 32
dataset_path = "/home/ritik/Desktop/ImageClassifier/animals"   # <<< UPDATE THIS

# ------------------------------
# DATA GENERATOR
# ------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    class_mode='categorical',
    batch_size=batch_size,
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    class_mode='categorical',
    batch_size=batch_size,
    subset='validation'
)

# ------------------------------
# SAVE CLASS NAMES
# ------------------------------
class_names = list(train_generator.class_indices.keys())
print("Class names:", class_names)

with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# ------------------------------
# TRANSFER LEARNING MODEL
# ------------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_size, img_size, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # freeze base model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------------------
# CALLBACKS (Recommended)
# ------------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "animal_model_best.h5",
        save_best_only=True,
        monitor="val_accuracy",
        mode="max"
    )
]

# ------------------------------
# TRAIN MODEL
# ------------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=12,
    callbacks=callbacks
)

# Final Save
model.save("animal_model.h5")

print("\nModel training finished and saved!")
