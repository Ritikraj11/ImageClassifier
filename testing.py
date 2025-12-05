import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load model
model = tf.keras.models.load_model("animal_model_best.h5")

# Class names from training
class_names = [
    'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar',
    'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow',
    'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly',
    'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish',
    'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog',
    'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish',
    'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster',
    'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter',
    'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin',
    'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer',
    'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake',
    'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey',
    'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra'
]

# Your test image
image_path = "test.jpg"

# Load and preprocess image
img = image.load_img(image_path, target_size=(160,160))
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

# Predict
pred = model.predict(img)
predicted_index = np.argmax(pred[0])

print("Predicted index:", predicted_index)
print("Predicted class:", class_names[predicted_index])
print("Confidence:", np.max(pred[0]) * 100)
