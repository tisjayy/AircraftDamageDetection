import os
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.applications import VGG16
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parameters
batch_size = 32
n_epochs = 5
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)

# Paths
train_dir = r"C:\Users\2594j\Desktop\aircraftproject\dataset\train"
valid_dir = r"C:\Users\2594j\Desktop\aircraftproject\dataset\valid"
test_dir  = r"C:\Users\2594j\Desktop\aircraftproject\dataset\test"

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

# Load base model VGG16 without top
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
output = base_model.layers[-1].output
output = Flatten()(output)
base_model = Model(base_model.input, output)

# Freeze VGG16 base layers
for layer in base_model.layers:
    layer.trainable = False

# Build full model
model = Sequential()
model.add(base_model)
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# Train
history = model.fit(
    train_generator,
    epochs=n_epochs,
    validation_data=valid_generator
)

# Save model weights
model.save_weights('aircraft_model.weights.h5')
print("Model weights saved to aircraft_model.weights.h5")

