
import cv2
import keras

# Initialize image data generator with rescaling
train_data = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_data = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Preprocess all test images
train_gen = train_data.flow_from_directory(
        'train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# Preprocess all train images
validation_gen = validation_data.flow_from_directory(
        'test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# create model structure
emotions = keras.models.Sequential()

emotions.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotions.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotions.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
emotions.add(keras.layers.Dropout(0.25))

emotions.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotions.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
emotions.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotions.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
emotions.add(keras.layers.Dropout(0.25))

emotions.add(keras.layers.Flatten())
emotions.add(keras.layers.Flatten())
emotions.add(keras.layers.Dense(1024, activation='relu'))
emotions.add(keras.layers.Dropout(0.5))
emotions.add(keras.layers.Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

emotions.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
emotion_model = emotions.fit_generator(
        train_gen,
        steps_per_epoch=448,
        epochs=50,
        validation_data=validation_gen,
        validation_steps=112)

model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

emotions.save_weights('emotion_model.h5')

