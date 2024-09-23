import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Directory for the training dataset (which has two subfolders: Fresh and Rotten)
train_dir = 'final_dataset'

# ImageDataGenerator with a 70-30 split for training and testing
train_datagen = ImageDataGenerator(
    rescale=1./255,        # Normalize pixel values
    rotation_range=20,     # Random rotations
    width_shift_range=0.2, # Horizontal shift
    height_shift_range=0.2,# Vertical shift
    shear_range=0.2,       # Shearing
    zoom_range=0.2,        # Zooming
    horizontal_flip=True,  # Horizontal flip
    fill_mode='nearest',   # Fill missing pixels
    validation_split=0.3   # 30% of the data for testing
)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',    
    subset='training'       
)


test_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'     
)

#CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary output
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=test_generator,  # Use the 30% test data as validation
    validation_steps=test_generator.samples // test_generator.batch_size,
    epochs=15
)

# Save the model
model.save('fruit_veg_classifier.h5')

#Prediction Function
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array /= 255.0  

    prediction = model.predict(img_array)
    confidence_score = prediction[0][0]

    # Convert confidence score to freshness index
    if confidence_score >= 0.5:
        freshness_index = confidence_score  # Confidence score for fresh
        class_label = 'Fresh'
    else:
        freshness_index = 1 - confidence_score  # Confidence score for rotten
        class_label = 'Rotten'

    return class_label, freshness_index

# # Example usage
# image_path = ''
# label, confidence = predict_image(image_path)
# print(f'Prediction: {label}, Confidence Score: {confidence}')
