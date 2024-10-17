import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import cv2

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

model.load_weights('Grid-6.0/fruit_veg_classifier.h5')

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
        freshness_index = (1 - confidence_score)*100  # Confidence score for fresh
        class_label = 'Rotten'
    else:
        freshness_index = 50 + (confidence_score*100)  # Confidence score for rotten
        class_label = 'Fresh'

    return class_label, round(freshness_index,4)

image_path = 'Grid-6.0/mango.jpeg'

label, confidence = predict_image(image_path)
print(f'Prediction: {label}, Freshness : {confidence} %')

img = cv2.imread(image_path)
cv2.putText(img, f"Freshness : {confidence} %", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
cv2.putText(img, label, (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow('Object Freshness',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
