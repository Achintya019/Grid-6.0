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
def predict_image(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize the image to (150, 150)
    img_resized = cv2.resize(img_rgb, (150, 150))
    
    # Convert to array and expand dimensions
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, axis=0)
    
    # Normalize the image
    img_array /= 255.0  

    # Make a prediction
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


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    label, confidence = predict_image(frame)
    print(f'Prediction: {label}, Freshness: {confidence} %')
    cv2.putText(frame, f"Freshness : {confidence} % ("+label+")", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video Inference',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
