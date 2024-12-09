# %%
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import RootMeanSquaredError

# # Step 1: Rebuild the architecture
# base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# base_model.trainable = True  # or set to False if you want to freeze it

# input_layer = layers.Input(shape=(224, 224, 3))
# x = base_model(input_layer, training=True)  # Ensure compatibility
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dense(32, activation='relu')(x)
# x = layers.Dense(32, activation='relu')(x)
# output = layers.Dense(1)(x)

# # Recreate the model
# model = models.Model(inputs=input_layer, outputs=output)

# # Compile the model
# model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(delta=1.5), metrics=[RootMeanSquaredError()])

# # Step 2: Load weights from the existing `.h5` file
# try:
#     model.load_weights('/content/Mark-V(final).h5')
#     print("Weights loaded successfully.")
# except Exception as e:
#     print(f"Error loading weights: {e}")

# # Step 3: Save the model again
# model.save('MARK5.h5')
# print("Model has been successfully re-saved as 'MARK5.h5'")


# # %%
# pip install streamlit

# # %%
# pip install pillow

# %%
# %%writefile app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2

# Define class names for your model's output
class_names = ['Apple', 'Banana', 'BitterGourd', 'Capsicum', 'Cucumber', 'Okra', 'Orange', 'Potato', 'Tomato', 'Apple',
               'Banana', 'BitterGourd', 'Capsicum', 'Cucumber', 'Okra', 'Orange', 'Potato', 'Tomato']

# Custom CSS for styling
def apply_custom_css():
    st.markdown("""
        <style>
        .main {
            background-color: #f0f8ff;
            padding: 20px;
            border-radius: 15px;
        }
        h1 {
            color: #333;
            font-family: 'Arial', sans-serif;
        }
        .upload-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            background-color: #ffffff;
            border: 1px solid #dddddd;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
        }
        .result-fresh {
            background-color: #C8E6C9;
            padding: 10px;
            border-radius: 8px;
            font-weight: bold;
            color: #2E7D32;
            text-align: center;
        }
        .result-rotten {
            background-color: #FFCDD2;
            padding: 10px;
            border-radius: 8px;
            font-weight: bold;
            color: #B71C1C;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('/home/aryam/Projects/Grid-6.0/fruit_veg_classifier_custom.keras')  # Update file path to .keras format
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
apply_custom_css()

# Define the prediction function
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence_score = np.max(prediction)

    predicted_class_name = class_names[predicted_class_index]

    state = 'Fresh'
    freshness_index = confidence_score * 100
    if predicted_class_index >= 9:  # Assuming 'Rotten' classes are indexed >= 9
        state = 'Rotten'
        freshness_index = (1 - confidence_score) * 100

    return predicted_class_name, state, freshness_index

# App interface
st.title('🍎 Fruit & Vegetable Freshness Detector')
st.markdown('<div class="upload-container">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload a fruit or vegetable image", type=["jpg", "jpeg", "png"])

# Camera input only when the button is clicked
captured_image = None
if "camera_clicked" not in st.session_state:
    st.session_state.camera_clicked = False

if uploaded_file is None:
    if st.button("📷 Take a picture"):
        st.session_state.camera_clicked = True

    if st.session_state.camera_clicked:
        captured_image = st.camera_input("Or, take a picture using the camera")
        if captured_image is not None:
            st.session_state.camera_clicked = False

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='🖼️ Your Uploaded Image', use_column_width=True)
elif captured_image is not None:
    image = Image.open(io.BytesIO(captured_image.getvalue()))

if image is not None:
    image_path = "/tmp/temp_image.jpg"
    image.save(image_path)

    # Predict freshness
    with st.spinner("🔍 Analyzing the image... Please wait"):
        if model:
            try:
                name, state, confidence = predict_image(image_path)

                # Display results
                if state == 'Fresh':
                    st.markdown(f'<div class="result-fresh">🍏 The {name} is Fresh! Freshness: {confidence:.2f}%</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-rotten">🍂 The {name} is Rotten. Freshness: {confidence:.2f}%</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

st.markdown('</div>', unsafe_allow_html=True)





# # %%
# !pip install streamlit

# # %%
# !pip install pyngrok

# # %%
# !ngrok authtoken 2nva2klX0kqiSYSLs9Hf5afOfzT_5HfyQx3B2bKqvTR1c484g

# # %%
# !nohup streamlit run app.py &

# # %%
# from pyngrok import ngrok
# url=ngrok.connect(8501)
# url

# # %% [markdown]
# # # to kill ngrok website

# # %%
# !pkill ngrok


# # %%



