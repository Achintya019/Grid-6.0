import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import easyocr
import cv2
from ultralytics import YOLO
import re
from datetime import datetime
import tempfile
import sqlite3

# Define class names for freshness detection
class_names = ['Apple', 'Banana', 'BitterGourd', 'Capsicum', 'Cucumber', 'Okra', 'Orange', 'Potato', 'Tomato',
               'Rotten Apple', 'Rotten Banana', 'Rotten BitterGourd', 'Rotten Capsicum', 'Rotten Cucumber', 'Rotten Okra', 'Rotten Orange', 'Rotten Potato', 'Rotten Tomato']

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
        .result {
            padding: 10px;
            border-radius: 8px;
            font-weight: bold;
            text-align: center;
        }
        .result-fresh {
            background-color: #C8E6C9;
            color: #2E7D32;
        }
        .result-rotten {
            background-color: #FFCDD2;
            color: #B71C1C;
        }
        .result-brand {
            background-color: #BBDEFB;
            color: #0D47A1;
        }
        </style>
    """, unsafe_allow_html=True)

# Initialize database
def init_db():
    conn = sqlite3.connect('results.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_type TEXT,
            result TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Save results to database
def save_to_db(task_type, result, confidence=None):
    conn = sqlite3.connect('results.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO results (task_type, result, confidence)
        VALUES (?, ?, ?)
    ''', (task_type, result, confidence))
    conn.commit()
    conn.close()

# View database table
def view_db():
    conn = sqlite3.connect('results.db')
    c = conn.cursor()
    c.execute('SELECT * FROM results')
    data = c.fetchall()
    conn.close()
    return data

# Delete database entries
def delete_db_entry(entry_id):
    conn = sqlite3.connect('results.db')
    c = conn.cursor()
    c.execute('DELETE FROM results WHERE id = ?', (entry_id,))
    conn.commit()
    conn.close()

# Load the TensorFlow model for freshness detection
@st.cache_resource
def load_freshness_model():
    return tf.keras.models.load_model('Freshness_Predicter/fruit_veg_classifier_custom.keras')

# Load the YOLO model for date extraction
@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)

# Freshness detection prediction function
def predict_freshness(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence_score = np.max(prediction)

    predicted_class_name = class_names[predicted_class_index]
    state = 'Fresh' if predicted_class_index < 9 else 'Rotten'
    freshness_index = (50 + (confidence_score * 50)) if state == 'Fresh' else (1 - confidence_score) * 50

    return predicted_class_name, state, freshness_index

# Date extraction utility functions
def find_all_dates(text):
    date_patterns = [
        r'\b\d{2}[/-]\d{2}[/-]\d{4}\b',
        r'\b\d{4}[/-]\d{2}[/-]\d{2}\b',
        r'\b\d{1,2}[ ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[ ]\d{4}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[ ]\d{1,2},[ ]\d{4}\b',
    ]
    regex = '|'.join(date_patterns)
    matches = re.findall(regex, text)
    return matches

def extract_dates(image_path, model):
    frame = cv2.imread(image_path)
    results = model(frame)

    reader = easyocr.Reader(['en'])
    date_strings = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_image = frame[y1:y2, x1:x2]
            text_results = reader.readtext(cropped_image)
            for _, text, _ in text_results:
                date_strings.extend(find_all_dates(text))

    return ', '.join(date_strings)

# Brand detection function
def detect_brands(image_path, model):
    frame = cv2.imread(image_path)
    results = model(frame)

    reader = easyocr.Reader(['en'])
    detected_brands = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cropped_image = frame[y1:y2, x1:x2]
            text_results = reader.readtext(cropped_image)
            for _, text, _ in text_results:
                detected_brands.append(text)

    annotated_image_path = '/tmp/annotated_brand_image.jpg'
    cv2.imwrite(annotated_image_path, frame)
    return detected_brands, annotated_image_path

# Video processing functions
def extract_dates_from_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    reader = easyocr.Reader(['en'])
    date_strings = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_image = frame[y1:y2, x1:x2]
                text_results = reader.readtext(cropped_image)
                for _, text, _ in text_results:
                    date_strings.extend(find_all_dates(text))
    cap.release()
    return ', '.join(date_strings)

def detect_brands_from_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    reader = easyocr.Reader(['en'])
    detected_brands = []

    annotated_video_path = '/tmp/annotated_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(annotated_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cropped_image = frame[y1:y2, x1:x2]
                text_results = reader.readtext(cropped_image)
                for _, text, _ in text_results:
                    detected_brands.append(text)
        out.write(frame)

    cap.release()
    out.release()
    return detected_brands, annotated_video_path

# Streamlit app interface
apply_custom_css()
init_db()
st.title('🛠️ Multi-Function Tool: Freshness Detector, Date Extractor & Brand Detector')

# Task selection
task_type = st.selectbox('Select a task:', ['Freshness Detection', 'Date Extraction', 'Brand Detection'])

# Input method selection
task = st.selectbox('Choose an input type:', ['Upload Image', 'Capture Image', 'Upload Video'])

if task == 'Upload Image':
    uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_path = "/tmp/temp_image.jpg"
        image = image.convert("RGB")
        image.save(image_path)
        st.image(image, caption='Uploaded Image', use_container_width=True)

elif task == 'Capture Image':
    captured_image = st.camera_input("📸 Capture an image")

    if captured_image is not None:
        image = Image.open(captured_image)
        image_path = "/tmp/captured_image.jpg"
        image = image.convert("RGB")
        image.save(image_path)
        st.image(image, caption='Captured Image', use_container_width=True)

elif task == 'Upload Video':
    uploaded_video = st.file_uploader("📁 Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        video_path = f"/tmp/{uploaded_video.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.video(video_path)


if st.button('Process'):
    if task_type == 'Freshness Detection' and 'image_path' in locals():
        model = load_freshness_model()
        predicted_class_name, state, freshness_index = predict_freshness(image_path, model)
        st.subheader(f"Prediction: {predicted_class_name} ({state})")
        st.write(f"Freshness Index: {freshness_index:.2f}%")
        save_to_db('Freshness Detection', f"{predicted_class_name} ({state})", freshness_index)

    elif task_type == 'Date Extraction' and 'image_path' in locals():
        model = load_yolo_model('Expiry_Date_Identifier/exp_date.pt')
        dates = extract_dates(image_path, model)
        st.subheader("Extracted Dates:")
        st.write(dates if dates else "No dates detected.")
        save_to_db('Date Extraction', dates)

    elif task_type == 'Brand Detection' and 'image_path' in locals():
        model = load_yolo_model('Brand_Identifier/model.pt')
        brands, annotated_image_path = detect_brands(image_path, model)
        st.subheader("Detected Brands:")
        st.write(', '.join(brands) if brands else "No brands detected.")
        st.image(annotated_image_path, caption="Annotated Image", use_container_width=True)
        save_to_db('Brand Detection', ', '.join(brands))

    elif task_type == 'Date Extraction' and 'video_path' in locals():
        model = load_yolo_model('Expiry_Date_Identifier/exp_date.pt')
        dates = extract_dates_from_video(video_path, model)
        st.subheader("Extracted Dates from Video:")
        st.write(dates if dates else "No dates detected in video.")
        save_to_db('Date Extraction (Video)', dates)

    elif task_type == 'Brand Detection' and 'video_path' in locals():
        model = load_yolo_model('Brand_Identifier/model.pt')
        brands, annotated_video_path = detect_brands_from_video(video_path, model)
        st.subheader("Detected Brands from Video:")
        st.write(', '.join(brands) if brands else "No brands detected in video.")
        st.video(annotated_video_path)
        save_to_db('Brand Detection (Video)', ', '.join(brands))

    else:
        st.error("Please upload or capture a valid input for the selected task.")

# View and manage database entries
if st.checkbox('View Database Entries'):
    data = view_db()
    st.subheader("Database Entries")
    if data:
        for entry in data:
            st.write(f"ID: {entry[0]}, Task: {entry[1]}, Result: {entry[2]}, Confidence: {entry[3]}, Timestamp: {entry[4]}")
            if st.button(f"Delete Entry {entry[0]}"):
                delete_db_entry(entry[0])
                st.success(f"Deleted entry ID {entry[0]}.")
    else:
        st.write("No entries found.")
