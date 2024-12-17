import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import easyocr
import cv2
from ultralytics import YOLO
import re
from datetime import datetime
import sqlite3
import tempfile
import pandas as pd

# Define class names for freshness detection
class_names = ['Apple', 'Banana', 'BitterGourd', 'Capsicum', 'Cucumber', 'Okra', 'Orange', 'Potato', 'Tomato',
               'Rotten Apple', 'Rotten Banana', 'Rotten BitterGourd', 'Rotten Capsicum', 'Rotten Cucumber',
               'Rotten Okra', 'Rotten Orange', 'Rotten Potato', 'Rotten Tomato']

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
    # Connect to a SQLite database named 'results.db'. 
    # If the database doesn't exist, it will be created.
    conn = sqlite3.connect('results.db')
    
    # Create a cursor object to interact with the database.
    c = conn.cursor()
    
    # Execute a SQL command to create a table named 'results' if it doesn't already exist.
    # The table has the following columns:
    # - id: An auto-incremented primary key.
    # - task_type: A text field to store the type of task.
    # - result: A text field to store the result of the task.
    # - confidence: A real (float) field to store the confidence level of the result.
    # - timestamp: A datetime field that stores the record creation time, with a default value of the current timestamp.
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_type TEXT,
            result TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Save the changes to the database.
    conn.commit()
    
    # Close the connection to the database.
    conn.close()

# Save results to database
def save_to_db(task_type, result, confidence=None):
    # Connect to the SQLite database named 'results.db'.
    # If the database doesn't exist, it will be created (although the table is expected to exist already).
    conn = sqlite3.connect('results.db')
    
    # Create a cursor object to interact with the database.
    c = conn.cursor()
    
    # Insert a new record into the 'results' table with the provided values:
    # - task_type: Type of the task (stored as text).
    # - result: Result of the task (stored as text).
    # - confidence: Confidence level of the result (optional, stored as a real/float value).
    # The values are safely parameterized using placeholders (?, ?, ?) to prevent SQL injection.
    c.execute(
        'INSERT INTO results (task_type, result, confidence) VALUES (?, ?, ?)', 
        (task_type, result, confidence)
    )
    
    # Commit the transaction to save the changes to the database.
    conn.commit()
    
    # Close the connection to the database to free up resources.
    conn.close()

# View database table
def view_db():
    # Connect to the SQLite database named 'results.db'.
    conn = sqlite3.connect('results.db')
    
    # Create a cursor object to interact with the database.
    c = conn.cursor()
    
    # Execute a SQL query to retrieve all records from the 'results' table.
    c.execute('SELECT * FROM results')
    
    # Fetch all rows of the query result.
    data = c.fetchall()
    
    # Close the connection to the database to free up resources.
    conn.close()
    
    # Return the retrieved data to the caller.
    return data


# Function to delete entries from the database.
def delete_db_entry(entry_id=None):
    # Connect to the SQLite database named 'results.db'.
    conn = sqlite3.connect('results.db')
    
    # Create a cursor object to interact with the database.
    c = conn.cursor()
    
    # Check if a specific entry ID is provided.
    if entry_id:
        # Delete the record with the specified ID from the 'results' table.
        c.execute('DELETE FROM results WHERE id = ?', (entry_id,))
    else:
        # If no ID is provided, delete all records from the 'results' table.
        c.execute('DELETE FROM results')
    
    # Commit the transaction to save the changes to the database.
    conn.commit()
    
    # Close the connection to the database to free up resources.
    conn.close()


# Load TensorFlow model
# Cache the resource to avoid reloading the model multiple times during app execution.
@st.cache_resource
def load_freshness_model():
    # Load and return the TensorFlow Keras model from the specified path.
    # This model is used for classifying the freshness of fruits and vegetables.
    return tf.keras.models.load_model('/content/fruit_veg_classifier_custom.keras')


# Cache the resource to avoid reloading the YOLO model multiple times during app execution.
@st.cache_resource
def load_yolo_model(model_path):
    # Load and return the YOLO model using the specified path.
    # This model is likely used for object detection tasks.
    return YOLO(model_path)


# Function to predict the freshness of a fruit or vegetable from an image.
def predict_freshness(image_path, model):
    # Load the image from the specified path and resize it to 150x150 pixels,
    # as expected by the freshness classification model.
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    
    # Convert the image to a numpy array format.
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Expand the array's dimensions to match the model's expected input shape (batch size, height, width, channels).
    # Normalize pixel values to the range [0, 1] by dividing by 255.0.
    img_array = tf.expand_dims(img_array, axis=0) / 255.0

    # Use the loaded model to predict the class probabilities for the input image.
    prediction = model.predict(img_array)
    
    # Find the index of the class with the highest predicted probability.
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    
    # Extract the highest confidence score from the prediction.
    confidence_score = np.max(prediction)

    # Map the predicted class index to its corresponding class name.
    predicted_class_name = class_names[predicted_class_index]
    
    # Determine if the item is 'Fresh' or 'Rotten' based on the predicted class index.
    # Assuming indices < 9 represent 'Fresh' and indices >= 9 represent 'Rotten'.
    state = 'Fresh' if predicted_class_index < 9 else 'Rotten'
    
    # Calculate the freshness index:
    # - If the item is 'Fresh', the index is proportional to the confidence score.
    # - If the item is 'Rotten', the index is inversely proportional to the confidence score.
    freshness_index = confidence_score * 100 if state == 'Fresh' else (1 - confidence_score) * 100

    # Return the predicted class name, the item's state ('Fresh' or 'Rotten'), and the freshness index.
    return predicted_class_name, state, freshness_index


def find_all_dates(text):
    # Define a list of regular expression patterns to match various date formats.
    date_patterns = [
        # Pattern 1: Matches dates in the format DD/MM/YYYY or DD-MM-YYYY.
        r'\b\d{2}[/-]\d{2}[/-]\d{4}\b',
        
        # Pattern 2: Matches dates in the format YYYY/MM/DD or YYYY-MM-DD.
        r'\b\d{4}[/-]\d{2}[/-]\d{2}\b',
        
        # Pattern 3: Matches dates in the format "D Month YYYY" or "DD Month YYYY",
        # where the month can be abbreviated or fully spelled (e.g., 1 Jan 2024, 01 January 2024).
        r'\b\d{1,2}[ ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[ ]\d{4}\b',
        
        # Pattern 4: Matches dates in the format "Month D, YYYY" or "Month DD, YYYY",
        # where the month can be abbreviated or fully spelled (e.g., January 1, 2024, Jan 01, 2024).
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[ ]\d{1,2},[ ]\d{4}\b',
    ]
    
    # Combine all date patterns into a single regular expression using '|' (OR operator),
    # and search for all matches in the input text.
    matches = re.findall('|'.join(date_patterns), text)
    
    # Return the list of all matched dates found in the input text.
    return matches

def extract_dates(image_path, model):
    # Load the image from the given file path using OpenCV.
    frame = cv2.imread(image_path)
    
    # Pass the image to the object detection model (e.g., YOLO) to detect regions of interest (ROIs).
    results = model(frame)

    # Initialize the EasyOCR reader for extracting text from images.
    # The 'en' parameter specifies that the OCR model should use the English language.
    reader = easyocr.Reader(['en'])
    
    # Initialize an empty list to store extracted date strings.
    date_strings = []

    # Iterate through the detection results returned by the object detection model.
    for result in results:
        # Loop through each detected bounding box in the result.
        for box in result.boxes:
            # Extract the coordinates of the bounding box (x1, y1, x2, y2).
            # Convert them to integers to properly index into the image array.
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Crop the image using the bounding box coordinates to focus on the ROI.
            cropped_image = frame[y1:y2, x1:x2]
            
            # Use EasyOCR to read text from the cropped image.
            text_results = reader.readtext(cropped_image)
            
            # Iterate through the OCR results for the cropped image.
            # Each result contains the bounding box, detected text, and confidence score.
            for _, text, _ in text_results:
                # Use the 'find_all_dates' function to extract date strings from the detected text.
                # Add the extracted date strings to the 'date_strings' list.
                date_strings.extend(find_all_dates(text))

    # Join the extracted date strings into a single comma-separated string for output.
    return ', '.join(date_strings)


def detect_brands(image_path, model):
    # Load the input image from the specified file path using OpenCV.
    frame = cv2.imread(image_path)
    
    # Use the object detection model (e.g., YOLO) to identify regions of interest (ROIs) in the image.
    results = model(frame)

    # Initialize the EasyOCR reader for text detection.
    # The 'en' parameter specifies that the OCR model should use the English language.
    reader = easyocr.Reader(['en'])
    
    # Initialize an empty list to store detected brand names.
    detected_brands = []

    # Iterate through the detection results returned by the object detection model.
    for result in results:
        # Loop through each detected bounding box in the result.
        for box in result.boxes:
            # Extract the coordinates of the bounding box (x1, y1, x2, y2).
            # Convert them to integers to properly index into the image array.
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Crop the image using the bounding box coordinates to isolate the ROI.
            cropped_image = frame[y1:y2, x1:x2]
            
            # Use EasyOCR to extract text from the cropped image.
            text_results = reader.readtext(cropped_image)
            
            # Iterate through the OCR results for the cropped image.
            # Each result contains the bounding box, detected text, and confidence score.
            for _, text, _ in text_results:
                # Append the detected text (potential brand name) to the list of detected brands.
                detected_brands.append(text)

    # Save the annotated image to a temporary file path for visualization or further use.
    annotated_image_path = '/tmp/annotated_brand_image.jpg'
    cv2.imwrite(annotated_image_path, frame)
    
    # Return the list of detected brand names and the path to the annotated image.
    return detected_brands, annotated_image_path


def extract_dates_from_video(video_path, model):
    # Open the video file using OpenCV's VideoCapture.
    cap = cv2.VideoCapture(video_path)
    
    # Initialize the EasyOCR reader for text detection.
    # The 'en' parameter specifies that the OCR model should use the English language.
    reader = easyocr.Reader(['en'])
    
    # Initialize an empty list to store extracted date strings.
    date_strings = []

    # Loop through the video frames until the end of the video.
    while cap.isOpened():
        # Read the next frame from the video.
        ret, frame = cap.read()
        if not ret:  # Exit the loop if there are no more frames.
            break

        # Use the object detection model (e.g., YOLO) to detect regions of interest (ROIs) in the frame.
        results = model(frame)
        
        # Iterate through the detection results returned by the model.
        for result in results:
            # Loop through each detected bounding box in the result.
            for box in result.boxes:
                # Extract the coordinates of the bounding box (x1, y1, x2, y2).
                # Convert them to integers to properly index into the image array.
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Crop the frame using the bounding box coordinates to isolate the ROI.
                cropped_image = frame[y1:y2, x1:x2]
                
                # Use EasyOCR to read text from the cropped image.
                text_results = reader.readtext(cropped_image)
                
                # Iterate through the OCR results for the cropped image.
                # Each result contains the bounding box, detected text, and confidence score.
                for _, text, _ in text_results:
                    # Use the 'find_all_dates' function to extract date strings from the detected text.
                    # Add the extracted date strings to the 'date_strings' list.
                    date_strings.extend(find_all_dates(text))

    # Release the video capture object to free resources.
    cap.release()
    
    # Return the extracted dates as a single comma-separated string.
    return ', '.join(date_strings)


def detect_brands_from_video(video_path, model):
    # Open the video file using OpenCV's VideoCapture.
    cap = cv2.VideoCapture(video_path)
    
    # Initialize the EasyOCR reader for text detection.
    # The 'en' parameter specifies that the OCR model should use the English language.
    reader = easyocr.Reader(['en'])
    
    # Initialize an empty list to store detected brand names.
    detected_brands = []

    # Define the output path for the annotated video.
    annotated_video_path = '/tmp/annotated_video.mp4'
    
    # Define the codec and create a VideoWriter object to save the annotated video.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(annotated_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # Loop through the video frames until the end of the video.
    while cap.isOpened():
        # Read the next frame from the video.
        ret, frame = cap.read()
        if not ret:  # Exit the loop if there are no more frames.
            break

        # Use the object detection model (e.g., YOLO) to detect regions of interest (ROIs) in the frame.
        results = model(frame)
        
        # Iterate through the detection results returned by the model.
        for result in results:
            # Loop through each detected bounding box in the result.
            for box in result.boxes:
                # Extract the coordinates of the bounding box (x1, y1, x2, y2).
                # Convert them to integers to properly index into the image array.
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw a rectangle around the detected ROI on the frame.
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Crop the frame using the bounding box coordinates to isolate the ROI.
                cropped_image = frame[y1:y2, x1:x2]
                
                # Use EasyOCR to read text from the cropped image.
                text_results = reader.readtext(cropped_image)
                
                # Iterate through the OCR results for the cropped image.
                # Each result contains the bounding box, detected text, and confidence score.
                for _, text, _ in text_results:
                    # Append the detected text (potential brand name) to the list of detected brands.
                    detected_brands.append(text)
        
        # Write the annotated frame to the output video file.
        out.write(frame)

    # Release the video capture and writer objects to free resources.
    cap.release()
    out.release()
    
    # Return the list of detected brand names and the path to the annotated video.
    return detected_brands, annotated_video_path


# Streamlit app interface
# Custom CSS application function (implementation assumed elsewhere)
apply_custom_css()  # Apply custom styles to the Streamlit app

# Initialize the database (implementation assumed elsewhere)
init_db()  # Initialize the database for task-related operations

# App title
st.title('🛠️ Multi-Function Tool: Freshness Detector, Date Extractor & Brand Detector')

# --- TASK SELECTION ---
# Dropdown to select the type of task to perform
# Options include 'Freshness Detection', 'Date Extraction', and 'Brand Detection'
task_type = st.selectbox('Select a task:', ['Freshness Detection', 'Date Extraction', 'Brand Detection'])

# Dropdown to choose the input type (image or video)
# Users can upload an image, capture an image via the camera, or upload a video
task = st.selectbox('Choose an input type:', ['Upload Image', 'Capture Image', 'Upload Video'])

# --- UPLOAD IMAGE TASK ---
if task == 'Upload Image':
    # File uploader widget for image input
    uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

    # Process the uploaded image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)  # Open the image using PIL
        image_path = "/tmp/temp_image.jpg"  # Temporary path to save the image
        image = image.convert("RGB")  # Ensure image is in RGB format
        image.save(image_path)  # Save the image to the temporary path

        # Display the uploaded image with a caption
        st.image(image, caption='Uploaded Image', use_container_width=True)

# --- CAPTURE IMAGE TASK ---
elif task == 'Capture Image':
    # Camera input widget to capture an image using the webcam
    captured_image = st.camera_input("📸 Capture an image")

    # Process the captured image
    if captured_image is not None:
        image = Image.open(captured_image)  # Open the captured image
        image_path = "/tmp/captured_image.jpg"  # Temporary path to save the captured image
        image = image.convert("RGB")  # Convert to RGB format
        image.save(image_path)  # Save the image to the temporary path

        # Display the captured image with a caption
        st.image(image, caption='Captured Image', use_container_width=True)

# --- UPLOAD VIDEO TASK ---
elif task == 'Upload Video':
    # File uploader widget for video input
    uploaded_video = st.file_uploader("📁 Upload a video", type=["mp4", "avi", "mov"])

    # Process the uploaded video
    if uploaded_video is not None:
        # Define a path to save the uploaded video
        video_path = f"/tmp/{uploaded_video.name}"

        # Save the uploaded video to the defined path
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        # Display the uploaded video in the Streamlit app
        st.video(video_path)


# --- PROCESS TASK BUTTON ---
if st.button('Process'):
    # --- FRESHNESS DETECTION TASK ---
    if task_type == 'Freshness Detection' and 'image_path' in locals():
        model = load_freshness_model()  # Load the model for freshness detection
        predicted_class_name, state, freshness_index = predict_freshness(image_path, model)  # Predict freshness
        
        # Display the prediction results
        st.subheader(f"Prediction: {predicted_class_name} ({state})")
        st.write(f"Freshness Index: {freshness_index:.2f}%")
        save_to_db('Freshness Detection', f"{predicted_class_name} ({state})", freshness_index)  # Save to DB

    # --- DATE EXTRACTION FROM IMAGE TASK ---
    elif task_type == 'Date Extraction' and 'image_path' in locals():
        model = load_yolo_model('/content/exp_date.pt')  # Load the YOLO model for date extraction
        dates = extract_dates(image_path, model)  # Extract dates from the image
        
        # Display the extracted dates
        st.subheader("Extracted Dates:")
        st.write(dates if dates else "No dates detected.")
        save_to_db('Date Extraction', dates)  # Save to DB

    # --- BRAND DETECTION FROM IMAGE TASK ---
    elif task_type == 'Brand Detection' and 'image_path' in locals():
        model = load_yolo_model('/content/model.pt')  # Load the YOLO model for brand detection
        brands, annotated_image_path = detect_brands(image_path, model)  # Detect brands in the image
        
        # Display detected brands and annotated image
        st.subheader("Detected Brands:")
        st.write(', '.join(brands) if brands else "No brands detected.")
        st.image(annotated_image_path, caption="Annotated Image", use_container_width=True)
        save_to_db('Brand Detection', ', '.join(brands))  # Save to DB

    # --- DATE EXTRACTION FROM VIDEO TASK ---
    elif task_type == 'Date Extraction' and 'video_path' in locals():
        model = load_yolo_model('/content/exp_date.pt')  # Load the YOLO model for video date extraction
        dates = extract_dates_from_video(video_path, model)  # Extract dates from the video
        
        # Display the extracted dates from video
        st.subheader("Extracted Dates from Video:")
        st.write(dates if dates else "No dates detected in video.")
        save_to_db('Date Extraction (Video)', dates)  # Save to DB

    # --- BRAND DETECTION FROM VIDEO TASK ---
    elif task_type == 'Brand Detection' and 'video_path' in locals():
        model = load_yolo_model('/content/model.pt')  # Load the YOLO model for brand detection in video
        brands, annotated_video_path = detect_brands_from_video(video_path, model)  # Detect brands in the video
        
        # Display detected brands and annotated video
        st.subheader("Detected Brands from Video:")
        st.write(', '.join(brands) if brands else "No brands detected in video.")
        st.video(annotated_video_path)
        save_to_db('Brand Detection (Video)', ', '.join(brands))  # Save to DB

    # --- ERROR HANDLING FOR MISSING INPUT ---
    else:
        st.error("Please upload or capture a valid input for the selected task.")

# View and manage database entries
if st.checkbox('View Database Entries'):  # Provides a toggle option to view database content
    data = view_db()  # Function to retrieve all entries from the database
    
    if data:  # Check if there is any data in the database
        st.subheader("Database Entries")  # Section title for displaying entries
        
        # Convert the data to a pandas DataFrame for easier visualization and manipulation
        df = pd.DataFrame(data, columns=['ID', 'Task', 'Result', 'Confidence', 'Timestamp'])
        st.dataframe(df, hide_index=True)  # Display the table without the default index

        # Button to delete all entries in the database
        if st.button('Delete All Entries'):  
            conn = sqlite3.connect('results.db')  # Connect to the database
            c = conn.cursor()  # Create a cursor object to execute SQL commands
            c.execute('DELETE FROM results')  # SQL command to delete all rows in the 'results' table
            conn.commit()  # Commit the changes to the database
            conn.close()  # Close the database connection
            st.success("All entries deleted.")  # Confirm deletion to the user

        # Section for deleting individual entries
        st.subheader("Delete Individual Entries")  
        # Dropdown menu to select an ID from existing entries for deletion
        entry_id_to_delete = st.selectbox('Select an entry ID to delete:', df['ID'].tolist())
        
        # Button to delete the selected entry
        if st.button('Delete Selected Entry'):
            delete_db_entry(entry_id_to_delete)  # Function to delete a specific entry based on its ID
            st.success(f"Deleted entry ID {entry_id_to_delete}.")  # Confirm deletion to the user
            
    else:
        # Message displayed if the database is empty
        st.write("No entries found.")  

