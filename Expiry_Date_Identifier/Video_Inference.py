import easyocr
import cv2
from ultralytics import YOLO
#from google.colab.patches import cv2_imshow
import re
from datetime import datetime

def find_all_dates(text):
    date_patterns = [
        r'\b\d{2}[/-]\d{2}[/-]\d{4}\b',  # Matches DD/MM/YYYY, DD-MM-YYYY
        r'\b\d{4}[/-]\d{2}[/-]\d{2}\b',  # Matches YYYY/MM/DD, YYYY-MM-DD
        r'\b\d{1,2}[ ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[ ]\d{4}\b',  # Matches DD Month YYYY (e.g., 15 Jan 2024)
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[ ]\d{1,2},[ ]\d{4}\b',  # Matches Month DD, YYYY (e.g., January 15, 2024)
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[ ]\d{4}\b',  # Matches Month YYYY (e.g., January 2024)
        r'\b\d{4}[ ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b',  # Matches YYYY Month (e.g., 2024 January)
        r'\b\d{2}[/-]\d{4}\b',  # Matches MM/YYYY or MM-YYYY
        r'\b\d{4}\b'  # Matches only the year (e.g., 2024)
    ]
    
    regex = '|'.join(date_patterns)
    
    matches = re.findall(regex, text)
    
    def parse_date(date_str):
        for fmt in [
            '%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y-%m-%d',  
            '%d %b %Y', '%d %B %Y', '%b %d, %Y', '%B %d, %Y',  
            '%b %Y', '%B %Y', 
            '%Y %b', '%Y %B', 
            '%m/%Y', '%m-%Y',  
            '%Y'  
        ]:
            try:
                dt = datetime.strptime(date_str, fmt)
                if '%d' not in fmt:  
                    dt = dt.replace(day=1)  
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                pass
        return None
        
    date_strings = [parse_date(date) for date in matches if parse_date(date) is not None]
    return ' '.join(date_strings)


model = YOLO('exp_date.pt')



# Initialize the EasyOCR reader (English language)
reader = easyocr.Reader(['en'])

# Open the video file or capture device (use 0 for webcam)
cap = cv2.VideoCapture(0)  # Replace 'input_video.mp4' with 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Replace with your model inference to get bounding boxes

    tempVar = None
    # Loop over the detected bounding boxes
    for result in results:
        for box in result.boxes:
            # Convert box coordinates to integers
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw bounding boxes on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Crop the detected region
            cropped_image = frame[y1:y2, x1:x2]

            
            emptystring=""
            
            # Use EasyOCR to extract text from the cropped image
            text_results = reader.readtext(cropped_image)
            for text in text_results:
                emptystring= emptystring + " " + text[1]

            displayString = find_all_dates(emptystring)
            if(displayString==None):
                displayString = tempVar
            else:
                tempVar = displayString
            print(f"Date : {displayString}")
            # Display the extracted text on the frame
            for (bbox, text, _) in text_results:
                cv2.putText(frame, displayString, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
               

    # Display the frame with bounding boxes and text
    cv2.imshow('Video Inference',frame)



        # Press 'q' to exit the video loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()





    
