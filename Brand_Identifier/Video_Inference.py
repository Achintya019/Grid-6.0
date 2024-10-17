import easyocr
import cv2
from ultralytics import YOLO
#from google.colab.patches import cv2_imshow


model = YOLO('model.pt')



# Initialize the EasyOCR reader (English language)
reader = easyocr.Reader(['en'])

# Open the video file or capture device (use 0 for webcam)
cap = cv2.VideoCapture(0)  # Replace 'input_video.mp4' with 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Replace with your model inference to get bounding boxes

    # Loop over the detected bounding boxes
    for result in results:
        for box in result.boxes:
            # Convert box coordinates to integers
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw bounding boxes on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Crop the detected region
            cropped_image = frame[y1:y2, x1:x2]
            
            # Use EasyOCR to extract text from the cropped image
            text_results = reader.readtext(cropped_image)
            for text in text_results:
                print("Detected Text",text[1])
            
            # Display the extracted text on the frame
            for (bbox, text, _) in text_results:
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            

    # Display the frame with bounding boxes and text
    cv2.imshow('Video Inference',frame)



        # Press 'q' to exit the video loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()





    