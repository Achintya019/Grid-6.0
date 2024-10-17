import easyocr
import cv2
from ultralytics import YOLO

model = YOLO('exp_date.pt')
image_path = 'image1.jpeg'
frame = cv2.imread(image_path)

results = model(frame)

# Initialize the EasyOCR reader (English language)
reader = easyocr.Reader(['en'])

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
            print("Expiry Date is:",text[1])
        
        # Display the extracted text on the frame
        for (bbox, text, _) in text_results:
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    
cv2.imshow('Image Inference',frame)

cv2.waitKey(0)
cv2.destroyAllWindows()