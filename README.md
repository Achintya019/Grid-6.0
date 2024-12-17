# Website UI
![image](https://github.com/user-attachments/assets/d9a8363e-e285-4a0a-a2cc-830495a7d30d)

## Users can choose a task they wish to perfom 
![image](https://github.com/user-attachments/assets/d5425ab7-b937-4b67-83de-31634a56a901)


## Users can choose upload a picture/video or capture an image with capture image feature
![image](https://github.com/user-attachments/assets/1de9f9a0-0b78-4eb6-af23-5f8ca7a398d9)


# Sample Interaction with the interface
![Screenshot 2024-12-12 180725](https://github.com/user-attachments/assets/19adae73-57ee-4df0-b885-a5c71b7e4752)
![image](https://github.com/user-attachments/assets/e3e742de-226b-4537-be48-0474e24c884c)


## Database Entries 
![image](https://github.com/user-attachments/assets/ef0636a0-4330-4669-9ac1-5447518ce0e6)


### Schema
            CREATE TABLE IF NOT EXISTS results 
            (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_type TEXT,
            result TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )


# Brand Identifier Dataset Structure
* Download the training and validation dataset from [Google Drive Link](https://drive.google.com/file/d/11JR2Fvedr9-CtLX6JB_6t4a8S_HSsUfP/view?usp=drive_link)
```
BRAND DETECTION DATASET/        # Root directory for the dataset
├── BRAND_DETECTION/            # Main dataset folder
│   ├── BRAND_DATASET/          # Subfolder containing the actual dataset
│   │   ├── train/              # Training data
│   │       ├── images/         # Folder for images
│   │       └── label/          # Folder for corresponding labels
│   │   ├── valid/              # Validation data
│   │       ├── images/         # Folder for images
│   │       └── label/          # Folder for corresponding labels
│   │   └── test/               # Test data
│   │       ├── images/         # Folder for images
│   │       └── label/          # Folder for corresponding labels
└── Data.yaml                   # Yaml file for the dataset
```
## Inference
![image](https://github.com/user-attachments/assets/84917543-98c3-4248-a8de-40970772d513)
![image](https://github.com/user-attachments/assets/949a8235-898c-45e2-a637-c86a23395958)


# Expiry Date Detection Dataset Structure
* Download the training and validation dataset from [Google Drive Link](https://drive.google.com/file/d/1pgiL1aY-hD70cq_Cum1_w3-JJ3xJX7fy/view?usp=drive_link)
```
EXPIRY DATE DETECTION DATASET/        # Root directory for the dataset
├── EXPIRY_DATE/            # Main dataset folder
│   ├── EXPIRY_DATASET/          # Subfolder containing the actual dataset
│   │   ├── train/              # Training data
│   │       ├── images/         # Folder for images
│   │       └── label/          # Folder for corresponding labels
│   │   ├── valid/              # Validation data
│   │       ├── images/         # Folder for images
│   │       └── label/          # Folder for corresponding labels
│   │   └── test/               # Test data
│   │       ├── images/         # Folder for images
│   │       └── label/          # Folder for corresponding labels
└── Data.yaml                   # Yaml file for the dataset
```
## Inference
![image](https://github.com/user-attachments/assets/349c4c10-4bb4-4f71-a76c-1466c7ba061e)
![image](https://github.com/user-attachments/assets/f8919c09-7fef-4baa-b568-14a00f2474b7)



* Download the training and validation dataset from [Google Drive Link](https://drive.google.com/drive/folders/1xEngKyROVF7RoKpfSpabhQ8wd76B9jhB?usp=drive_link)
```
Fruits_Vegetables_Dataset/     # Root directory
├── train/                  # Subfolder containing fresh and rotten fruits, vegetables images
│   ├── FreshApple
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...                    # More fresh images
│   ├── RottenApple
│   ├── image_001.jpg
│   ├── image_002.jpg
...                          # More classes containing images
├── test/                    # Subfolder containing test images of fruits and vegetables 
│   ├── FreshApple
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...                   
│   ├── RottenApple
│   ├── image_001.jpg
│   ├── image_002.jpg              
```
## Inference
![image1](https://github.com/user-attachments/assets/b56c662b-709a-4440-9659-3950432a5fea)
![image2](https://github.com/user-attachments/assets/2faca2ae-0b59-46ca-bc36-7f20865be19f)


# Authors and Contributers
•  [Achintya Agarwal](https://www.linkedin.com/in/achintya-agarwal-bab26a21b/)

•  [Aryam Juneja](https://www.linkedin.com/in/aryam-juneja/)

•  [Pratham](https://www.linkedin.com/in/pratham-sharma-771397227/)
