EyeCatcher: Real-time Eye Tracking Using CNNs
=============================================

Overview
--------

EyeCatcher implements a real-time eye-tracking system, utilizing convolutional neural network (CNN) architectures to accurately detect and track eye positions in video frames. By applying advanced data augmentation techniques and optimizing the model’s hyperparameters, the system enhances performance and accuracy in tracking eye movements.

Installation and Dependencies
-----------------------------

To install EyeCatcher and its dependencies, run the following command:  
pip install opencv-python dlib tensorflow keras numpy matplotlib  

### Dependencies:

*   - **OpenCV** ([`opencv-python`](https://pypi.org/project/opencv-python/)): 
  - Used for video processing tasks such as reading frames from video files and manipulating images.
  
- **Matplotlib** ([`matplotlib`](https://pypi.org/project/matplotlib/)): 
  - Utilized for visualizing data, particularly for plotting and displaying images.
  
- **Imageio** ([`imageio`](https://pypi.org/project/imageio/)): 
  - Facilitates reading and writing a wide range of image data, including animated sequences.
  
- **gdown** ([`gdown`](https://pypi.org/project/gdown/)): 
  - A simple Python tool to download files from Google Drive.
  
- **TensorFlow** ([`tensorflow`](https://pypi.org/project/tensorflow/)): 
  - The core library for developing and training the deep learning model.
  
- **NumPy** ([`numpy`](https://pypi.org/project/numpy/)): 
  - Essential for numerical operations and handling arrays, which are extensively used in machine learning.
 
- **Dlib** ([`dlib`](https://pypi.org/project/dlib/)):  
  A toolkit for machine learning and computer vision tasks, used for facial landmark detection and feature extraction in lip reading.
      
    
Dataset Collection and Preprocessing
------------------------------------
    
EyeCatcher requires labeled video datasets of human faces to train the model effectively. Data collection was performed using OpenCV, followed by preprocessing steps such as frame extraction, grayscale conversion, and landmark detection.
   ### Data Collection Script:
```python
    import cv2

def collect_data(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f'{output_dir}/frame_{count}.png', frame)
        count += 1
        
    cap.release()
    cv2.destroyAllWindows()
```
    
This script processes each video frame, saving them for further preprocessing and model training.


### Preprocessing:

The key preprocessing steps include:

1.  **Grayscale Conversion**: Converts images to grayscale to reduce model complexity.
2.  **Eye Region Cropping**: Detects and crops eye regions based on facial landmarks using the `dlib` library.
3.  **Data Augmentation**: Random rotations, zooms, and shifts are applied to increase training data diversity.


CNN Model Architecture
----------------------

The core of EyeCatcher is a Convolutional Neural Network (CNN) designed to detect and track eye positions across video frames. Below is the model architecture:  
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

# First convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(60, 60, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third convolutional layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))  # Output layer for eye position (x, y coordinates)
```

### Why CNNs?

CNNs are chosen for their ability to capture spatial features such as shapes and textures, making them highly effective for visual tasks like eye tracking. Pooling layers reduce the dimensionality, allowing the model to generalize better and prevent overfitting.

Training the Model
------------------

The model is trained using mean squared error (MSE) as the loss function to minimize the error between predicted and true eye positions. Adam optimizer is employed for efficient training.
```python
model.compile(optimizer='adam', loss='mse')

# Training
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)
```


### Training Data Augmentation

Augmentation techniques include:

*   **Random Rotations**: To handle head tilts.
*   **Zooming**: To account for varying distances from the camera.
*   **Shifting**: To simulate slight head movements.

Real-time Eye Tracking
----------------------
The system tracks eye positions in real-time using a combination of OpenCV and the trained CNN model.
```python
def track_eye(video_source):
    cap = cv2.VideoCapture(video_source)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        eye_position = model.predict(process_frame(frame))
        cv2.circle(frame, eye_position, 5, (0, 255, 0), -1)  # Draw a circle at the predicted eye position
        
        cv2.imshow('Eye Tracker', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

  

Results
-------

After training, EyeCatcher successfully tracks eyes in real-time, maintaining high accuracy even in varying lighting conditions and head movements.


Conclusion
----------

EyeCatcher demonstrates the effectiveness of CNNs in real-time eye tracking applications. By leveraging advanced data augmentation and model optimization techniques, it achieves reliable performance in detecting and tracking eye movements in video frames.


