# Real-Time Hand Gesture Recognition with MediaPipe, ASL/ISL Support, and Translation

This project implements a real-time hand gesture recognition system using Google's MediaPipe framework. It can recognize both single-handed American Sign Language (ASL) gestures and two-handed Indian Sign Language (ISL) gestures from a webcam feed. Recognized gestures are used to form sentences, which can then be translated from English to Tamil. The system also includes functionality for logging new gesture data to extend its capabilities.

![Demo GIF](https://user-images.githubusercontent.com/37477845/102222442-c452cd00-3f26-11eb-93ec-c387c98231be.gif)
*(The GIF shows the ASL functionality of the original project. This enhanced version includes ISL and other features described below.)*

This project is an enhanced English-translated version with added features, based on the foundational work by [Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe).

## Table of Contents

1.  [Detailed Features](#detailed-features)
2.  [System Workflow (The Detailed Process)](#system-workflow-the-detailed-process)
    * [Data Collection (Landmark Extraction)](#data-collection-landmark-extraction)
    * [Model Training](#model-training)
    * [Real-time Inference Application](#real-time-inference-application)
3.  [Directory Structure](#directory-structure)
4.  [Requirements](#requirements)
5.  [Installation](#installation)
6.  [Usage / How to Run](#usage--how-to-run)
    * [Running the Web Application](#running-the-web-application)
    * [Command-Line Arguments for `app.py`](#command-line-arguments-for-apppy)
7.  [Training Your Own Models](#training-your-own-models)
    * [1. Data Preparation & Landmark Extraction](#1-data-preparation--landmark-extraction)
    * [2. Model Training Process](#2-model-training-process)
8.  [Key Files Description](#key-files-description)
9.  [Contributing](#contributing)
10. [License](#license)
11. [Acknowledgements](#acknowledgements)

## Detailed Features

This system boasts a range of features designed for robust gesture recognition and user interaction:

1.  **Real-time Hand Gesture Recognition:**
    * Utilizes **MediaPipe Hands** for accurate and fast detection of 21 hand landmarks (for each hand).
    * Processes webcam feed in real-time to identify gestures.

2.  **Dual Sign Language Mode Support:**
    * **ASL (American Sign Language) Mode:** Recognizes gestures performed with a single hand.
    * **ISL (Indian Sign Language) Mode:** Recognizes gestures performed using two hands simultaneously.
    * Users can switch between ASL and ISL modes through the web interface.

3.  **Gesture-to-Text Conversion & Sentence Building:**
    * Recognized gestures are mapped to their corresponding characters or words.
    * **Sentence Construction:**
        * **Concatenation:** Appends recognized characters/words to form a sentence.
        * **"Space" Gesture:** Adds a space to the sentence.
        * **"Delete" Gesture:** Removes the last character from the sentence.
        * **"Clear" Gesture:** Clears the entire constructed sentence.
        * **"Nothing" / "Unknown":** These recognized states are typically ignored in sentence formation to prevent noise.
    * A history buffer and most-common-gesture logic (`history_length_fg`) are used to stabilize gesture recognition before adding to the sentence, preventing rapid, unintended character additions.

4.  **Visual Feedback - Recognized Sign Image Display:**
    * The web interface displays an image corresponding to the currently recognized sign language gesture (e.g., an image of the letter 'A' when 'A' is signed).
    * Images are stored in `static/ASL_Images/` and `static/ISL_Images/`.

5.  **English to Tamil Translation:**
    * An optional feature to translate the constructed English sentence into Tamil.
    * Uses the **NLLB (No Language Left Behind) model** (`facebook/nllb-200-distilled-600M`) from Hugging Face Transformers for translation.
    * Can be toggled on/off (translation is enabled by default but can be disabled via a command-line argument).
    * Translated sentences are displayed in the UI and saved to a text file in the `outputs/` directory.

6.  **Keypoint Logging Mode:**
    * Allows users to easily add new gesture data for model retraining or expansion.
    * When the 'k' key is pressed (in the OpenCV window while `app.py` is running), the system enters logging mode.
    * It captures the current hand landmarks from the webcam.
    * The user is prompted in the console to enter a numeric label for the gesture being performed.
    * These landmarks and the label are appended to the respective CSV files (`model_asl/keypoint_classifier/keypoint.csv` or `model_isl/keypoint_classifier/keypoint_isl_2h.csv`).

7.  **Interactive Web Interface (Flask-based):**
    * Provides a user-friendly interface to interact with the system.
    * **Controls:**
        * Start/Stop Webcam.
        * Select Mode (ASL/ISL).
        * "Translate" button to trigger Tamil translation.
    * **Displays:**
        * Live webcam feed with MediaPipe landmark overlays.
        * Currently recognized gesture label.
        * Image of the recognized gesture.
        * Constructed English sentence.
        * Translated Tamil sentence.
        * Frames Per Second (FPS) of the video processing.

8.  **Customizable Confidence Thresholds:**
    * Allows adjustment of minimum detection and tracking confidence for MediaPipe via command-line arguments to fine-tune performance based on lighting conditions or camera quality.

9.  **Pre-trained Models:**
    * Includes pre-trained TFLite models for both ASL and ISL gestures.
    * Labels for gestures are provided in corresponding CSV files.

## System Workflow (The Detailed Process)

The project operates in three main stages: Data Collection, Model Training, and Real-time Inference.

### Data Collection (Landmark Extraction)

This stage involves capturing hand gesture images and converting them into a numerical format (landmarks) suitable for machine learning.

1.  **Image Acquisition:**
    * Collect a dataset of images for each gesture you want to recognize.
    * Organize images into subdirectories where each subdirectory name is the gesture label (e.g., `dataset/asl/A/`, `dataset/asl/B/`, `dataset/isl/Hello/`).
    * For single-handed gestures (ASL), each image should clearly show one hand performing the gesture.
    * For two-handed gestures (ISL), each image should clearly show both hands performing the gesture.

2.  **Landmark Extraction (`extract_landmarks.py` & `extract_landmarks_2h.py`):**
    * **`extract_landmarks.py` (Single Hand - ASL):**
        * **Input:** Path to the directory containing gesture image subdirectories.
        * **Process:**
            * Iterates through each image.
            * Uses MediaPipe Hands (configured for `max_num_hands=1`) to detect 21 landmarks on the hand.
            * **Normalization:** Converts absolute landmark coordinates (pixels) into coordinates relative to the hand's wrist and bounding box. This makes the model robust to hand size and position variations.
            * The 21 landmarks (x, y coordinates each) result in 42 features per gesture.
        * **Output:** A CSV file (e.g., `model_asl/keypoint_classifier/keypoint.csv`) where each row contains: `label, x1, y1, x2, y2, ..., x21, y21`.
    * **`extract_landmarks_2h.py` (Two Hands - ISL):**
        * **Input:** Path to the directory containing two-handed gesture image subdirectories.
        * **Process:**
            * Iterates through each image.
            * Uses MediaPipe Hands (configured for `max_num_hands=2`) to detect landmarks for up to two hands.
            * **Normalization:** Similar to single-hand, landmarks for each hand are normalized.
            * If two hands are detected, their landmarks are concatenated (e.g., Hand1_x1, Hand1_y1, ..., Hand1_x21, Hand1_y21, Hand2_x1, Hand2_y1, ..., Hand2_x21, Hand2_y21). This results in 84 features.
            * The script handles cases where one or no hands are detected by padding with zeros, ensuring a consistent 84-feature vector.
        * **Output:** A CSV file (e.g., `model_isl/keypoint_classifier/keypoint_isl_2h.csv`) where each row contains: `label, h1_x1, h1_y1, ..., h1_x21, h1_y21, h2_x1, h2_y1, ..., h2_x21, h2_y21`.

### Model Training

Once the landmark data is prepared, machine learning models are trained to classify gestures.

* **Scripts:** `keypoint_classification_EN.ipynb` (for ASL/single-hand) and `keypoint_classification_2h.ipynb` (for ISL/two-hands).
* **Process (Common to both notebooks):**
    1.  **Load Data:** Reads the landmark CSV file generated in the previous step.
    2.  **Data Preparation:**
        * Separates features (landmark coordinates) from labels (gesture IDs).
        * Converts labels to categorical format (one-hot encoding).
        * Splits the data into training and testing sets.
    3.  **Model Definition (TensorFlow/Keras):**
        * A sequential neural network model is defined.
        * Typically consists of several Dense (fully connected) layers with ReLU activation functions, Dropout layers to prevent overfitting.
        * The final layer is a Dense layer with Softmax activation for multi-class classification, outputting probabilities for each gesture class.
    4.  **Model Compilation:**
        * Specifies the optimizer (e.g., Adam), loss function (e.g., `categorical_crossentropy`), and metrics (e.g., `accuracy`).
    5.  **Model Training:**
        * Trains the model using the training data.
        * Callbacks like `ModelCheckpoint` (to save the best model) and `EarlyStopping` (to prevent overfitting by stopping training if performance on a validation set doesn't improve) are used.
    6.  **Model Evaluation:**
        * Evaluates the trained model on the test set to assess its performance (accuracy).
        * Generates and displays a confusion matrix to visualize classification accuracy for each gesture.
    7.  **TFLite Conversion:**
        * The trained Keras model (`.h5`) is converted into a TensorFlow Lite (`.tflite`) format. TFLite models are lightweight and optimized for on-device inference, making them suitable for real-time applications.
* **Output:**
    * A `.tflite` model file (e.g., `keypoint_classifier.tflite`).
    * An updated label CSV file (e.g., `keypoint_classifier_label.csv`) mapping gesture IDs to human-readable names.

### Real-time Inference Application (`app.py`)

This is the main application that uses the trained models to recognize gestures from a live webcam feed.

1.  **Initialization:**
    * Loads the pre-trained TFLite models and corresponding label files for ASL and ISL.
    * Initializes the MediaPipe Hands solution.
    * Sets up the Flask web server.
    * (Optional) Loads the NLLB translation model and tokenizer if translation is enabled.

2.  **Webcam Video Stream Processing (per frame):**
    * **Capture Frame:** Reads a frame from the webcam using OpenCV.
    * **Hand Detection & Landmark Extraction:**
        * The frame is passed to MediaPipe Hands.
        * If hands are detected, MediaPipe returns the 21 landmarks for each hand. The number of hands processed (`max_num_hands`) depends on the selected mode (1 for ASL, 2 for ISL).
    * **Landmark Preprocessing:**
        * The detected landmarks are normalized (converted to relative coordinates) similar to the training data preparation process. This is crucial for the model to make accurate predictions.
    * **Gesture Classification:**
        * The preprocessed landmarks are fed as input to the appropriate loaded TFLite model (ASL or ISL based on UI selection).
        * The model outputs a probability distribution over the known gestures.
        * The gesture with the highest probability is selected as the recognized gesture ID.
    * **Label Mapping:** The recognized gesture ID is mapped to its human-readable label (e.g., 'A', 'Hello').
    * **Gesture Debouncing/History:** A short history of recognized gestures is maintained. The most frequent gesture in this history is chosen as the final recognized sign to improve stability and reduce jitter.
    * **Sentence Logic:** Based on the final recognized sign:
        * If it's a character/word, append it to the `sentence` string.
        * If it's "Space", add " ".
        * If it's "Delete", remove the last character.
        * If it's "Clear", empty the `sentence`.
    * **Image Display:** The image corresponding to the recognized gesture is selected.

3.  **User Interface Update (via Flask and WebSockets/Streaming):**
    * The processed frame (with landmarks drawn) is streamed to the web UI.
    * The recognized gesture label, corresponding image, constructed English sentence, and (if translated) Tamil sentence are updated on the web page.
    * FPS is calculated and displayed.

4.  **Translation (on demand):**
    * When the "Translate" button is clicked in the UI:
        * The current English sentence is passed to the `translate_to_tamil` function.
        * The NLLB model translates the text.
        * The translated Tamil sentence is displayed and saved to a file.

5.  **Keypoint Logging (on 'k' key press):**
    * If the 'k' key is pressed in the OpenCV window:
        * The system prompts for a gesture label (number) in the console.
        * The current hand landmarks (normalized) are logged along with the entered label to the appropriate CSV file, facilitating model retraining with new data.

## Directory Structure

```
hand-gesture-recognition-mediapipe-main/
│
├── app.py                            # Main Flask web application for real-time inference
├── extract_landmarks.py              # Script to extract single-hand landmarks from images for training
├── extract_landmarks_2h.py           # Script to extract two-handed landmarks from images for training
├── keypoint_classification_EN.ipynb  # Jupyter Notebook for ASL (single-hand) model training
├── keypoint_classification_2h.ipynb # Jupyter Notebook for ISL (two-hands) model training
├── README.md                         # This file
├── flowchart.html                    # HTML file illustrating project workflow (if available)
│
├── model_asl/                        # ASL (Single-Hand) Model and related files
│   ├── __init__.py
│   └── keypoint_classifier/
│       ├── keypoint.csv              # Landmark training data for ASL gestures
│       ├── keypoint_classifier.py    # Python module for ASL TFLite model inference
│       ├── keypoint_classifier.tflite # Trained ASL TFLite model
│       └── keypoint_classifier_label.csv # Labels for ASL gestures
│
├── model_isl/                        # ISL (Two-Handed) Model and related files
│   ├── __init__.py
│   └── keypoint_classifier/
│       ├── keypoint_isl_2h.csv       # Landmark training data for ISL gestures (example name)
│       ├── keypoint_classifier.py    # Python module for ISL TFLite model inference
│       ├── keypoint_classifier.tflite # Trained ISL TFLite model
│       └── keypoint_classifier_label.csv # Labels for ISL gestures
│
├── static/                           # Static files for the web app (CSS, JS, Images)
│   ├── ASL_Images/                   # Images for ASL gestures (e.g., A.png, B.png)
│   └── ISL_Images/                   # Images for ISL gestures (e.g., Hello.png)
│
├── templates/
│   └── index.html                    # HTML template for the web interface
│
├── utils/
│   ├── __init__.py
│   └── cvfpscalc.py                  # Utility for FPS calculation
│
└── outputs/                          # Directory where translated sentences are saved
    └── sentence_output_YYYYMMDD_HHMMSS.txt # Example output file
```

## Requirements

* Python 3.8+
* MediaPipe >= 0.8.1
* OpenCV (cv2) >= 3.4.2
* TensorFlow >= 2.5.0 (for training and TFLite conversion)
* Flask >= 2.0
* NumPy >= 1.19
* Pillow (PIL) (for image handling in `app.py` if needed, though OpenCV often suffices)
* **For Translation Feature (Optional):**
    * Transformers (by Hugging Face) >= 4.0
    * PyTorch >= 1.8 (or TensorFlow, NLLB can run on TF too)
    * SentencePiece (for NLLB tokenizer)
    * Accelerate (for efficient model loading with PyTorch)
* **For Model Training Notebooks (Optional, if retraining):**
    * Scikit-learn >= 0.23.2 (for train/test split, confusion matrix)
    * Matplotlib >= 3.3.2 (for plotting confusion matrix)
    * Jupyter Notebook or JupyterLab

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required packages:**
    A `requirements.txt` file would be ideal. If not present, install manually:
    ```bash
    pip install mediapipe opencv-python tensorflow flask numpy pillow
    ```
    For the **translation feature**:
    ```bash
    pip install transformers torch sentencepiece accelerate
    ```
    For **retraining models** using the notebooks:
    ```bash
    pip install scikit-learn matplotlib jupyter
    ```
    *Note: Installing PyTorch might have specific instructions depending on your CUDA version if you plan to use a GPU. Refer to the [official PyTorch website](https://pytorch.org/).*

## Usage / How to Run

### Running the Web Application

1.  Ensure all requirements are installed.
2.  Navigate to the project's root directory in your terminal.
3.  Run the Flask application:
    ```bash
    python app.py
    ```
4.  Open your web browser and go to the URL displayed in the terminal (usually `http://127.0.0.1:5000` or `http://localhost:5000`).

**Interacting with the UI:**
* Click "Start Webcam" to begin gesture recognition.
* Select "ASL Mode" or "ISL Mode" from the dropdown.
* Perform gestures in front of the webcam.
* The recognized gesture, its image, the formed sentence, and (if translated) Tamil text will appear.
* Click "Translate" to get the Tamil translation of the current sentence.
* Press 'k' in the OpenCV window (the one showing the raw camera feed with landmarks) if you want to log new keypoints for a gesture (you'll be prompted for a label in the console).

### Command-Line Arguments for `app.py`

You can customize the application's behavior using these command-line arguments:

```bash
python app.py [options]
```

* `--device`: Camera device number (Default: `0`).
* `--width`: Camera capture width (Default: `960`).
* `--height`: Camera capture height (Default: `540`).
* `--use_static_image_mode`: Boolean, use `static_image_mode` for MediaPipe Hands (Default: `False`).
* `--min_detection_confidence`: Minimum detection confidence for MediaPipe Hands (Default: `0.7`).
* `--min_tracking_confidence`: Minimum tracking confidence for MediaPipe Hands (Default: `0.5`).
* `--model_path_asl`: Path to the ASL TFLite model file (Default: `model_asl/keypoint_classifier/keypoint_classifier.tflite`).
* `--label_path_asl`: Path to the ASL labels CSV file (Default: `model_asl/keypoint_classifier/keypoint_classifier_label.csv`).
* `--model_path_isl`: Path to the ISL TFLite model file (Default: `model_isl/keypoint_classifier/keypoint_classifier.tflite`).
* `--label_path_isl`: Path to the ISL labels CSV file (Default: `model_isl/keypoint_classifier/keypoint_classifier_label.csv`).
* `--history_length_fg`: Length of the gesture history buffer for stabilizing recognition (Default: `16`).
* `--no_translation`: Add this flag to disable the English to Tamil translation feature.

Example:
```bash
python app.py --device 1 --min_detection_confidence 0.6 --no_translation
```

## Training Your Own Models

If you want to add new gestures or improve the existing models, follow these steps:

### 1. Data Preparation & Landmark Extraction

* **Collect Gesture Images:**
    * For each new gesture, capture multiple images from various angles and slightly different hand positions. Ensure good lighting and a clear background.
    * Organize these images into subdirectories named after the gesture. For example, if adding a "ThumbsUp" gesture for ASL: `dataset/asl_custom/ThumbsUp/image1.jpg, image2.jpg ...`
    * For ISL, ensure both hands are clearly visible.

* **Extract Landmarks:**
    * **For Single-Hand Gestures (e.g., ASL):**
        Use `extract_landmarks.py`.
        ```bash
        python extract_landmarks.py --dataset_dir path/to/your/single_hand_gesture_images --output_csv path/to/your_asl_keypoints.csv
        ```
        This script will process images, extract MediaPipe hand landmarks, and save them to the specified CSV file. Each row will have `label,x1,y1,...,x21,y21`.
    * **For Two-Handed Gestures (e.g., ISL):**
        Use `extract_landmarks_2h.py`.
        ```bash
        python extract_landmarks_2h.py --dataset_dir path/to/your/two_handed_gesture_images --output_csv path/to/your_isl_keypoints.csv
        ```
        This script saves 84 landmark features for two hands. Each row will have `label,h1_x1,h1_y1,...,h1_x21,h1_y21,h2_x1,h2_y1,...,h2_x21,h2_y21`.

    *Make sure the labels in your dataset directory correspond to the numeric IDs you want for training. You might need to manually update the `keypoint.csv` and `keypoint_classifier_label.csv` files or adapt the notebooks if your labeling scheme is different.*

### 2. Model Training Process

* Use the Jupyter Notebooks provided:
    * `keypoint_classification_EN.ipynb` for single-hand/ASL models.
    * `keypoint_classification_2h.ipynb` for two-hand/ISL models.

* **Steps in the Notebook:**
    1.  **Modify Paths:** Update the path to your newly generated landmark CSV file (e.g., `your_asl_keypoints.csv`).
    2.  **Adjust Number of Classes:** If you've added or removed gesture classes, ensure the `NUM_CLASSES` variable in the notebook reflects this. The output layer of the neural network will also need to be adjusted accordingly.
    3.  **Run Cells:** Execute the cells in the notebook sequentially. This will:
        * Load and preprocess your landmark data.
        * Train the neural network.
        * Evaluate its performance.
        * Save the trained model as a `.h5` file and then convert it to a `.tflite` file.
        * It will also generate an updated `keypoint_classifier_label.csv`.
    4.  **Replace Old Files:** Once training is complete, replace the old `.tflite` model and `keypoint_classifier_label.csv` file in the respective `model_asl/keypoint_classifier/` or `model_isl/keypoint_classifier/` directory with your newly trained ones.

## Key Files Description

* `app.py`: The core Flask application that integrates webcam input, MediaPipe processing, model inference, sentence logic, translation, and serves the web UI.
* `extract_landmarks.py`: Script for batch processing image datasets of single-hand gestures to extract MediaPipe landmarks and save them in CSV format for training.
* `extract_landmarks_2h.py`: Similar to above, but for two-handed gestures, extracting 84 landmark features.
* `keypoint_classification_EN.ipynb`: Jupyter Notebook detailing the training pipeline for the single-hand (ASL) gesture recognition model. It covers data loading, preprocessing, model building (TensorFlow/Keras), training, evaluation, and TFLite conversion.
* `keypoint_classification_2h.ipynb`: Jupyter Notebook for training the two-handed (ISL) gesture recognition model, following a similar pipeline but adapted for 84 input features.
* `model_*/keypoint_classifier/keypoint_classifier.py`: Contains the `KeyPointClassifier` class used by `app.py` to load a TFLite model and perform inference on input landmarks.
* `model_*/keypoint_classifier/*.tflite`: The lightweight, optimized TensorFlow Lite models used for real-time gesture classification.
* `model_*/keypoint_classifier/*_label.csv`: CSV files mapping numerical class IDs (used by the model) to human-readable gesture names.
* `templates/index.html`: The HTML file that defines the structure and elements of the web user interface.
* `utils/cvfpscalc.py`: A utility class to calculate and display Frames Per Second (FPS).

## Contributing

Contributions are welcome! If you have suggestions for improvements or want to add new features, please follow these steps:

1.  Fork the Project.
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the Branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

Please ensure your code adheres to clear coding standards and is well-commented.

## License

This project does not explicitly state a license. The original base project by Kazuhito00 also does not specify a license in its `README.md`. Please be mindful of this if you plan to use or distribute this software. It's good practice to add an open-source license (e.g., MIT, Apache 2.0) if you intend for wider use and contribution.

## Acknowledgements

* This project is heavily based on the work of **Kazuhito Nakashima (Kazuhito00)** and his repository [hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe).
* **Google MediaPipe** for the powerful hand tracking solution.
* **TensorFlow** and **Keras** for the machine learning framework.
* **Flask** for the web application framework.
* **Hugging Face Transformers** and the **NLLB team** for the translation model.
* The creators of OpenCV and other open-source libraries used.

```
