#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import copy
import csv
import itertools
import logging  # Import logging
import os
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2 as cv
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, url_for

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Pillow (PIL) for Image Conversion ---
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    logging.warning("Pillow library not found. Install using: pip install Pillow")
    PIL_AVAILABLE = False


# --- NLLB Translation Handling ---
TARGET_TRANSLATION_LANG = "tam_Taml"  # Target language (Tamil)
SOURCE_TRANSLATION_LANG = "eng_Latn"  # Source language (English)
NLLB_MODEL_CHECKPOINT = "facebook/nllb-200-1.3B"  # ~5GB model download

translator_pipeline = None
translation_active = False
translation_result = ""
translation_thread = None  # For managing the translation thread
NLLB_PREREQUISITES_AVAILABLE = False  # Default to False

try:
    # Try importing necessary libraries
    import torch  # Check if torch is available
    from transformers import AutoTokenizer, pipeline

    logging.info("Transformers and PyTorch libraries found.")
    NLLB_PREREQUISITES_AVAILABLE = True
except ImportError as e:
    logging.warning(
        f"NLLB Translation prerequisites not found. Translation disabled. Error: {e}"
    )
    logging.warning(
        "Install them using: pip install transformers torch sentencepiece accelerate"
    )
    # Keep NLLB_PREREQUISITES_AVAILABLE as False


# --- MediaPipe Handling ---
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    logging.error("mediapipe library not found.")
    logging.error("Install it using: pip install mediapipe")
    sys.exit(1)


# --- Local Module Handling ---
try:
    from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
    from utils.cvfpscalc import CvFpsCalc

    UTILS_AVAILABLE = True
except ImportError as e:
    logging.error("Could not import local modules 'utils' or classifier.")
    logging.error(f"Specific error: {e}")
    logging.error(
        "Ensure 'utils/cvfpscalc.py', 'model/keypoint_classifier/keypoint_classifier.py' are accessible and their packages have __init__.py."
    )

    class CvFpsCalc:  # Dummy class if import fails
        def __init__(self, buffer_len=1):
            pass

        def get(self):
            return 0.0

    UTILS_AVAILABLE = False
    KeyPointClassifier = None  # Ensure it's defined even if import fails


# --- Constants ---
MODE_ASL = 0
MODE_ISL = 1
MODE_LOGGING = 2
MODE_NAMES = {
    MODE_ASL: "ASL Mode",
    MODE_ISL: "ISL Mode",
    MODE_LOGGING: "Logging",
}

LOG_LANG_ASL = "ASL (1 Hand)"
LOG_LANG_ISL = "ISL (2 Hands)"
LOG_CSV_PATHS = {
    LOG_LANG_ASL: "model_asl/keypoint_classifier/keypoint.csv",
    LOG_LANG_ISL: "model_isl/keypoint_classifier/keypoint_isl_2h.csv",
}

NUM_LANDMARKS_SINGLE_HAND = 21
NUM_COORDS = 2
SINGLE_HAND_VECTOR_SIZE = NUM_LANDMARKS_SINGLE_HAND * NUM_COORDS  # 42
ZERO_VECTOR_SINGLE_HAND = [0.0] * SINGLE_HAND_VECTOR_SIZE
COMBINED_VECTOR_SIZE = SINGLE_HAND_VECTOR_SIZE * 2  # 84
ZERO_VECTOR_COMBINED = [0.0] * COMBINED_VECTOR_SIZE

SIGN_CONFIRM_THRESHOLD = 2.0  # Seconds to hold sign
SPACE_INSERT_THRESHOLD = 3.5  # Seconds with no hand for space
DEBOUNCE_TIME = 0.5  # Min time between adding chars


# --- Flask App Setup ---
# Serve static files from the 'static' directory
app = Flask(__name__, static_url_path="/static", static_folder="static")


# --- Global Variables ---
args = None
cap = None
hands = None
asl_keypoint_classifier = None
asl_keypoint_classifier_labels = []
isl_keypoint_classifier = None
isl_keypoint_classifier_labels = []
cvFpsCalc = CvFpsCalc(buffer_len=10)

# Application State
current_mode = MODE_ASL
current_sentence = []
last_added_time = time.time()
last_hand_detected_time = time.time()
current_sign_label = None  # Label of the sign currently being held
sign_start_time = None
sign_progress = 0.0
output_filename = None
actively_detected_sign_label = None  # Label of the sign being held OR just added (for image display)

# Logging State
selected_log_symbol_index = -1
logging_language = LOG_LANG_ASL
processed_landmarks_log_single = None
processed_landmarks_log_combined = None


# --- Helper Functions ---
def get_args_flask():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument(
        "--use_static_image_mode", action="store_true", default=False
    )
    parser.add_argument(
        "--min_detection_confidence", type=float, default=0.7
    )
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    parser.add_argument(
        "--model_path_asl",
        type=str,
        default="model_asl/keypoint_classifier/keypoint_classifier.tflite",
    )
    parser.add_argument(
        "--label_path_asl",
        type=str,
        default="model_asl/keypoint_classifier/keypoint_classifier_label.csv",
    )
    parser.add_argument(
        "--model_path_isl",
        type=str,
        default="model_isl/keypoint_classifier/keypoint_classifier.tflite",
    )
    parser.add_argument(
        "--label_path_isl",
        type=str,
        default="model_isl/keypoint_classifier/keypoint_classifier_label.csv",
    )
    parser.add_argument(
        "--no_translation", action="store_true", default=False
    )

    known_args, _ = parser.parse_known_args()
    return known_args


# --- Landmark Calculation Functions ---
def calc_bounding_rect(image, landmarks):
    if landmarks is None or landmarks.landmark is None:
        return [0, 0, 0, 0]
    h, w, _ = image.shape
    landmark_array = np.empty((0, 2), int)
    for lm in landmarks.landmark:
        if lm is None:
            continue
        lx, ly = int(lm.x * w), int(lm.y * h)
        landmark_array = np.append(landmark_array, [[lx, ly]], axis=0)
    if landmark_array.shape[0] == 0:
        return [0, 0, 0, 0]
    try:
        x, y = np.min(landmark_array, axis=0)
        x_max, y_max = np.max(landmark_array, axis=0)
        rw, rh = x_max - x, y_max - y
        padding = 10
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(w, x + rw + padding), min(h, y + rh + padding)
        return [x1, y1, x2, y2]
    except ValueError:
        return [0, 0, 0, 0]


def calc_landmark_list(image, landmarks):
    if landmarks is None or landmarks.landmark is None:
        return []
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for lm in landmarks.landmark:
        if lm is None:
            continue
        lx = min(int(lm.x * image_width), image_width - 1)
        ly = min(int(lm.y * image_height), image_height - 1)
        landmark_point.append([lx, ly])
    return landmark_point


def pre_process_landmark(landmark_list):
    if not landmark_list or len(landmark_list) != NUM_LANDMARKS_SINGLE_HAND:
        return ZERO_VECTOR_SINGLE_HAND
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for i in range(len(temp_landmark_list)):
        temp_landmark_list[i][0] -= base_x
        temp_landmark_list[i][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = (
        max(map(abs, temp_landmark_list)) if temp_landmark_list else 1.0
    )
    if max_value == 0:
        return ZERO_VECTOR_SINGLE_HAND
    normalize_ = lambda n: n / max_value
    normalized_landmark_list = list(map(normalize_, temp_landmark_list))
    normalized_landmark_list = (
        normalized_landmark_list + ZERO_VECTOR_SINGLE_HAND
    )[:SINGLE_HAND_VECTOR_SIZE]
    return normalized_landmark_list


# --- Logging Function ---
def logging_csv(number, landmark_vector, csv_path):
    global logging_language
    if number < 0 or not landmark_vector:
        logging.error("Log Error: Invalid number or empty landmark vector.")
        return
    try:
        dirname = os.path.dirname(csv_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
    except OSError as e:
        logging.error(
            f"Log Error: Could not create directory for {csv_path}: {e}"
        )
        return

    vector_size = len(landmark_vector)
    logging.info(
        f"Logging Class {number} to {os.path.basename(csv_path)} ({logging_language}, Vector size: {vector_size})..."
    )
    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_vector])
    except Exception as e:
        logging.error(
            f"Log Error: Error writing to CSV file {csv_path}: {e}"
        )


# --- Drawing Functions ---
def draw_landmarks(image, landmarks):
    if landmarks:
        mp_drawing.draw_landmarks(
            image,
            landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
    return image


def draw_bounding_rects(image, brects):
    for brect in brects:
        if brect[2] > brect[0] and brect[3] > brect[1]:
            cv.rectangle(
                image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 2
            )
            cv.rectangle(
                image,
                (brect[0], brect[1]),
                (brect[2], brect[3]),
                (255, 255, 255),
                1,
            )
    return image


def draw_info_texts(image, brects, handedness_list, sign_texts):
    for i, brect in enumerate(brects):
        if i >= len(handedness_list) or handedness_list[i] is None:
            continue
        if brect[2] <= brect[0] or brect[3] <= brect[1]:
            continue

        handedness_info = handedness_list[i].classification[0]
        handedness_label = handedness_info.label
        handedness_score = handedness_info.score
        sign_text_to_display = sign_texts[i] if i < len(sign_texts) else ""

        text_box_height = 22
        text_y = brect[1] - 6
        box_y1 = max(0, brect[1] - text_box_height)
        box_y2 = brect[1]

        cv.rectangle(
            image, (brect[0], box_y1), (brect[2], box_y2), (0, 0, 0), -1
        )
        info_text = f"{handedness_label[0]} ({handedness_score:.2f})"

        # Only show classified sign text if it's valid
        if sign_text_to_display and sign_text_to_display not in [
            "Unknown",
            "Processing...",
            "Classifier Error",
            "Unknown ID",
        ]:
            info_text += f": {sign_text_to_display}"

        cv.putText(
            image,
            info_text,
            (brect[0] + 5, text_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
    return image


def draw_sign_progress(image, progress):
    h, w, _ = image.shape
    bar_h = 10
    bar_y = h - bar_h
    prog_w = int(w * progress)

    # Darker gray background
    cv.rectangle(image, (0, bar_y), (w, h), (70, 70, 70), -1)
    # Blue progress
    cv.rectangle(image, (0, bar_y), (prog_w, h), (0, 123, 255), -1)
    return image


def draw_mode_info(image, mode):
    mode_string = MODE_NAMES.get(mode, "Unknown Mode")
    cv.putText(
        image,
        "MODE:" + mode_string,
        (10, 90),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )
    return image


# --- Sentence Logic ---
def add_char_to_sentence(char):
    global current_sentence, last_added_time, output_filename, actively_detected_sign_label
    logging.info(f"Adding ({MODE_NAMES[current_mode]}): '{char}'")
    actively_detected_sign_label = char  # Update active label when confirmed

    is_space = char == " "
    if not is_space:
        current_sentence.append(char)
    elif current_sentence and current_sentence[-1] != " ":
        current_sentence.append(char)
    else:
        logging.info("Skipping redundant space.")
        return

    last_added_time = time.time()
    if output_filename:
        try:
            with open(output_filename, "a", encoding="utf-8") as f:
                f.write(char)
        except Exception as e:
            logging.error(f"Error writing character to file: {e}")


def clear_sentence_state():
    global current_sentence, output_filename, actively_detected_sign_label
    current_sentence = []
    actively_detected_sign_label = None  # Clear active sign too
    if output_filename:
        try:
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write("")
        except Exception as e:
            logging.error(f"Error clearing output file: {e}")


def delete_last_char_state():
    global current_sentence, output_filename, actively_detected_sign_label
    if current_sentence:
        deleted_char = current_sentence.pop()
        logging.info(f"Deleting last character: '{deleted_char}'")
        # If sentence becomes empty, clear active sign
        if not current_sentence:
            actively_detected_sign_label = None

        current_text = "".join(current_sentence)
        if output_filename:
            try:
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write(current_text)
            except Exception as e:
                logging.error(f"Error updating output file: {e}")


# --- Initialization ---
def initialize_app():
    global args, cap, hands, asl_keypoint_classifier, asl_keypoint_classifier_labels
    global isl_keypoint_classifier, isl_keypoint_classifier_labels, output_filename
    global translator_pipeline  # Ensure it's global

    args = get_args_flask()

    # Load Labels
    try:
        with open(args.label_path_asl, encoding="utf-8-sig") as f:
            asl_keypoint_classifier_labels = [
                row[0] for row in csv.reader(f) if row and row[0]
            ]
        logging.info(
            f"Loaded {len(asl_keypoint_classifier_labels)} ASL labels."
        )
    except Exception as e:
        logging.error(
            f"Error loading ASL labels from {args.label_path_asl}: {e}"
        )
        asl_keypoint_classifier_labels = []

    try:
        with open(args.label_path_isl, encoding="utf-8-sig") as f:
            isl_keypoint_classifier_labels = [
                row[0] for row in csv.reader(f) if row and row[0]
            ]
        logging.info(
            f"Loaded {len(isl_keypoint_classifier_labels)} ISL labels."
        )
    except Exception as e:
        logging.error(
            f"Error loading ISL labels from {args.label_path_isl}: {e}"
        )
        isl_keypoint_classifier_labels = []

    # Initialize Classifiers
    if KeyPointClassifier:
        try:
            asl_keypoint_classifier = KeyPointClassifier(
                model_path=args.model_path_asl
            )
            logging.info(
                f"ASL KeyPointClassifier loaded from {args.model_path_asl}."
            )
        except Exception as e:
            logging.error(f"Error loading ASL KeyPointClassifier: {e}")
            asl_keypoint_classifier = None
        try:
            isl_keypoint_classifier = KeyPointClassifier(
                model_path=args.model_path_isl
            )
            logging.info(
                f"ISL KeyPointClassifier loaded from {args.model_path_isl}."
            )
        except Exception as e:
            logging.error(f"Error loading ISL KeyPointClassifier: {e}")
            isl_keypoint_classifier = None
    else:
        logging.warning(
            "KeyPointClassifier class not available. Classification disabled."
        )

    # Initialize MediaPipe Hands
    try:
        hands = mp_hands.Hands(
            static_image_mode=args.use_static_image_mode,
            max_num_hands=2,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        logging.info("MediaPipe Hands initialized.")
    except Exception as e:
        logging.error(f"Error initializing MediaPipe Hands: {e}")
        hands = None

    # Initialize Camera
    cap = cv.VideoCapture(args.device)
    if not cap.isOpened():
        logging.error(f"Could not open camera device {args.device}.")
        cap = None
    else:
        cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
        logging.info(
            f"Camera {args.device} initialized ({args.width}x{args.height})."
        )

    # Initialize NLLB Pipeline if prerequisites are met
    if not args.no_translation and NLLB_PREREQUISITES_AVAILABLE:
        logging.info("Attempting to load NLLB translation pipeline...")
        try:
            device_index = 0 if torch.cuda.is_available() else -1
            device_name = f"cuda:{device_index}" if device_index == 0 else "cpu"
            logging.info(f"NLLB using device: {device_name}")
            tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_CHECKPOINT)
            translator_pipeline = pipeline(
                "translation",
                model=NLLB_MODEL_CHECKPOINT,
                tokenizer=tokenizer,
                src_lang=SOURCE_TRANSLATION_LANG,
                tgt_lang=TARGET_TRANSLATION_LANG,
                device=device_index,
            )
            logging.info(
                f"NLLB translation pipeline ({NLLB_MODEL_CHECKPOINT}) loaded successfully."
            )
        except Exception as e:
            logging.error(f"Failed to load NLLB translation pipeline: {e}")
            logging.error("Translation feature will be unavailable.")
            translator_pipeline = None  # Ensure it remains None on error
    elif args.no_translation:
        logging.info(
            "NLLB translation disabled via --no_translation argument."
        )
    else:
        logging.info(
            "NLLB prerequisites not available, skipping pipeline initialization."
        )

    # Setup Logging Output File
    output_folder = "outputs"
    try:
        os.makedirs(output_folder, exist_ok=True)
        output_filename = os.path.join(
            output_folder,
            f"sentence_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )
        logging.info(f"Saving sentence output to: {output_filename}")
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("")
    except Exception as e:
        logging.warning(f"Could not set up output file: {e}")
        output_filename = None


# --- Video Frame Generation ---
def generate_frames():
    global current_mode, current_sentence, last_added_time, last_hand_detected_time
    global current_sign_label, sign_start_time, sign_progress, actively_detected_sign_label
    global processed_landmarks_log_single, processed_landmarks_log_combined

    # Error Handling for Camera/Hands Initialization Failure
    if cap is None or not cap.isOpened():
        logging.error("Camera not available for frame generation.")
        error_img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        cv.putText(
            error_img,
            "Camera Error",
            (50, args.height // 2),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        _, buffer = cv.imencode(".jpg", error_img)
        frame_bytes = buffer.tobytes()
        while True:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
            time.sleep(1)

    if hands is None:
        logging.error("MediaPipe Hands not initialized. Cannot process frames.")
        error_img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        cv.putText(
            error_img,
            "MediaPipe Error",
            (50, args.height // 2),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        _, buffer = cv.imencode(".jpg", error_img)
        frame_bytes = buffer.tobytes()
        while True:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
            time.sleep(1)

    # Main frame processing loop
    while True:
        fps = cvFpsCalc.get()
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to grab frame")
            time.sleep(0.1)
            continue

        image = cv.flip(frame, 1)
        debug_image = copy.deepcopy(image)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True

        hand_detected_this_frame = False
        processed_sign_label = "Unknown"  # Label from classifier this frame
        brects = []
        handedness_list_results = []
        (
            first_hand_norm,
            left_hand_norm,
            right_hand_norm,
        ) = (
            ZERO_VECTOR_SINGLE_HAND,
            ZERO_VECTOR_SINGLE_HAND,
            ZERO_VECTOR_SINGLE_HAND,
        )
        processed_landmarks_combined = ZERO_VECTOR_COMBINED

        if results.multi_hand_landmarks:
            hand_detected_this_frame = True
            last_hand_detected_time = time.time()

            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                brects.append(calc_bounding_rect(debug_image, hand_landmarks))
                if i < len(results.multi_handedness):
                    handedness = results.multi_handedness[i]
                    handedness_list_results.append(handedness)
                    hand_label_lr = handedness.classification[0].label
                    landmark_pixels = calc_landmark_list(
                        debug_image, hand_landmarks
                    )
                    normalized_single = pre_process_landmark(landmark_pixels)
                    if i == 0:
                        first_hand_norm = normalized_single
                    if hand_label_lr == "Left":
                        left_hand_norm = normalized_single
                    elif hand_label_lr == "Right":
                        right_hand_norm = normalized_single
                else:
                    handedness_list_results.append(None)

            processed_landmarks_log_single = first_hand_norm
            processed_landmarks_log_combined = left_hand_norm + right_hand_norm
            processed_landmarks_combined = left_hand_norm + right_hand_norm

            # Classification
            classifier, labels, input_landmarks = None, None, None
            if current_mode == MODE_ASL and asl_keypoint_classifier:
                classifier, labels, input_landmarks = (
                    asl_keypoint_classifier,
                    asl_keypoint_classifier_labels,
                    first_hand_norm,
                )
            elif current_mode == MODE_ISL and isl_keypoint_classifier:
                classifier, labels, input_landmarks = (
                    isl_keypoint_classifier,
                    isl_keypoint_classifier_labels,
                    processed_landmarks_combined,
                )

            if (
                classifier
                and labels
                and input_landmarks
                and any(lm != 0.0 for lm in input_landmarks)
            ):
                try:
                    sign_id = classifier(input_landmarks)
                    if 0 <= sign_id < len(labels):
                        processed_sign_label = labels[sign_id]
                    else:
                        processed_sign_label = "Unknown ID"
                        logging.warning(
                            f"Classifier ({MODE_NAMES[current_mode]}) returned invalid sign ID: {sign_id}"
                        )
                except Exception as e:
                    logging.error(
                        f"Classifier error ({MODE_NAMES[current_mode]}): {e}"
                    )
                    processed_sign_label = "Classifier Error"
            elif classifier:
                # Indicate model active but no input/result
                processed_sign_label = "Processing..."
        else:  # No hands detected
            processed_landmarks_log_single, processed_landmarks_log_combined = (
                None,
                None,
            )

        # Update Sentence & Active Sign Logic
        if current_mode in [MODE_ASL, MODE_ISL]:
            current_time = time.time()
            valid_sign_detected = hand_detected_this_frame and processed_sign_label not in [
                "Unknown",
                "Processing...",
                "Classifier Error",
                "Unknown ID",
            ]

            if valid_sign_detected:
                # Update the sign being actively held/processed for image display
                actively_detected_sign_label = processed_sign_label

                if current_sign_label != processed_sign_label:  # New valid sign
                    current_sign_label = processed_sign_label
                    sign_start_time = current_time
                    sign_progress = 0.0
                elif sign_start_time is not None:  # Continue holding same sign
                    elapsed_time = current_time - sign_start_time
                    sign_progress = min(
                        elapsed_time / SIGN_CONFIRM_THRESHOLD, 1.0
                    )
                    if (
                        sign_progress >= 1.0
                        and (current_time - last_added_time >= DEBOUNCE_TIME)
                    ):
                        add_char_to_sentence(current_sign_label)
                        current_sign_label = None  # Reset to allow new sign
                        sign_start_time = None
                        sign_progress = 0.0
            else:  # No valid sign detected this frame
                # Reset hold timer/progress if we were holding a sign
                if current_sign_label is not None:
                    current_sign_label = None
                    sign_start_time = None
                    sign_progress = 0.0
                # Clear active display label if enough time passed
                if (
                    current_time - max(last_hand_detected_time, last_added_time)
                    > DEBOUNCE_TIME
                ):
                    actively_detected_sign_label = None

            # Check for space insertion
            if not hand_detected_this_frame and (
                current_time - last_hand_detected_time >= SPACE_INSERT_THRESHOLD
            ):
                if current_sentence and current_sentence[-1] != " ":
                    add_char_to_sentence(" ")
                # Reset timer after inserting space to prevent multiple spaces
                last_hand_detected_time = current_time

            # Draw progress bar only if a sign is being held
            if sign_start_time is not None:
                debug_image = draw_sign_progress(debug_image, sign_progress)
        else:  # Logging mode or other modes
            actively_detected_sign_label = None  # No active sign

        # Draw Overlays
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                debug_image = draw_landmarks(debug_image, hand_landmarks)

        debug_image = draw_bounding_rects(debug_image, brects)
        sign_texts_to_draw = (
            [processed_sign_label] * len(brects)
            if current_mode != MODE_LOGGING
            else ["Logging"] * len(brects)
        )
        debug_image = draw_info_texts(
            debug_image, brects, handedness_list_results, sign_texts_to_draw
        )
        cv.putText(
            debug_image,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )
        debug_image = draw_mode_info(debug_image, current_mode)

        # Encode and yield frame
        ret, buffer = cv.imencode(".jpg", debug_image)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# --- Flask Routes ---
@app.route("/")
def index():
    """Render the main HTML page."""
    return render_template(
        "index.html",
        current_mode=current_mode,
        sentence="".join(current_sentence),
        modes=MODE_NAMES,
        log_mode_index=MODE_LOGGING,
        asl_mode_index=MODE_ASL,
        isl_mode_index=MODE_ISL,
    )


@app.route("/video_feed")
def video_feed():
    """Video streaming route."""
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/set_mode", methods=["POST"])
def set_mode():
    """Endpoint to change the recognition mode."""
    global current_mode, current_sign_label, sign_start_time, sign_progress, actively_detected_sign_label
    try:
        new_mode = int(request.json.get("mode"))
        if new_mode in MODE_NAMES:
            if new_mode != current_mode:
                logging.info(f"Switching Mode to {MODE_NAMES[new_mode]}")
                current_mode = new_mode
                # Reset state when mode changes
                clear_sentence_state()  # Clears sentence and active sign
                (
                    current_sign_label,
                    sign_start_time,
                    sign_progress,
                ) = (None, None, 0.0)
            return jsonify(
                {
                    "success": True,
                    "mode": current_mode,
                    "mode_name": MODE_NAMES[current_mode],
                    "sentence": "",
                }
            )
        else:
            return jsonify({"success": False, "error": "Invalid mode"}), 400
    except Exception as e:
        logging.error(f"Error setting mode: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/clear_sentence", methods=["POST"])
def clear_sentence_route():
    """Endpoint to clear the current sentence."""
    clear_sentence_state()
    return jsonify({"success": True, "sentence": ""})


@app.route("/delete_last", methods=["POST"])
def delete_last_route():
    """Endpoint to delete the last character."""
    delete_last_char_state()
    # Return current sentence and potentially the active sign after deletion
    sign_label = actively_detected_sign_label
    img_url = get_sign_image_url(sign_label, current_mode)
    return jsonify(
        {
            "success": True,
            "sentence": "".join(current_sentence),
            "current_sign": sign_label,
            "sign_image_url": img_url,
        }
    )


@app.route("/get_current_status")
def get_current_status():
    """Endpoint to fetch current sentence, active sign label, and image URL."""
    global current_sentence, actively_detected_sign_label, current_mode
    sign_label = actively_detected_sign_label
    img_url = get_sign_image_url(sign_label, current_mode)

    return jsonify(
        {
            "sentence": "".join(current_sentence),
            "current_sign": sign_label,
            "sign_image_url": img_url,
            "mode": current_mode,
        }
    )


def get_sign_image_url(sign_label, mode):
    """Helper to construct the sign image URL."""
    if not sign_label or sign_label == " ":  # No image for null or space
        return None

    # Basic filename cleaning (adjust as needed)
    safe_filename = sign_label.replace(" ", "_").replace("?", "").replace("!", "")

    base_path = None
    filename = None

    if mode == MODE_ASL:
        base_path = "static/ASL_Images"
        filename = f"{safe_filename.upper()}.jpg"  # e.g., A.jpg
    elif mode == MODE_ISL:
        base_path = "static/ISL_Images"
        filename = f"{safe_filename}.jpg"  # e.g., Hello.jpg

    if base_path and filename:
        # Construct path relative to 'static' folder for url_for
        relative_path_parts = [
            os.path.relpath(base_path, app.static_folder),
            filename,
        ]
        # Filter out empty parts (e.g., if base_path is just 'static')
        relative_path_parts = [part for part in relative_path_parts if part]
        relative_path = os.path.join(*relative_path_parts)

        # Check if the actual file exists before returning URL
        full_path = os.path.join(app.static_folder, relative_path)
        # logging.debug(f"Checking for image: {full_path}") # Debug line

        if os.path.exists(full_path):
            try:
                # Normalize path separators for URL
                url = url_for(
                    "static", filename=relative_path.replace(os.sep, "/")
                )
                # logging.debug(f"Image URL generated: {url}") # Debug line
                return url
            except Exception as e:
                logging.error(f"Error generating URL for {filename}: {e}")
                return None
        else:
            logging.warning(f"Sign image file not found: {full_path}")
            return None
    return None


@app.route("/log_data", methods=["POST"])
def log_data_route():
    """Endpoint to log landmark data for the selected symbol."""
    global processed_landmarks_log_single, processed_landmarks_log_combined, logging_language
    if current_mode != MODE_LOGGING:
        return jsonify({"success": False, "error": "Not in logging mode"}), 400

    try:
        log_symbol_index = int(request.json.get("symbol_index", -1))
        log_lang_req = request.json.get("log_lang", LOG_LANG_ASL)
        logging_language = log_lang_req

        if log_symbol_index == -1:
            return (
                jsonify({"success": False, "error": "No symbol index provided"}),
                400,
            )

        target_csv = LOG_CSV_PATHS.get(logging_language)
        if not target_csv:
            return (
                jsonify({
                    "success": False,
                    "error": f"No CSV path for {logging_language}",
                }),
                400,
            )

        landmarks_to_log, expected_size = None, 0
        if logging_language == LOG_LANG_ASL:
            landmarks_to_log, expected_size = (
                processed_landmarks_log_single,
                SINGLE_HAND_VECTOR_SIZE,
            )
        elif logging_language == LOG_LANG_ISL:
            landmarks_to_log, expected_size = (
                processed_landmarks_log_combined,
                COMBINED_VECTOR_SIZE,
            )

        if (
            landmarks_to_log
            and len(landmarks_to_log) == expected_size
            and any(lm != 0.0 for lm in landmarks_to_log)
        ):
            logging_csv(log_symbol_index, landmarks_to_log, target_csv)
            return jsonify(
                {
                    "success": True,
                    "message": f"Logged symbol index {log_symbol_index} for {logging_language}",
                }
            )
        elif not landmarks_to_log or not any(
            lm != 0.0 for lm in landmarks_to_log
        ):
            return (
                jsonify({
                    "success": False,
                    "error": f"No valid landmarks detected for {logging_language}",
                }),
                400,
            )
        else:
            return (
                jsonify({
                    "success": False,
                    "error": f"Incorrect landmark vector size ({len(landmarks_to_log)} vs {expected_size}) for {logging_language}",
                }),
                400,
            )
    except Exception as e:
        logging.error(f"Error logging data: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# --- Translation Routes ---
def perform_translation_thread(text_to_translate, source_lang, target_lang):
    """Background thread function for NLLB translation."""
    global translation_active, translation_result, translator_pipeline
    if translator_pipeline is None:
        translation_result = "[Translation Pipeline Error]"
        translation_active = False
        return

    translation_active = True
    translation_result = "Translating..."
    logging.info(
        f"NLLB: Starting translation for '{text_to_translate}' from {source_lang} to {target_lang}..."
    )
    try:
        result = translator_pipeline(text_to_translate)
        if (
            result
            and isinstance(result, list)
            and len(result) > 0
            and "translation_text" in result[0]
        ):
            translated_text = result[0]["translation_text"]
            logging.info(f"NLLB Translation: '{translated_text}'")
            translation_result = translated_text
        else:
            logging.error(f"NLLB Invalid Response Format: {result}")
            translation_result = "[NLLB Error: Invalid Response]"
    except Exception as e:
        logging.error(f"NLLB Translation failed: {e}")
        translation_result = "[NLLB Error: Check Logs]"
    finally:
        translation_active = False


@app.route("/translate", methods=["POST"])
def translate_route():
    """Endpoint to initiate translation."""
    global translation_active, translation_thread, translation_result, args
    if args.no_translation:
        return (
            jsonify({
                "success": False,
                "error": "Translation disabled by command line argument",
            }),
            400,
        )
    if not NLLB_PREREQUISITES_AVAILABLE:
        return (
            jsonify({
                "success": False,
                "error": "Translation prerequisites not installed",
            }),
            400,
        )
    if translator_pipeline is None:
        return (
            jsonify({
                "success": False,
                "error": "Translation pipeline not initialized or failed to load",
            }),
            500,
        )

    if translation_active:
        if translation_thread and translation_thread.is_alive():
            return (
                jsonify({
                    "success": False,
                    "error": "Translation already in progress",
                }),
                400,
            )
        else:
            # Allow new request if thread died unexpectedly
            logging.warning(
                "Previous translation thread inactive, allowing new request."
            )
            translation_active = False

    source_text = request.json.get("text", "").strip()
    if not source_text:
        return (
            jsonify({
                "success": False,
                "error": "No text provided for translation",
            }),
            400,
        )

    translation_thread = threading.Thread(
        target=perform_translation_thread,
        args=(source_text, SOURCE_TRANSLATION_LANG, TARGET_TRANSLATION_LANG),
        daemon=True,
    )
    translation_thread.start()
    translation_result = "Translation started..."
    return jsonify({"success": True, "message": "Translation started"})


@app.route("/get_translation_status")
def get_translation_status():
    """Endpoint to poll translation status/result."""
    global translation_active, translation_thread, translation_result
    # Check if thread finished unexpectedly
    if translation_active and translation_thread and not translation_thread.is_alive():
        logging.warning(
            "Translation thread died unexpectedly. Resetting active flag."
        )
        translation_active = False
    return jsonify({"active": translation_active, "result": translation_result})


# --- Main Execution ---
if __name__ == "__main__":
    if not MEDIAPIPE_AVAILABLE:
        logging.critical("Exiting due to missing prerequisite: mediapipe.")
        sys.exit(1)
    if not UTILS_AVAILABLE:
        logging.warning(
            "Local utils (CvFpsCalc) not found. FPS calculation disabled."
        )
    if KeyPointClassifier is None:
        logging.warning(
            "KeyPointClassifier not available. Gesture classification disabled."
        )

    initialize_app()
    logging.info("Flask app initialized. Starting server...")
    logging.info(
        "Access the application at http://127.0.0.1:5000 or http://<your-ip>:5000"
    )
    # Run with debug=False to prevent auto-reload issues with NLLB/models
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)