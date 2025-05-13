import argparse
import csv
import copy
import itertools
import os
import sys

import cv2 as cv
import mediapipe as mp
import numpy as np

# --- Constants ---
NUM_LANDMARKS_SINGLE_HAND = 21
NUM_COORDS = 2
SINGLE_HAND_VECTOR_SIZE = NUM_LANDMARKS_SINGLE_HAND * NUM_COORDS # Should be 42
ZERO_VECTOR_SINGLE_HAND = [0.0] * SINGLE_HAND_VECTOR_SIZE
COMBINED_VECTOR_SIZE = SINGLE_HAND_VECTOR_SIZE * 2 # Should be 84

# --- Argument Parser ---
def get_args():
    """Parses command line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe 2-handed landmarks from an image dataset using fixed labels (a-z, 0-9)."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the root directory of the image dataset. Expects subdirectories named a-z, A-Z, or 0-9.", # Updated help
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save the output CSV file containing landmarks (label_index, left_lm..., right_lm...).",
    )
    parser.add_argument(
        "--max_num_hands",
        type=int,
        default=2,
        help="Maximum number of hands to detect per image.",
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence threshold for MediaPipe Hands.",
    )
    args = parser.parse_args()
    if args.max_num_hands != 2:
        print("Warning: This script is designed for max_num_hands=2. Setting it to 2.")
        args.max_num_hands = 2
    return args

# --- Landmark Pre-processing Function ---
def pre_process_landmarks_normalized(landmark_list_normalized):
    """
    Normalizes landmarks for a SINGLE hand relative to its wrist using normalized coordinates.
    Returns a list of 42 zeros if input is invalid.
    """
    if not landmark_list_normalized or len(landmark_list_normalized) != NUM_LANDMARKS_SINGLE_HAND:
        return ZERO_VECTOR_SINGLE_HAND

    temp_landmark_coords = [[lm.x, lm.y] for lm in landmark_list_normalized]
    temp_processed_coords = copy.deepcopy(temp_landmark_coords)

    base_x, base_y = temp_processed_coords[0]
    for i in range(len(temp_processed_coords)):
        temp_processed_coords[i][0] -= base_x
        temp_processed_coords[i][1] -= base_y

    flat_landmark_list = list(itertools.chain.from_iterable(temp_processed_coords))
    max_abs_value = max(map(abs, flat_landmark_list)) if flat_landmark_list else 1.0
    if max_abs_value == 0:
        return ZERO_VECTOR_SINGLE_HAND

    normalized_list = [val / max_abs_value for val in flat_landmark_list]
    return normalized_list


# --- Main Data Extraction Function ---
def extract_landmark_data(args):
    """Finds images, extracts landmarks for two hands, preprocesses, and prepares for saving."""
    dataset_dir = args.dataset_dir
    output_csv = args.output_csv
    max_num_hands = args.max_num_hands
    min_detection_confidence = args.min_detection_confidence

    # --- Define the fixed labels (lowercase a-z, 0-9) and create the mapping ---
    # NOTE: We define the canonical labels as lowercase internally
    labels = list('abcdefghijklmnopqrstuvwxyz') + [str(i) for i in range(10)]
    label_to_index = {label: index for index, label in enumerate(labels)}
    print(f"Using fixed labels (Indices 0-{len(labels)-1}): {', '.join(labels)}")
    print(f"Expecting subdirectories in '{dataset_dir}' named a-z, A-Z, or 0-9.") # Updated print

    print("\nInitializing MediaPipe Hands...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
    )

    print(f"Scanning dataset directory: {dataset_dir}")
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_landmark_data = []
    processed_image_count = 0
    logged_image_count = 0
    skipped_image_count = 0
    skipped_dirs = set()

    try:
        items = sorted(os.listdir(dataset_dir))
    except FileNotFoundError:
        print(f"Error: Dataset directory not found: {dataset_dir}")
        hands.close(); sys.exit(1)

    for item_name in items:
        item_path = os.path.join(dataset_dir, item_name)
        if not os.path.isdir(item_path): continue

        label_name = item_name
        # --- MODIFIED: Convert directory name to lowercase for lookup ---
        label_lookup = label_name.lower()
        # --- End Modification ---

        # --- Check if lowercase directory name is a valid label ---
        if label_lookup in label_to_index:
            label_index = label_to_index[label_lookup] # Use the index corresponding to the lowercase label
            image_files = [f for f in os.listdir(item_path) if os.path.splitext(f)[1].lower() in supported_extensions]

            if not image_files:
                print(f"\nWarning: No images found in directory '{label_name}'. Skipping.")
                continue

            print(f"\nProcessing label '{label_name}' -> Index: {label_index} ({len(image_files)} images found)...") # Show original name and index

            for i, image_name in enumerate(image_files):
                image_path = os.path.join(item_path, image_name)
                print(f"  Processing image {i+1}/{len(image_files)}: {image_name}", end='\r')

                image = cv.imread(image_path)
                if image is None:
                    print(f"\n  Warning: Could not read image {image_path}. Skipping.")
                    skipped_image_count += 1; continue

                image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = hands.process(image_rgb)
                image_rgb.flags.writeable = True

                left_hand_landmarks_norm = ZERO_VECTOR_SINGLE_HAND
                right_hand_landmarks_norm = ZERO_VECTOR_SINGLE_HAND
                num_hands_detected = 0

                if results.multi_hand_landmarks and results.multi_handedness:
                    num_hands_detected = len(results.multi_hand_landmarks)
                    for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        if hand_index >= len(results.multi_handedness): continue
                        handedness = results.multi_handedness[hand_index]
                        hand_label = handedness.classification[0].label

                        processed_single_hand = pre_process_landmarks_normalized(hand_landmarks.landmark)
                        if hand_label == 'Left':
                            left_hand_landmarks_norm = processed_single_hand
                        elif hand_label == 'Right':
                            right_hand_landmarks_norm = processed_single_hand

                combined_landmarks = left_hand_landmarks_norm + right_hand_landmarks_norm

                if any(lm != 0.0 for lm in combined_landmarks):
                    all_landmark_data.append([label_index] + combined_landmarks) # Save with the correct index
                    logged_image_count += 1
                else:
                    skipped_image_count += 1
                processed_image_count += 1

            print(f"\nFinished processing label '{label_name}'. Logged {logged_image_count}/{processed_image_count} images for this label.")
            processed_image_count = 0
            logged_image_count = 0

        else:
            if label_name not in skipped_dirs:
                print(f"\nWarning: Directory '{label_name}' (lowercase '{label_lookup}') does not match any defined label (a-z, 0-9). Skipping.") # Updated warning
                skipped_dirs.add(label_name)

    hands.close()
    total_processed = sum(len(files) for _, _, files in os.walk(dataset_dir) if any(f.lower().endswith(tuple(supported_extensions)) for f in files))
    total_logged = len(all_landmark_data)

    print("\n--- Processing Summary ---")
    print(f"Total images scanned (approx): {total_processed}")
    print(f"Total images where landmarks were logged: {total_logged}")
    print(f"Total images skipped (no hand detected or error): {skipped_image_count}")
    print(f"Total valid label directories processed: {len(label_to_index) - len(skipped_dirs)}")
    if skipped_dirs:
        print(f"Skipped directories (not in a-z, A-Z, 0-9): {', '.join(sorted(list(skipped_dirs)))}") # Updated print

    if not all_landmark_data:
        print("\nError: No landmark data was extracted. CSV file will not be created.")
        return

    print(f"\nSaving extracted landmark data to: {output_csv}")
    try:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(all_landmark_data)
        print("CSV file saved successfully.")
    except Exception as e:
        print(f"Error saving CSV file '{output_csv}': {e}")

# --- Script Entry Point ---
if __name__ == "__main__":
    args = get_args()
    extract_landmark_data(args)
    print("\nAll tasks completed.")
    sys.exit(0)
