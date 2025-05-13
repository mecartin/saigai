import argparse
import csv
import copy
import itertools
import os
import sys

import cv2 as cv
import mediapipe as mp
import numpy as np

# --- Argument Parser ---
def get_args():
    """Parses command line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks from an image dataset using fixed labels (a-z, 0-9)." # Updated description
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the root directory of the image dataset. Expects subdirectories named a-z or 0-9.", # Updated help text
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save the output CSV file containing landmarks (label_index, landmark_data...).",
    )
    parser.add_argument(
        "--max_num_hands",
        type=int,
        default=1,
        help="Maximum number of hands to detect per image.",
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5, # Adjusted default for static images
        help="Minimum detection confidence threshold for MediaPipe Hands.",
    )
    args = parser.parse_args()
    return args

# --- Landmark Pre-processing Function ---
def pre_process_landmarks_normalized(landmark_list_normalized):
    """
    Normalizes landmarks relative to the wrist (point 0) using normalized coordinates.

    Args:
        landmark_list_normalized: A list of MediaPipe normalized landmark objects
                                   (containing x, y, z).

    Returns:
        A list of pre-processed, flattened landmark coordinates (x1, y1, x2, y2,...),
        normalized relative to the wrist and scaled by max deviation. Returns None if input is invalid.
    """
    if not landmark_list_normalized or len(landmark_list_normalized) < 21:
        return None

    # Convert to a list of [x, y] coordinates first
    temp_landmark_coords = [[lm.x, lm.y] for lm in landmark_list_normalized]

    # Copy to avoid modifying original list
    temp_processed_coords = copy.deepcopy(temp_landmark_coords)

    # Calculate relative coordinates based on wrist (index 0)
    # The wrist point itself will become (0.0, 0.0)
    base_x, base_y = temp_processed_coords[0]
    for i in range(len(temp_processed_coords)):
        temp_processed_coords[i][0] -= base_x
        temp_processed_coords[i][1] -= base_y

    # Flatten the list of relative coordinates
    flat_landmark_list = list(itertools.chain.from_iterable(temp_processed_coords))

    # Normalize by the maximum absolute value to scale between -1 and 1
    # This ensures consistency even if the initial relative values are small
    max_abs_value = max(map(abs, flat_landmark_list)) if flat_landmark_list else 1.0
    if max_abs_value == 0:
        # Handle case where all points are identical to the base (wrist)
        # Return list of zeros matching the expected length (21 * 2 = 42)
        max_abs_value = 1.0 # Avoid division by zero, effectively returns zeros


    normalized_list = [val / max_abs_value for val in flat_landmark_list]

    # The first two values (wrist x, y) should be 0.0 after this process
    # print(f"Debug Preprocess Output Sample: {normalized_list[:4]}") # Uncomment for debugging

    return normalized_list


# --- Main Data Extraction Function ---
def extract_landmark_data(args):
    """Finds images, extracts landmarks, preprocesses, and prepares for saving."""
    dataset_dir = args.dataset_dir
    output_csv = args.output_csv
    max_num_hands = args.max_num_hands
    min_detection_confidence = args.min_detection_confidence

    # --- Define the fixed labels (lowercase a-z, 0-9) and create the mapping ---
    labels = list('abcdefghijklmnopqrstuvwxyz') + [str(i) for i in range(10)] # USE LOWERCASE
    label_to_index = {label: index for index, label in enumerate(labels)}
    print(f"Using fixed labels (Indices 0-{len(labels)-1}): {', '.join(labels)}")
    print(f"Expecting subdirectories in '{dataset_dir}' named accordingly (a-z, 0-9).") # Updated print
    # --- End Label Definition ---

    print("\nInitializing MediaPipe Hands...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True, # Crucial for processing static images
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
    )

    print(f"Scanning dataset directory: {dataset_dir}")
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_landmark_data = []
    processed_image_count = 0
    skipped_image_count = 0
    skipped_dirs = set()

    # --- Iterate through items in dataset directory ---
    try:
        items = sorted(os.listdir(dataset_dir))
    except FileNotFoundError:
        print(f"Error: Dataset directory not found: {dataset_dir}")
        hands.close()
        sys.exit(1)

    # --- Process only subdirectories that match the defined labels ---
    for item_name in items:
        item_path = os.path.join(dataset_dir, item_name)
        if not os.path.isdir(item_path):
            continue # Skip files, process only directories

        label_name = item_name # Directory name is the potential label

        # --- Check if directory name is a valid label (now checks a-z, 0-9) ---
        if label_name in label_to_index:
            label_index = label_to_index[label_name]
            image_files = [
                f for f in os.listdir(item_path)
                if os.path.splitext(f)[1].lower() in supported_extensions
            ]

            if not image_files:
                 print(f"\nWarning: No images found in directory '{label_name}'. Skipping.")
                 continue

            print(f"\nProcessing label '{label_name}' (Index: {label_index}) - {len(image_files)} images found...")

            # --- Iterate through images in the valid subdirectory ---
            for i, image_name in enumerate(image_files):
                image_path = os.path.join(item_path, image_name)
                # Use end='\r' for progress update on the same line
                print(f"  Processing image {i+1}/{len(image_files)}: {image_name}", end='\r')

                # Read Image
                image = cv.imread(image_path)
                if image is None:
                    print(f"\n  Warning: Could not read image {image_path}. Skipping.")
                    skipped_image_count += 1
                    continue

                # Process with MediaPipe
                image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False # Improve performance
                results = hands.process(image_rgb)
                image_rgb.flags.writeable = True

                # --- Extract and Preprocess Landmarks ---
                if results.multi_hand_landmarks:
                    # Process the first detected hand
                    hand_landmarks = results.multi_hand_landmarks[0]

                    # Preprocess using the normalized coordinates directly
                    processed_landmarks = pre_process_landmarks_normalized(hand_landmarks.landmark)

                    if processed_landmarks:
                        # Add label index and landmarks to our data list
                        all_landmark_data.append([label_index] + processed_landmarks)
                        processed_image_count += 1
                    else:
                        print(f"\n  Warning: Could not preprocess landmarks for {image_path}. Skipping.")
                        skipped_image_count += 1
                else:
                    # No hand detected
                    # print(f"\n  Warning: No hand detected in {image_path}. Skipping.") # Optional: can be verbose
                    skipped_image_count += 1
            # Add a newline after processing all images for a label
            print(f"\nFinished processing label '{label_name}'.")
        else:
            # Directory name does not match any defined label
            if label_name not in skipped_dirs:
                 print(f"\nWarning: Directory '{label_name}' does not match any defined label (a-z, 0-9). Skipping.") # Updated warning
                 skipped_dirs.add(label_name)


    hands.close()
    print("\n--- Processing Summary ---")
    print(f"Total images processed successfully: {processed_image_count}")
    print(f"Total images skipped (no hand detected or error): {skipped_image_count}")
    print(f"Total valid label directories processed: {len(label_to_index) - len(skipped_dirs)}")
    if skipped_dirs:
        print(f"Skipped directories (not in a-z, 0-9): {', '.join(sorted(list(skipped_dirs)))}") # Updated print


    # --- Save Data to CSV ---
    if not all_landmark_data:
        print("\nError: No landmark data was extracted. CSV file will not be created.")
        return

    print(f"\nSaving extracted landmark data to: {output_csv}")
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        # Write to CSV
        # Expected columns: label_index, rel_norm_x0, rel_norm_y0, rel_norm_x1, rel_norm_y1, ... (43 columns total)
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(all_landmark_data)
        print("CSV file saved successfully.")

        # No longer need to save a separate label map, as it's fixed in the code

    except Exception as e:
        print(f"Error saving CSV file '{output_csv}': {e}")


# --- Script Entry Point ---
if __name__ == "__main__":
    args = get_args()
    extract_landmark_data(args)
    print("All tasks completed successfully.")
    print("Exiting...")
    sys.exit(0)
