import sys
from uos_depth_est_core import (
    kp_recognition_gradient,
    )

def process_file(file_path):
    # Read the Excel file
    try:
        df, l_pos = kp_recognition_gradient(file_path)
        # Process the data here
        # ...
        # ...
        print(l_pos)
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uos_depth_estimation.py <file_path>")
    else:
        file_path = sys.argv[1]
        process_file(file_path)