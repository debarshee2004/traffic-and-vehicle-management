import os
import shutil
from datetime import datetime
from pathlib import Path


SOURCE_FOLDER = "../../data/processed/frames"  # Default source folder
DEST_FOLDER = "../../server/demo_test_ip_camera/images"  # Default destination folder


def rename_and_move_files():
    """
    Rename files from source folder with camera ID and timestamp format,
    then move them to destination folder.
    """

    # Get user inputs
    try:
        num_files = int(input("Enter the number of files to process: "))
        if num_files <= 0:
            print("Number of files must be positive!")
            return
    except ValueError:
        print("Please enter a valid number!")
        return

    source_folder = SOURCE_FOLDER
    dest_folder = DEST_FOLDER
    camera_id = input("Enter the camera ID: ").strip()

    # Validate source folder
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist!")
        return

    if not os.path.isdir(source_folder):
        print(f"'{source_folder}' is not a directory!")
        return

    # Create destination folder if it doesn't exist
    try:
        os.makedirs(dest_folder, exist_ok=True)
        print(f"Destination folder created/verified: {dest_folder}")
    except Exception as e:
        print(f"Error creating destination folder: {e}")
        return

    # Get list of files from source folder
    try:
        all_files = [
            f
            for f in os.listdir(source_folder)
            if os.path.isfile(os.path.join(source_folder, f))
        ]

        if len(all_files) == 0:
            print("No files found in the source folder!")
            return

        if len(all_files) < num_files:
            print(f"Only {len(all_files)} files available, but {num_files} requested.")
            choice = input("Process all available files? (y/n): ").lower()
            if choice != "y":
                return
            num_files = len(all_files)

        # Sort files to ensure consistent processing order
        all_files.sort()
        files_to_process = all_files[:num_files]

    except Exception as e:
        print(f"Error reading source folder: {e}")
        return

    # Process files
    processed_count = 0
    errors = []

    print(f"\nProcessing {len(files_to_process)} files...")
    print("-" * 50)

    for i, filename in enumerate(files_to_process, 1):
        try:
            # Get file extension
            file_path = os.path.join(source_folder, filename)
            file_ext = Path(filename).suffix

            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
                :-3
            ]  # Include milliseconds

            # Create new filename: {CAMERA_ID}-{TIMESTAMP}{EXTENSION}
            new_filename = f"{camera_id}-{timestamp}{file_ext}"
            new_file_path = os.path.join(dest_folder, new_filename)

            # Copy file to destination with new name
            shutil.copy2(file_path, new_file_path)

            print(f"{i:2d}. {filename} -> {new_filename}")
            processed_count += 1

            # Small delay to ensure unique timestamps
            import time

            time.sleep(0.001)

        except Exception as e:
            error_msg = f"Error processing '{filename}': {e}"
            print(f"{i:2d}. {error_msg}")
            errors.append(error_msg)

    # Summary
    print("-" * 50)
    print(f"Processing complete!")
    print(f"Successfully processed: {processed_count}/{len(files_to_process)} files")

    if errors:
        print(f"Errors encountered: {len(errors)}")
        print("\nError details:")
        for error in errors:
            print(f"  - {error}")

    print(f"\nFiles saved to: {dest_folder}")


def list_files_preview():
    """
    Preview files in a folder before processing
    """
    folder_path = input("Enter folder path to preview: ").strip()

    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist!")
        return

    try:
        files = [
            f
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        ]

        if not files:
            print("No files found in the folder!")
            return

        files.sort()
        print(f"\nFound {len(files)} files:")
        print("-" * 40)

        for i, filename in enumerate(files, 1):
            print(f"{i:2d}. {filename}")

        print("-" * 40)

    except Exception as e:
        print(f"Error reading folder: {e}")


def main():
    """
    Main function with menu system
    """
    print("=" * 60)
    print("FILE RENAMER - Camera ID & Timestamp Format")
    print("=" * 60)

    while True:
        print("\nOptions:")
        print("1. Rename and move files")
        print("2. Preview files in folder")
        print("3. Exit")

        choice = input("\nSelect an option (1-3): ").strip()

        if choice == "1":
            rename_and_move_files()
        elif choice == "2":
            list_files_preview()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()
