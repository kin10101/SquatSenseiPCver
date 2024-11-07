import os
import shutil

def move_files(source_folder, destination_folder, file_extensions, include_subfolders=False):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Walk through the directory
    for root, _, files in os.walk(source_folder):
        for filename in files:
            if any(filename.lower().endswith(ext.lower()) for ext in file_extensions):
                # Construct full file path
                source_file = os.path.join(root, filename)
                destination_file = os.path.join(destination_folder, filename)

                # Move the file and overwrite if it exists
                shutil.move(source_file, destination_file, copy_function=shutil.copy2)
                print(f'Moved: {filename}')

        # If not including subfolders, break after the first iteration
        if not include_subfolders:
            break

if __name__ == '__main__':
    #source_folder = r"C:\Users\ijoac\Downloads"
    source_folder = r"F:\transfer from laptop\Kin 2023 laptop Backup\Documents"
    destination_folder = r"C:\Users\ijoac\Downloads\PDFfiles"
    file_extensions = ['.pdf']  # Add the file extensions you want to move
    include_subfolders = True  # Set to True to include subfolders
    move_files(source_folder, destination_folder, file_extensions, include_subfolders)