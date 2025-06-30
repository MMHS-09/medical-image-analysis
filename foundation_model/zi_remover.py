import os

def remove_zone_identifier_files(root_path):
    """
    Removes Zone.Identifier files from all directories and subdirectories under the given root_path.
    """
    removed_files_count = 0
    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith("Zone.Identifier"):
                full_path = os.path.join(dirpath, file)
                try:
                    os.remove(full_path)
                    print(f"Removed: {full_path}")
                    removed_files_count += 1
                except Exception as e:
                    print(f"Error removing {full_path}: {e}")

    print(f"\nTotal Zone.Identifier files removed: {removed_files_count}")

root_directory = r"/home/jobayer/research/mhs/medical-image-analysis/fmodel/"
remove_zone_identifier_files(root_directory)