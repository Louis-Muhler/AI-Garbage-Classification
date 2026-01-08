import splitfolders
import os

def split_data(input_dir, output_dir, ratio=(0.8, 0.2)):
    """
    Splits the dataset into train and val sets.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    print(f"Splitting files from '{input_dir}' to '{output_dir}'...")
    # ratio: (train, val) - no test split
    splitfolders.ratio(input_dir, output=output_dir, seed=1337, ratio=ratio, group_prefix=None, move=False)
    print("Done!")
