import shutil
import pandas as pd
import os
import argparse

def make_same_id(path_root, obj_list=['test'], files_order=4):
    """
    Generates and returns a list of formatted IDs based on the length of a DataFrame loaded from a CSV file.

    Parameters:
    - path_root (str): The root path where the CSV file is located.
    - obj_list (list, optional): A list of object names. Defaults to ['test'].
    - files_order (int, optional): The number of files to generate. Defaults to 4.

    Returns:
    - dict: A dictionary where keys are object names and values are lists of formatted IDs.

    Example:
    >>> make_same_id('/path/to/data', ['train', 'test'])
    {'train': ['0001', '0002', ..., 'length_of_dataframe'],
     'test': ['0001', '0002', ..., 'length_of_dataframe']}
    """
    result_dict = {}

    # create data folder
    data_folder = "data/"
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)
        print("Folder 'data' deleted successfully.")
    else:
        print("Folder 'data' does not exist.")
    print("Creating 'data' folder")
    os.makedirs(data_folder)

    for obj in obj_list:
        # make a new folder
        obj_folder = os.path.join(data_folder, obj)
        if os.path.exists(obj_folder):
            shutil.rmtree(obj_folder)
            print(f"Folder '{obj_folder}' deleted successfully.")
        else:
            print(f"Folder '{obj_folder}' does not exist.")
        print(f"Creating {obj_folder} folder")
        os.makedirs(obj_folder)

        # join path and file
        path = os.path.join(path_root, obj + '.csv')
        if os.path.exists(path):
            print(f"Found {obj}.csv in folder")
        else:
            print(path)
            print(f"Nothing with the name {obj} exists in the folder.")
            continue

        # load df
        df = pd.read_csv(path)

        # make same length
        length_of_df = len(df)
        print(f"Length of dataframe: {length_of_df}")
        max_length = len(str(length_of_df))
        id_range = list(range(1, length_of_df + 1))
        formatted_numbers = [str(num).zfill(max_length) for num in id_range]

        # write file
        def write_file(filename, content_func):
            file_path = os.path.join(obj_folder, filename)
            if os.path.exists(file_path):
                print(f"File {filename} already exists. Removing the existing file.")
                os.remove(file_path)
            with open(file_path, 'w', encoding='utf-8') as file:
                for i in range(len(formatted_numbers)):
                    line = content_func(i)
                    file.write(line)
            print(f"Write to {filename}")

        write_file("wav.scp", lambda i: f"{formatted_numbers[i]} {df.iloc[i]['wav']}\n")
        write_file("text", lambda i: f"{formatted_numbers[i]} {df.iloc[i]['wrd'].decode('utf-8') if isinstance(df.iloc[i]['wrd'], bytes) else str(df.iloc[i]['wrd'])}\n")

        if files_order == 2:
            result_dict[obj] = formatted_numbers
        else:
            write_file("utt2spk", lambda i: f"{formatted_numbers[i]} dummy\n")
            write_file("spk2utt", lambda i: f"dummy {formatted_numbers[i]}\n")
            result_dict[obj] = formatted_numbers

#     return result_dict

def main():
    parser = argparse.ArgumentParser(description="Process CSV file and generate formatted IDs.")
    parser.add_argument("path_root", help="The root path where the CSV file is located.")
    parser.add_argument("--obj", nargs='+', default=["test"], help="A list of object names. Defaults to ['test'].")
    parser.add_argument("--files_order", type=int, default=4, help="The number of files to generate. Defaults to 4.")
    args = parser.parse_args()

    make_same_id(args.path_root, args.obj, args.files_order)

if __name__ == "__main__":
    main()
