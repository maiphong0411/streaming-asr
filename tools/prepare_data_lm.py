# source_path = "/vinbrain/phongmt/wenet/wenet/examples/foo/s0/data/train/text"

# target_path = "text_only.txt"

import argparse

def main(source_path, target_path):
    with open(source_path, "r", encoding="utf-8") as f_input:
        with open(target_path, "w", encoding="utf-8") as f_output:
            for line in f_input:
                # Split the line into ID and text, and write only the text to the output file
                text = " ".join(line.split()[1:])
                if len(text) < 20:
                    continue
                if len(text) > 10000:
                    continue
                f_output.write(text + "\n")
    print("Text without IDs written to", target_path)
    

if __name__ == '__main__':
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='get text only')
    # Add arguments
    parser.add_argument('--source_file', type=str, help='source file')
    parser.add_argument('--target_file', type=str, help='target file')
    
    args = parser.parse_args()
    main(args.source_file, args.target_file)