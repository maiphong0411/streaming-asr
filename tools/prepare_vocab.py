import argparse

def get_unique_words(input_file_path, output_file_path):
    # Open input file in read mode
    with open(input_file_path, 'r') as input_file:
        # Read the contents of the input file
        input_text = input_file.read()

    # Extract words from input text
    words = input_text.split()

    # Convert words to lowercase and remove punctuation (if needed)
    words = [word.lower() for word in words]

    # Get unique words using a set
    unique_words = set(words)

    # Open output file in write mode
    with open(output_file_path, 'w') as output_file:
        # Write unique words to the output file
        for word in unique_words:
            output_file.write(word + '\n')

if __name__ == "__main__":
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Extract unique words from input text file')

    # Add arguments
    parser.add_argument('input_file', type=str, help='Path to the input text file')
    parser.add_argument('output_file', type=str, help='Path to the output text file')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function to get unique words
    get_unique_words(args.input_file, args.output_file)
