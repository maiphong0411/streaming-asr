#!/bin/bash


lexicon=/vinbrain/phongmt/wenet/wenet/examples/foo/s0/data/local/dict/lexicon.txt
# Define values for sys.argv variables
sys_argv1="/vinbrain/phongmt/wenet/wenet/examples/foo/s0/data/lang_char/train_bpe14336_units.txt"
sys_argv2="/vinbrain/phongmt/wenet/wenet/examples/foo/s0/vocab.txt"
sys_argv3="lexicon.txt"
sys_argv4="/vinbrain/phongmt/wenet/wenet/examples/foo/s0/data/lang_char/train_bpe14336.model"

# Run the Python script with provided arguments
python "prepare_dict2.py" "$sys_argv1" "$sys_argv2" "$sys_argv3" "$sys_argv4"


if [ -e "$lexicon" ]; then
    echo "File $lexicon exists. Removing..."
    rm "$lexicon"
    echo "File removed successfully."
else
    echo "File $lexicon does not exist."
fi

mv "lexicon.txt" "/vinbrain/phongmt/wenet/wenet/examples/foo/s0/data/local/dict/"