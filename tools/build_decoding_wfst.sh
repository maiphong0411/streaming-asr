#!/bin/bash
# build runtime
# run in terminal 
# cd runtime/libtorch
# mkdir build && cd build && cmake -DGRAPH_TOOLS=ON .. && cmake --build .
lm_available=/vinbrain/chuongct98/ngram_lms/ct_cxr_vlsp_pelvis_gmt_hbclean_f0.arpa
lm=data/local/lm

if [ -f $lm_available ]; then
   echo "Copy a available LM"
   cp $lm_available $lm/lm.arpa
else
   echo "No language model"
fi
lexicon=data/local/dict/lexicon.txt
mkdir -p $lm
mkdir -p data/local/dict

n_gram=5
root_path="/vinbrain/phongmt/wenet/wenet/examples/foo/s0"
# prepare data for lm
source_path="/vinbrain/phongmt/wenet/wenet/examples/foo/s0/data/train/text"
target_path="text_train_lm.txt"
tmp="/vinbrain/phongmt/wenet/wenet/examples/foo/s0/text_train_lm.txt"

if [ -f "$target_path" ]; then
    echo "File exists. Skip the step"
else
    echo "File does not exist"
    # run script to extract text
    python tools/prepare_data_lm.py --source_file $source_path --target_file $lm/$target_path
fi


#  train lm

file_path="/vinbrain/chuongct98/tools/kenlm/kenlm/build/bin/lmplz"
lm_path="lm.arpa"

if [ -f $lm/$lm_path ]; then
    echo "$lm_path existed. Skip building LM"
else
    echo "Building language model"
    /vinbrain/chuongct98/tools/kenlm/kenlm/build/bin/lmplz -o $n_gram -S 2G <$root_path/$target_path> $lm/$lm_path
fi

lexicon=data/local/dict/lexicon.txt

# Define values for sys.argv variables
unit_file=data/lang_char/train_bpe12000_units.txt
vocab=$lm/vocab.txt
output_lexicon="lexicon.txt"
bpe_model=data/lang_char/train_bpe12000.model

if [ -f  "$vocab" ]; then
    echo "Skip the step"
else
    echo "Building vocab for wfst"
    python tools/prepare_vocab.py $lm/$target_path $vocab
fi
# Run the Python script with provided arguments
python tools/fst/prepare_dict_bpe.py "$unit_file" "$vocab" "$lexicon" "$bpe_model"

# mv "lexicon.txt" "/vinbrain/phongmt/wenet/wenet/examples/foo/s0/data/local/dict/"

# build lexicon

# build decoding TLG

# decoding (test)