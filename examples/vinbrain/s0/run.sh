#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.

. ./path.sh || exit 1;

# Automatically detect number of gpus
if command -v nvidia-smi &> /dev/null; then
  num_gpus=1
  gpu_list="2"
else
  num_gpus=-1
  gpu_list="-1"
fi
# You can also manually specify CUDA_VISIBLE_DEVICES
# if you don't want to utilize all available GPU resources.
export CUDA_VISIBLE_DEVICES="${gpu_list}"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5

# Specify the path to your data folder
data_folder="/vinbrain/phongmt/wenet/wenet/examples/vinbrain/s0/data"

# List only the folder names within the data folder
data_list=$(find "$data_folder" -maxdepth 1 -type d -exec basename {} \; | grep -E 'test|train|valid')

# Print the list of folders
echo "List of folders in $data_folder:"
echo "$data_list"

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1
job_id=2023

# data
data_url=www.openslr.org/resources/12
# use your own data path
datadir=data
# wav data dir
wave_data=data
data_type=raw
# Optional train_config
# 1. conf/train_transformer_large.yaml: Standard transformer
train_config=conf/train_u2++_conformer.yaml
checkpoint=
num_workers=4
do_delta=false

dir=exp/sp_spec_aug
tensorboard_dir=tensorboard

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
# maybe you can try to adjust it if you can not get close results as README.md
average_num=10
decode_modes="attention_rescoring ctc_greedy_search ctc_prefix_beam_search attention"

# bpemode (unigram or bpe)
nbpe=12000
bpemode=bpe

set -e
set -u
set -o pipefail

train_set=train
dev_set=valid
test_set=test
recog_set="test"

train_csv="/vinbrain/phongmt/wenet/wenet/examples/foo/s0/tools_wenet/train.csv"
test_csv="/vinbrain/chuongct98/audio_datasets/benchmark_dataset/public30_f0_codemix_splits/test.csv"
dev_csv="/vinbrain/chuongct98/audio_datasets/benchmark_dataset/public30_f0_codemix_splits/valid.csv"

train_engine=torch_ddp

deepspeed_config=../../aishell/s0/conf/ds_stage2.json
deepspeed_save_states="model_only"

. tools/parse_options.sh || exit 1;


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  ### Task dependent. You have to design training and dev sets by yourself.
  ### But you can utilize Kaldi recipes in most cases
  echo "stage 1: Convert data"
  mkdir -p data_raw/csv
  cp $train_csv data_raw/csv
  cp $test_csv data_raw/csv
  cp $dev_csv data_raw/csv

  python tools/convert_kaldi_format.py data_raw/csv --obj "train" "test" "valid" --files_order 2

  tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
    --in_scp $wave_data/$train_set/wav.scp \
    --out_cmvn $wave_data/$train_set/global_cmvn

fi


dict=$wave_data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=$wave_data/lang_char/${train_set}_${bpemode}${nbpe}
# echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  ### Task dependent. You have to check non-linguistic symbols used in the corpus.
  echo "stage 2: Dictionary and Json Data Preparation"
  mkdir -p data/lang_char/

  echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
  echo "<unk> 1" >> ${dict} # <unk> must be 1
  echo "<sos/eos> 2" >> $dict # <eos>

  # we borrowed these code and scripts which are related bpe from ESPnet.
  cut -f 2- -d" " $wave_data/${train_set}/text > $wave_data/lang_char/input.txt
  tools/spm_train --input=$wave_data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --split_digits=true --allow_whitespace_only_pieces=true --byte_fallback=true --normalization_rule_name="identity" --character_coverage=1.0 
  tools/spm_encode --model=${bpemodel}.model --output_format=piece < $wave_data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+2}' >> ${dict}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Prepare wenet required data
  echo "Prepare data, prepare required format"
  for x in $dev_set $test_set $train_set ; do
    tools/make_raw_list.py $wave_data/$x/wav.scp $wave_data/$x/text \
        $wave_data/$x/data.list
  done

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Training
  mkdir -p $dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  # train.py will write $train_config to $dir/train.yaml with model input
  # and output dimension, train.yaml will be used for inference or model
  # export later
  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus --rdzv_endpoint=$HOST_NODE_ADDR \
           --rdzv_id=$job_id --rdzv_backend="c10d" \
    wenet/bin/train.py \
      --train_engine ${train_engine} \
      --config $train_config \
      --data_type ${data_type} \
      --train_data $wave_data/train/data.list \
      --cv_data $wave_data/valid/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --pin_memory \
      --deepspeed_config ${deepspeed_config} \
      --deepspeed.save_states ${deepspeed_save_states}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  # TODO, Add model average here
  mkdir -p $dir/test_turn3
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num 10 \
      --val_best
  fi
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  decoding_chunk_size=8
  ctc_weight=0.5
  for test in $recog_set; do
    result_dir=$dir/test_turn3
    python wenet/bin/recognize.py --gpu 1 \
      --modes $decode_modes \
      --config $dir/train.yaml \
      --data_type raw \
      --test_data $wave_data/$test/data.list \
      --checkpoint $decode_checkpoint \
      --simulate_streaming \
      --beam_size 10 \
      --batch_size 1 \
      --penalty 0.0 \
      --result_dir $result_dir \
      --ctc_weight $ctc_weight \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}

    for mode in $decode_modes; do
      test_dir=$result_dir/$mode
      python tools/compute-wer.py --char=1 --v=1 \
        $wave_data/$test/text $test_dir/text > $test_dir/wer
    done
  done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Export the best model you want
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip
fi

# Optionally, you can add LM and test it with runtime.
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then

  lm=data/local/lm
  lexicon=data/local/dict/lexicon.txt

  # 7.2 Prepare dict
  unit_file=$dict
  bpemodel=$bpemodel

  cp $unit_file data/local/dict/units.txt

  ./tools/build_decoding_wfst.sh
  
  # 7.3 Build decoding TLG
  tools/fst/compile_lexicon_token_fst.sh \
     data/local/dict data/local/tmp data/local/lang
  tools/fst/make_tlg.sh data/local/lm data/local/lang data/lang_test || exit 1;

  # 7.4 Decoding with runtime
  fst_dir=data/lang_test
  for test in ${recog_set}; do
    ./tools/decode.sh --nj 6 \
      --beam 10.0 --lattice_beam 5 --max_active 7000 --blank_skip_thresh 0.98 \
      --ctc_weight 0.5 --rescoring_weight 1.0 --acoustic_scale 1.2 \
      --fst_path $fst_dir/TLG.fst \
      --dict_path $fst_dir/words.txt \
      data/$test/wav.scp data/$test/text $dir/final.zip $fst_dir/units.txt \
      $dir/lm_with_runtime_${test}
    tail $dir/lm_with_runtime_${test}/wer
  done
fi

