# How to build ASR system with Wenet 2.0
First of all, please refer to README.md in root folder and make sure that all of dependencies are installed completely.

## Step 1: Convert data csv to kaldi format
Actually, I wrote a custom recipe for training, in this stage you only need ```csv``` files with 2 columns (```wav``` is path of audio, ```wrd``` is label)

``` sh
bash run.sh --stage 1 --stop_stage 1
```

In the step, we will directly convert data in csv file into kaldi format. Two files are generated when you run bash: 
* wav.scp : each line records two space-separated columns : ```wav_id``` and ```wav_path```
* text : each line records two space-separated columns : ```wav_id``` and ```text_label```

```tools/compute_cmvn_stats.py``` is used to extract global cmvn(cepstral mean and variance normalization) statistics. These statistics will be used to normalize the acoustic features. Setting ```cmvn=false``` will skip this step.

## Step 2: Train Tokenizer
In the stage, I only mention train BPE tokenizer using ```sentencepiece``` if you want to use another style (char), please refer to ```run.sh``` and change the config.
``` sh
bash run.sh --stage 2 --stop_stage 2
```
The model unit is defined as a dict in WeNet, which maps the a BPE into integer index.
``` text
<blank> 0
<unk> 1
' 2
▁ 3
A 4
▁A 5
AB 6
```
All configs to train BPE model are predefined in ```run.sh```.
## Step 3: Prepare Training data
To train ASR model in Wenet, we must make data in wenet format. Simply, run stage 3, all is built on the fly.
``` sh
bash run.sh --stage 3 --stop_stage 3
```
This stage generates the WeNet required format file ```data.list```. Each line in ```data.list``` is in json format which contains the following fields.
1. key: key of the utterance
2. wav: audio file path of the utterance
3. txt: normalized transcription of the utterance, the transcription will be tokenized to the model units on-the-fly at the training stage.

Wenet aslo design another format for ```data.list``` named ```shard``` which is for big data training. Please see ```gigaspeech(10k hours)``` or ```wenetspeech(10k hours)``` for how to use shard style data.list if you want to apply WeNet on big data set(more than 5k).
## Step 4: Train Neural Network
The NN model is trained in this step.
``` sh
bash run.sh --stage 4 --stop_stage 4
```
* Multi-GPU mode
If using DDP mode for multi-GPU, we suggest using ```dist_backend="nccl"```. If the NCCL does not work, try using ```gloo``` for CPU. Set the GPU ids in CUDA_VISIBLE_DEVICES. For example, set export CUDA_VISIBLE_DEVICES="0,1,2,3,6,7" to use card 0,1,2,3,6,7.

* Resume training
If your experiment is terminated after running several epochs for some reasons (e.g. the GPU is accidentally used by other people and is out-of-memory ), you could continue the training from a checkpoint model. Just find out the finished epoch in ```exp/your_exp/```, set ```checkpoint=exp/your_exp/$n.pt``` and run the run.sh --stage 4. Then the training will continue from the ```$n+1.pt```

* Config
The config of neural network structure, optimization parameter, loss parameters, and dataset can be set in a YAML format file.
In conf/, we provide several models like transformer and conformer. see ```conf/train_u2++_conformer.yaml``` for reference.



## Step 5: Recognize wav using the trained model
``` sh
bash run.sh --stage 5 --stop_stage 5
```
This stage shows how to recognize a set of wavs into texts. It also shows how to do the model averaging.
1. Do model averaging
2. Compute WER



## Step 6: (Optional): Export the trained model
``` sh
bash run.sh --stage 6 --stop_stage 6
```
```wenet/bin/export_jit.py``` will export the trained model using Libtorch. The exported model files can be easily used for C++ inference in our runtime. It is required if you want to integrate language model(LM), as shown in Stage 

## Step 7: (Optional): Add LM and test it with runtime
``` sh
bash run.sh --stage 7 --stop_stage 7
```
Make sure that you completely build runtime in wenet.
1. Build runtime
``` bash
# runtime build requires cmake 3.14 or above
cd runtime/libtorch
mkdir build && cd build && cmake -DGRAPH_TOOLS=ON .. && cmake --build .
```
2. Build Language Model and WFST
You can self-build a Language Model using ```kenlm``` and put the path in ```build_decoding_wfst.sh```, make sure it's ok to build WFST.
When building WFST, you need a lexicon for the language you're considering. Simply, building lexicon is not complicated. A word is composed of the tokens which are tokenized by BPE. 
