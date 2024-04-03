# WeNet

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.7%7C3.8-brightgreen)](https://github.com/wenet-e2e/wenet)

[**Roadmap**](https://github.com/wenet-e2e/wenet/issues/1683)
| [**Docs**](https://wenet-e2e.github.io/wenet)
| [**Papers**](https://wenet-e2e.github.io/wenet/papers.html)
| [**Runtime**](https://github.com/wenet-e2e/wenet/tree/main/runtime)
| [**Pretrained Models**](docs/pretrained_models.md)
| [**HuggingFace**](https://huggingface.co/spaces/wenet/wenet_demo)

**We** share **Net** together.

## Highlights

* **Production first and production ready**: The core design principle, WeNet provides full stack production solutions for speech recognition.
* **Accurate**: WeNet achieves SOTA results on a lot of public speech datasets.
* **Light weight**: WeNet is easy to install, easy to use, well designed, and well documented.


## Install

### Install python package

``` sh
pip install git+https://github.com/wenet-e2e/wenet.git
```

**Command-line usage** (use `-h` for parameters):

``` sh
wenet --language chinese audio.wav
```

**Python programming usage**:

``` python
import wenet

model = wenet.load_model('chinese')
result = model.transcribe('audio.wav')
print(result['text'])
```

Please refer [python usage](docs/python_package.md) for more command line and python programming usage.

### Install for training & deployment

- Clone the repo
``` sh
git clone https://github.com/wenet-e2e/wenet.git
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

``` sh
conda create -n wenet python=3.8
conda activate wenet
pip install -r requirements.txt
pre-commit install  # for clean and tidy code
```
- Run sample recipe
``` sh
cd examples/vinbrain/s0
```
**Build for deployment**

Optionally, if you want to use x86 runtime or language model(LM),
you have to build the runtime as follows. Otherwise, you can just ignore this step.

``` sh
# runtime build requires cmake 3.14 or above
cd runtime/libtorch
mkdir build && cd build && cmake -DGRAPH_TOOLS=ON .. && cmake --build .
```
**Build websocket**

I have already prepared 2 scripts to run client-server.
Start server:
``` sh
./runtime/libtorch/build/bin/run_server.sh
```

``` bash 
export GLOG_logtostderr=1
export GLOG_v=2
model_dir=/examples/foo/s0/exp/sp_spec_aug
TLG_path=/examples/foo/s0/data/lang_test/TLG.fst
words_path=/examples/foo/s0/data/lang_test/words.txt
./websocket_server_main \
    --port 10086 \
    --chunk_size 8 \
    --model_path $model_dir/final.zip \
    --unit_path $model_dir/units.txt 2>&1 | tee server.log \
    --fst_path $TLG_path \
    --dict_path $words_path

```

Start client:
``` sh
./runtime/libtorch/build/bin/run_client.sh
```

``` bash
export GLOG_logtostderr=1
export GLOG_v=2
wav_path=/zalo/program-0132/program-0132-00693.wav
./websocket_client_main \
    --hostname 127.0.0.1 --port 10086 \
    --wav_path $wav_path 2>&1 | tee client.log
```

Edit ```wav_path``` variable in run_client.sh if you try to test your own audio.

Please see [doc](https://github.com/wenet-e2e/wenet/tree/main/runtime) for building
runtime on more platforms and OS.
