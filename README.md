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

Please see [doc](https://github.com/wenet-e2e/wenet/tree/main/runtime) for building
runtime on more platforms and OS.

