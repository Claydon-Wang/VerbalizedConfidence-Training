# Installation

## 1. Create the Environment

```bash
conda create -n rl python=3.10
conda activate rl
```

## 2. Install Project Dependencies

```bash
pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple/
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 3. Install `trl` 

```bash
cd ../
git clone https://github.com/huggingface/trl.git
cd trl
git checkout 69ad852e5654a77f1695eb4c608906fe0c7e8624
pip install -e .
```

## 4. Log In to SwanLab

```bash
swanlab login
```
