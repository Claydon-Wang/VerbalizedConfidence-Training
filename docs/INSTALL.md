```bash
conda create -n rl python=3.10
pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple/
pip install flash-attn==2.7.4.post1 -i https://mirrors.cloud.tencent.com/pypi/simple/


```

### TRL Installation 

```bash
cd ../
git clone https://github.com/huggingface/trl.git
cd trl/
git checkout 69ad852e5654a77f1695eb4c608906fe0c7e8624
pip install -e .
```

### Login to swanlab 

```bash
swanlab login
```