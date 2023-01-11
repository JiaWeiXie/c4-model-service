# C4 Model Service

## Installtion

```bash
$ git clone https://github.com/JiaWeiXie/c4-model-service.git
$ cd c4-model-service
$ git submodule update --init --recursive
$ python -m venv .venv

$ source .venv/bin/activate
$ pip install -U pip
$ pip install -r requirements/cpu.txt
# CUDA 11.7
# pip install -r requirements/gpu.txt
# Add dev packages
# pip install -r requirements/cpu.txt -r requirements/dev.txt
# pip install -r requirements/gpu.txt -r requirements/dev.txt
```

## Run web app

- Download model checkpoint [`pytorch_model.bin`](https://ntubedutw-my.sharepoint.com/:f:/g/personal/11065001_ntub_edu_tw/ErjjyvML9xlPp_KqHet1YUABeEcof4-Jd5bO-GdVsnjOOQ?e=5a5c5h)
- Move `pytorch_model.bin` to `c4-model-service/checkpoints/pytorch_model.bin`

```bash
$ cd c4-model-service
$ source .venv/bin/activate

$ python main.py
```

> Open `http://localhost:8088`