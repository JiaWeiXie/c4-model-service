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
# pip install -r requirements/gpu.txt
# Add dev packages
# pip install -r requirements/cpu.txt -r requirements/dev.txt
# pip install -r requirements/gpu.txt -r requirements/dev.txt
```

## Run web app

```bash
$ cd c4-model-service
$ source .venv/bin/activate

$ python main.py
```

> open `http://localhost:8088`