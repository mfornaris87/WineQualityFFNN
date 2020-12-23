# WINE QUALITY

## Requirements

Config virtualenv:

```bash
$ git clone https://github.com/eadomenech/WineQuality.git src
$ cd src/
$ python3 -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

Run:

```bash
tensorboard --logdir=runs 
```

```bash
python WineQuality.py --batch-size 10 --epochs 100 
```
