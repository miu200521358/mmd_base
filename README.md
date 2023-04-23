# mmd_tool
Windows用MMDツールライブラリ

## Command

### Python3.9

 1. `conda create -n mbase pip python=3.9`
 2. `pip install -r requirements.txt`
 3. `<ライブラリパス>\bezier\extra-dll\bezier-2a44d276.dll` を `<ライブラリパス>\bezier\bezier.dll` に置く


### Python3.11

 1. `conda create -n mbase pip python=3.11`
 1. `conda activate mbase`
 1. `pip install -r requirements.txt`
 1. bezier のインストール
     1. `set BEZIER_NO_EXTENSION=true`
     1. `pip install bezier --no-binary=bezier`
 1. wxPython のインストール
     1. `pip install attrdict3 requests wxpython`


## profile

 1. `cls & python -m cProfile -s tottime example.py`


