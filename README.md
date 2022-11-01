# mmd_tool
Windows用MMDツールライブラリ

## Command

### Python3.9

 1. `conda create -n mbase pip python=3.9`
 2. `pip install -r requirements.txt`
 3. `<ライブラリパス>\bezier\extra-dll\bezier-2a44d276.dll` を `<ライブラリパス>\bezier\bezier.dll` に置く


### Python3.11

 1. `conda create -n mbase`
 2. `conda activate mbase`
 3. `conda install -c conda-forge python`
 4. `pip install -r requirements.txt`
 5. bezier, wxPython をソースコードからインストール


## profile

 1. `cls & python -m cProfile -s tottime example.py`


