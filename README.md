# mmd_tool
Windows用MMDツールライブラリ

## Command

 1. `conda create -n mmd_tool pip python=3.9`
 2. `conda activate mmd_tool`
 3. `pip install -r requirements.txt`
 4. `<ライブラリパス>\bezier\extra-dll\bezier-2a44d276.dll` を `<ライブラリパス>\bezier\bezier.dll` に置く
 
 
## profile
 
 1. `cls & python -m cProfile -s tottime example.py`