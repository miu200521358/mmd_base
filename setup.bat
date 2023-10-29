@echo off
cls

cd /d %~dp0

python setup_clear.py

python setup.py clean

python setup.py build_ext --force --inplace

cd ..

rem -- 音を鳴らす
rundll32 user32.dll,MessageBeep