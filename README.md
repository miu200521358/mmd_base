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


### Python3.11 pyinstaller ローカルビルド

 1. pyinstaller のローカル構築
    1. https://pyinstaller.org/en/stable/bootloader-building.html
    1. https://gamingpc.one/dev/python-pyinstaller/
 1. `conda create -n mbase2 pip python=3.11`
 1. `conda activate mbase2`
 1. vcbuildtoolsをインストール (管理者権限つきPowerShell)
    1. `Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))`
    1. `conda activate mbase2`
    1. `choco install -y python visualstudio2019-workload-vctools`
 1. pyinstaller のインストール
    1. `git clone https://github.com/pyinstaller/pyinstaller`
    1. (mbase2) PS C:\MMD\pyinstaller\bootloader> `python ./waf all`
 1. `pip install -r requirements.txt`
 1. bezier のインストール
     1. `set BEZIER_NO_EXTENSION=true`
     1. `pip install bezier --no-binary=bezier`

## profile

 1. `cls & python -m cProfile -s tottime example.py`


