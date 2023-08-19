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
 1. `conda create -n mbase3 pip python=3.11`
    1. `pip uninstall setuptools`
    1. `pip install setuptools==58.2.0`
 1. `conda activate mbase3`
 1. vcbuildtoolsをインストール (管理者権限つきPowerShell)
    1. `C:\Users\celes\OneDrive\�h�L�������g\WindowsPowerShell\profile.ps1`
    1. `Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))`
    1. `conda activate mbase3`
    1. `choco install -y --force python visualstudio2019-workload-vctools`
 1. pyinstaller のインストール
    1. `git clone https://github.com/pyinstaller/pyinstaller`
    1. (mbase3) PS C:\MMD\pyinstaller\bootloader> `python ./waf all`
 1. `pip install -r requirements.txt`
 1. `pip install -r requirements_test.txt`
 1. bezier のインストール
     1. `set BEZIER_NO_EXTENSION=true`
     1. `pip install bezier --no-binary=bezier`


### Python3.11 pyinstaller ローカルビルド (2023.08)

1. `conda create -n mtool pip python=3.11`
   1. `conda activate mtool`
   1. `pip install -U setuptools`
1. vcbuildtoolsをインストール (管理者権限つきPowerShell)
   1. `C:\MMD\mmd_base\profile.ps1`
   1. `conda activate mtool`
   1. `choco install -y --force python visualstudio2019-workload-vctools`
   1. pyinstaller のインストール
      1. `git clone https://github.com/pyinstaller/pyinstaller`
      1. (mtool) PS C:\MMD\pyinstaller\bootloader> `python ./waf all`
      1. (mtool) PS C:\MMD\pyinstaller> `pip install .`
1. (mtool) PS C:\MMD\mmd_base> `pip install -r requirements.txt`
1. (mtool) PS C:\MMD\mmd_base> `pip install -r requirements_test.txt`


### Python3.11 nuitka 版 (2023.08) 没

1. `conda create -n mtool2 pip python=3.11`
1. `conda activate mtool2`
1. `pip install bezier==2023.7.28 --no-binary=bezier`
1. `pip install -r requirements.txt`
1. `pip install -r requirements_test.txt`


## profile

 1. `cls & python -m cProfile -s tottime example.py`


