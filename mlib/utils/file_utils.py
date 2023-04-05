# パス解決
import os
from pathlib import Path
import sys


def get_root_dir():
    """
    ルートディレクトリパスを取得する
    exeに固めた後のルートディレクトリパスも取得
    """
    exec_path = sys.argv[0]
    root_path = Path(exec_path).parent if hasattr(sys, "frozen") else Path(__file__).parent

    return root_path


def get_path(relative_path: str):
    """ディレクトリ内のパスを解釈する"""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return relative_path
