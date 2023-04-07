# パス解決
from glob import glob
import json
import os
from pathlib import Path
import sys
from mlib.base.base import FileType

from mlib.base.logger import LoggingDecoration, MLogger

logger = MLogger(os.path.basename(__file__))


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


HISTORY_MAX = 50


def read_histories(history_keys: list[str]) -> dict[str, list[str]]:
    """ファイル履歴を取得する"""

    root_dir = get_root_dir()
    history_path = os.path.join(root_dir, "history.json")

    file_histories: dict[str, list[str]] = {}
    for key in history_keys:
        file_histories[key] = []

    # 履歴JSONファイルがあれば読み込み
    try:
        if not (os.path.exists(history_path) and os.path.isfile(history_path)):
            return file_histories

        with open(history_path, "r", encoding="utf-8") as f:
            file_histories = json.load(f)
            # キーが揃っているかチェック
            for key in history_keys:
                if key not in file_histories:
                    file_histories[key] = []
    finally:
        pass

    return file_histories


def insert_history(value: str, histories: list[str]):
    """ファイル履歴に追加する"""
    if value in histories:
        histories.remove(value)
    histories.insert(0, value)


def save_histories(histories: dict[str, list[str]]):
    """ファイル履歴を保存する"""
    root_dir = get_root_dir()

    limited_histories: dict[str, list[str]] = {}
    for key, values in histories.items():
        limited_histories[key] = values[:HISTORY_MAX]

    try:
        with open(os.path.join(root_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(limited_histories, f, ensure_ascii=False)
    except Exception as e:
        logger.error("履歴ファイルの保存に失敗しました", e, decoration=LoggingDecoration.DECORATION_BOX)


def get_dir_path(path: str) -> str:
    """指定されたパスの親を取得する"""
    if os.path.exists(path) and os.path.isfile(path):
        file_paths = [path]
    else:
        file_paths = [p for p in glob(path) if os.path.isfile(p)]

    if not len(file_paths):
        return ""

    try:
        # ファイルパスをオブジェクトとして解決し、親を取得する
        return str(Path(file_paths[0]).resolve().parents[0])
    except Exception:
        logger.error("ファイルパスの解析に失敗しました。\nパスに使えない文字がないか確認してください。\nファイルパス: {path}", path=path)
        return ""


def validate_file(
    path: str,
    file_type: FileType,
) -> bool:
    """利用可能なファイルであるか"""
    if not (os.path.exists(path) and os.path.isfile(path)):
        return False

    _, _, file_ext = separate_path(path)
    if not file_ext[1:].lower() in file_type.name.lower():
        return False

    return True


def validate_save_file(path: str, title: str) -> bool:
    """保存可能なファイルであるか"""
    try:
        open(path, "w")
        os.remove(path)
    except Exception:
        logger.warning(
            f"{title}のチェックに失敗しました。以下の原因が考えられます。\n"
            + f"{title}が255文字を超えている\n"
            + f'{title}に使えない文字列が含まれている（例) \\ / : * ? " < > |）'
            + f"{title}の親フォルダに書き込み権限がない"
            + f"{title}に書き込み権限がない"
        )
        return False
    return True


def separate_path(path: str) -> tuple[str, str, str]:
    """ファイルパスをディレクトリ・ファイル名・ファイル拡張子に分割する"""
    dir_path = os.path.dirname(path)
    file_name, file_ext = os.path.splitext(os.path.basename(path))

    return dir_path, file_name, file_ext


def escape_path(path: str):
    """ファイルパスをエスケープ"""
    for org_txt, rep_txt in (
        ("\\", "\\\\"),
        ("*", "\\*"),
        ("+", "\\+"),
        (".", "\\."),
        ("?", "\\?"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("(", "\\("),
        (")", "\\)"),
        ("[", "\\["),
        ("]", "\\]"),
        ("^", "\\^"),
        ("$", "\\$"),
        ("-", "\\-"),
        ("|", "\\|"),
        ("/", "\\/"),
    ):
        path = path.replace(org_txt, rep_txt)

    return path
