import gettext
import logging
import os
from datetime import datetime
from enum import Enum, IntEnum
from logging import Formatter, Handler, LogRecord, StreamHandler
import re
from typing import Optional

import numpy as np
import wx

from mlib.base.exception import MLibException


class LoggingMode(IntEnum):
    # 翻訳モード
    # 読み取り専用：翻訳リストにない文字列は入力文字列をそのまま出力する
    MODE_READONLY = 0
    # 更新あり：翻訳リストにない文字列は出力する
    MODE_UPDATE = 1


class LoggingDecoration(Enum):
    DECORATION_IN_BOX = "in_box"
    DECORATION_BOX = "box"
    DECORATION_LINE = "line"


class LoggingLevel(Enum):
    DEBUG_FULL = 2
    TEST = 5
    TIMER = 12
    FULL = 15
    INFO_DEBUG = 22
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class MLogger:
    DEFAULT_FORMAT = "%(message)s [%(call_file)s:%(call_func)s:%(call_lno)s][P-%(process)s](%(asctime)s)"

    # システム全体のロギングレベル
    total_level = logging.INFO
    # システム全体の開始出力日時
    output_datetime = ""

    # LoggingMode
    mode = LoggingMode.MODE_READONLY
    # デフォルトログファイルパス
    default_out_path = ""
    # デフォルト翻訳言語
    lang = "en"
    translator = None
    # i18n配置ディレクトリ
    lang_dir = None
    # セーブモード
    saving = True
    # ログ出力モード
    out_log = False

    console_handler: Optional["ConsoleHandler"] = None
    re_break = re.compile(r"\n")

    def __init__(
        self,
        module_name,
        level=logging.INFO,
        out_path: Optional[str] = None,
    ):
        self.file_name = module_name
        self.default_level = level

        if not out_path:
            # クラス単位の出力パスがない場合、デフォルトパス
            out_path = self.default_out_path

        # ロガー
        self.logger = logging.getLogger("mutool").getChild(self.file_name)

        self.stream_handler = StreamHandler()
        # self.stream_handler.setFormatter(Formatter(self.DEFAULT_FORMAT))
        self.logger.addHandler(self.stream_handler)

        if out_path:
            # ファイル出力ハンドラ
            self.file_handler = logging.FileHandler(out_path)
            self.file_handler.setLevel(self.default_level)
            # self.file_handler.setFormatter(Formatter(self.DEFAULT_FORMAT))
            self.logger.addHandler(self.file_handler)

        self.logger.setLevel(level)

    def add_console_handler(self):
        if self.console_handler:
            for h in self.logger.handlers:
                if isinstance(h, ConsoleHandler):
                    return
            self.logger.addHandler(self.console_handler)

    def get_extra(self, msg: str, func: Optional[str] = "", lno: Optional[int] = 0):
        return {"original_msg": msg, "call_file": self.file_name, "call_func": func, "call_lno": str(lno)}

    def debug(
        self,
        msg,
        *args,
        decoration: Optional[LoggingDecoration] = None,
        func: Optional[str] = "",
        lno: Optional[int] = 0,
        **kwargs,
    ):
        self.add_console_handler()
        self.logger.debug(self.create_message(msg, logging.DEBUG, None, decoration, **kwargs), extra=self.get_extra(msg, func, lno))

    def info(
        self,
        msg,
        *args,
        title: Optional[str] = None,
        decoration: Optional[LoggingDecoration] = None,
        func: Optional[str] = "",
        lno: Optional[int] = 0,
        **kwargs,
    ):
        self.add_console_handler()
        self.logger.info(self.create_message(msg, logging.INFO, title, decoration, **kwargs), extra=self.get_extra(msg, func, lno))

    # ログレベルカウント
    def count(
        self,
        msg,
        fno,
        fnos,
        *args,
        title: Optional[str] = None,
        decoration: Optional[LoggingDecoration] = None,
        func: Optional[str] = "",
        lno: Optional[int] = 0,
        **kwargs,
    ):
        self.add_console_handler()
        last_fno = 0

        if fnos and 0 < len(fnos) and 0 < fnos[-1]:
            last_fno = fnos[-1]

        if not fnos and kwargs and "last_fno" in kwargs and 0 < kwargs["last_fno"]:
            last_fno = kwargs["last_fno"]

        if 0 < last_fno:
            if not kwargs:
                kwargs = {}

            kwargs["level"] = LoggingLevel.INFO.value
            kwargs["fno"] = fno
            kwargs["per"] = round((fno / last_fno) * 100, 3)
            kwargs["msg"] = msg
            log_msg = "-- {fno}フレーム目:終了({per}％){msg}"

            self.logger.info(self.create_message(log_msg, logging.INFO, title, **kwargs), decoration, extra=self.get_extra(msg, func, lno))

    def warning(
        self,
        msg,
        *args,
        title: Optional[str] = None,
        decoration: Optional[LoggingDecoration] = None,
        func: Optional[str] = "",
        lno: Optional[int] = 0,
        **kwargs,
    ):
        self.add_console_handler()
        self.logger.warning(self.create_message(msg, logging.INFO, title, decoration, **kwargs), extra=self.get_extra(msg, func, lno))

    def error(
        self,
        msg,
        *args,
        title: Optional[str] = None,
        decoration: Optional[LoggingDecoration] = None,
        func: Optional[str] = "",
        lno: Optional[int] = 0,
        **kwargs,
    ):
        self.add_console_handler()
        self.logger.error(self.create_message(msg, logging.INFO, title, decoration, **kwargs), extra=self.get_extra(msg, func, lno))

    def critical(
        self,
        msg,
        *args,
        title: Optional[str] = None,
        decoration: Optional[LoggingDecoration] = None,
        func: Optional[str] = "",
        lno: Optional[int] = 0,
        **kwargs,
    ):
        self.add_console_handler()
        self.logger.critical(self.create_message(msg, logging.INFO, title, decoration, **kwargs), exc_info=True, extra=self.get_extra(msg, func, lno))

    def quit(self):
        # 終了ログ
        with open("../log/quit.log", "w") as f:
            f.write("quit")

    def get_text(self, text: str, **kwargs) -> str:
        """指定された文字列の翻訳結果を取得する"""
        if not self.translator:
            if kwargs:
                return text.format(**kwargs)
            return text

        # 翻訳する
        if self.mode == LoggingMode.MODE_UPDATE and logging.DEBUG < self.total_level:
            # 更新ありの場合、既存データのチェックを行って追記する
            messages = []
            with open(f"{self.lang_dir}/messages.pot", mode="r", encoding="utf-8") as f:
                messages = f.readlines()

            new_msg = self.re_break.sub("\\\\n", text)
            added_msg_idxs = [n + 1 for n, inmsg in enumerate(messages) if "msgid" in inmsg and new_msg in inmsg]

            if not added_msg_idxs:
                messages.append(f'\nmsgid "{new_msg}"\n')
                messages.append('msgstr ""\n')
                messages.append("\n")
                self.logger.debug("add message: ", new_msg)

                with open(f"{self.lang_dir}/messages.pot", mode="w", encoding="utf-8") as f:
                    f.writelines(messages)

        # 翻訳結果を取得する
        trans_text = self.translator.gettext(text)
        if kwargs:
            return trans_text.format(**kwargs)
        return trans_text

    # 実際に出力する実態
    def create_message(
        self,
        msg,
        level: int,
        title: Optional[str] = None,
        decoration: Optional[LoggingDecoration] = None,
        **kwargs,
    ) -> str:
        # 翻訳結果を取得する
        trans_msg = self.get_text(msg, **kwargs)

        if decoration:
            if decoration == LoggingDecoration.DECORATION_BOX:
                output_msg = self.create_box_message(trans_msg, level, title)
            elif decoration == LoggingDecoration.DECORATION_LINE:
                output_msg = self.create_line_message(trans_msg, level, title)
            elif decoration == LoggingDecoration.DECORATION_IN_BOX:
                output_msg = self.create_in_box_message(trans_msg, level, title)
            else:
                output_msg = msg
        else:
            output_msg = msg

        return output_msg

    def create_box_message(self, msg, level, title=None):
        msg_block = []
        msg_block.append("■■■■■■■■■■■■■■■■■")

        if level == logging.CRITICAL:
            msg_block.append("■　**CRITICAL**  ")

        if level == logging.ERROR:
            msg_block.append("■　**ERROR**  ")

        if level == logging.WARNING:
            msg_block.append("■　**WARNING**  ")

        if logging.INFO >= level and title:
            msg_block.append(f"■　**{title}**  ")

        msg_block.extend([f"■　{msg_line}" for msg_line in msg.split("\n")])
        msg_block.append("■■■■■■■■■■■■■■■■■")

        return "\n".join(msg_block)

    def create_line_message(self, msg, level, title=None):
        msg_block = [f"-- {msg_line} --------------------" for msg_line in msg.split("\n")]
        return "\n".join(msg_block)

    def create_in_box_message(self, msg, level, title=None):
        msg_block = [f"■　{msg_line}" for msg_line in msg.split("\n")]
        return "\n".join(msg_block)

    @classmethod
    def initialize(
        cls,
        lang: str,
        root_dir: str,
        mode: LoggingMode = LoggingMode.MODE_READONLY,
        saving: bool = True,
        level=logging.INFO,
        out_path=None,
    ):
        logging.basicConfig(level=level)
        cls.total_level = level
        cls.mode = LoggingMode.MODE_READONLY if lang != "ja" else mode
        cls.lang = lang
        cls.saving = saving
        cls.lang_dir = f"{root_dir}/i18n"

        # 翻訳用クラスの設定
        cls.translator = gettext.translation(
            "messages",  # domain: 辞書ファイルの名前
            localedir=f"{root_dir}/i18n",  # 辞書ファイル配置ディレクトリ
            languages=[lang],  # 翻訳に使用する言語
            fallback=True,  # .moファイルが見つからなかった時は未翻訳の文字列を出力
        )

        output_datetime = "{0:%Y%m%d_%H%M%S}".format(datetime.now())
        cls.output_datetime = output_datetime
        log_dir = f"{root_dir}/../log"

        # ファイル出力ありの場合、ログファイル名生成
        if not out_path:
            os.makedirs(log_dir, exist_ok=True)
            cls.default_out_path = f"{log_dir}/mutool_{output_datetime}.log"
            cls.out_log = False
        else:
            cls.default_out_path = out_path
            cls.out_log = True

        if os.path.exists(f"{log_dir}/quit.log"):
            # 終了ログは初期化時に削除
            os.remove(f"{log_dir}/quit.log")


def parse2str(obj: object) -> str:
    """オブジェクトの変数の名前と値の一覧を文字列で返す

    Parameters
    ----------
    obj : object

    Returns
    -------
    str
        変数リスト文字列
        Sample[x=2, a=sss, child=ChildSample[y=4.5, b=xyz]]
    """
    return f"{obj.__class__.__name__}[{', '.join([f'{k}={round_str(v)}' for k, v in vars(obj).items()])}]"


def round_str(v: object, decimals=5) -> str:
    """
    丸め処理付き文字列変換処理

    小数だったら丸めて一定桁数までしか出力しない
    """
    if isinstance(v, float):
        return f"{round(v, decimals)}"
    elif isinstance(v, np.ndarray):
        return f"{np.round(v, decimals)}"
    elif hasattr(v, "data"):
        return f"{np.round(v.__getattribute__('data'), decimals)}"
    else:
        return f"{v}"


# ファイルのエンコードを取得する
def get_file_encoding(file_path):
    try:
        f = open(file_path, "rb")
        fbytes = f.read()
        f.close()
    except:
        raise MLibException("unknown encoding!")

    codes = ("utf-8", "shift-jis")

    for encoding in codes:
        try:
            fstr = fbytes.decode(encoding)  # bytes文字列から指定文字コードの文字列に変換
            fbytes = fstr.encode("utf-8")  # uft-8文字列に変換
            # 問題なく変換できたらエンコードを返す
            return encoding
        except Exception as e:
            print(e)
            pass

    raise MLibException("unknown encoding!")


class ConsoleHandler(Handler):
    def __init__(self, text_ctrl: wx.TextCtrl):
        super().__init__()
        self.text_ctrl = text_ctrl

    def emit(self, record: LogRecord):
        msg = self.format(record)
        wx.CallAfter(self.text_ctrl.WriteText, msg + "\n")
