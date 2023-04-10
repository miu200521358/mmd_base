from typing import Optional


class MLibException(Exception):
    """ライブラリ内基本エラー"""

    def __init__(self, message: str = "", exception: Optional[Exception] = None, *args, **kwargs):
        super().__init__(*args)
        self.message = message
        self.exception = exception
        self.kwargs = kwargs

    def __str__(self) -> str:
        return self.message


class MApplicationException(MLibException):
    """ツールがメイン処理出来なかった時のエラー"""

    def __init__(self, message: str = "", exception: Optional[Exception] = None, *args, **kwargs):
        super().__init__(message, exception, *args, **kwargs)


class MParseException(MLibException):
    """ツールがパース出来なかった時のエラー"""

    def __init__(self, message: str = "", exception: Optional[Exception] = None, *args, **kwargs):
        super().__init__(message, exception, *args, **kwargs)


class MViewerException(MLibException):
    """ツールが描画出来なかった時のエラー"""

    def __init__(self, message: str = "", exception: Optional[Exception] = None, *args, **kwargs):
        super().__init__(message, exception, *args, **kwargs)


class MKilledException(MLibException):
    """ツールの実行が停止された時のエラー"""

    def __init__(self, message: str = "", exception: Optional[Exception] = None, *args, **kwargs):
        super().__init__(message, exception, *args, **kwargs)
