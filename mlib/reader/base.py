import struct
from abc import ABCMeta, abstractmethod
from typing import Any, Generic, Union

import numpy as np
from mlib.base import BaseModel, Encoding, TBaseModel
from mlib.exception import MParseException
from mlib.math import MQuaternion, MVector2D, MVector3D, MVector4D
from mlib.model.base import TBaseHashModel


class BaseReader(Generic[TBaseHashModel], BaseModel, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        self.offset = 0
        self.buffer: bytes = b""

    def read_name_by_filepath(self, path: str) -> str:
        """
        指定されたパスのファイルから該当名称を読み込む

        Parameters
        ----------
        path : str
            ファイルパス

        Returns
        -------
        str
            読み込み結果文字列
        """
        # モデルを新規作成
        model: TBaseHashModel = self.create_model(path)

        # バイナリを解凍してモデルに展開
        try:
            with open(path, "rb") as f:
                self.buffer = f.read()
                self.read_by_buffer_header(model)
        except Exception:
            return ""

        return model.get_name()

    def read_by_filepath(self, path: str) -> TBaseHashModel:
        """
        指定されたパスのファイルからデータを読み込む

        Parameters
        ----------
        path : str
            ファイルパス

        Returns
        -------
        TBaseHashModel
            読み込み結果
        """
        # モデルを新規作成
        model: TBaseHashModel = self.create_model(path)

        # バイナリを解凍してモデルに展開
        try:
            with open(path, "rb") as f:
                self.buffer = f.read()
                self.read_by_buffer_header(model)
                self.read_by_buffer(model)
        except MParseException as pe:
            raise pe
        except Exception as e:
            # TODO
            raise MParseException("予期せぬエラー", exception=e)

        # ハッシュを保持
        model.digest = model.hexdigest()

        return model

    @abstractmethod
    def create_model(self, path: str) -> TBaseHashModel:
        """
        読み取り対象モデルオブジェクトを生成する

        Returns
        -------
        TBaseHashModel
            モデルオブジェクト
        """
        pass

    @abstractmethod
    def read_by_buffer_header(self, model: TBaseHashModel):
        """
        バッファからモデルデータヘッダを読み取る

        Parameters
        ----------
        buffer : TBuffer
            バッファ
        model : TBaseHashModel
            モデルオブジェクト
        """
        pass

    @abstractmethod
    def read_by_buffer(self, model: TBaseHashModel):
        """
        バッファからモデルデータを読み取る

        Parameters
        ----------
        buffer : TBuffer
            バッファ
        model : TBaseHashModel
            モデルオブジェクト
        """
        pass

    def define_encoding(self, encoding: Encoding):
        """
        エンコードを設定し、それに基づくテキストデータ読み取り処理を定義する

        Parameters
        ----------
        encoding : Encoding
            エンコード
        """
        self.encoding = encoding
        self.read_text = self.define_read_text(self.encoding)

    def define_read_text(self, encoding: Encoding):
        """
        テキストの解凍定義

        Parameters
        ----------
        encoding : Encoding
            デコードエンコード
        """

        def read_text() -> str:
            format_size = self.read_int()
            return self.decode_text(
                encoding, self.unpack(f"{format_size}s", format_size)
            )

        return read_text

    def decode_text(self, main_encoding: Encoding, fbytes: bytearray) -> str:
        """
        テキストデコード

        Parameters
        ----------
        main_encoding : Encoding
            基本のエンコーディング
        fbytes : bytearray
            バイト文字列

        Returns
        -------
        Optional[str]
            デコード済み文字列
        """
        # 基本のエンコーディングを第一候補でデコードして、ダメなら順次テスト
        for target_encoding in [
            main_encoding,
            Encoding.SHIFT_JIS,
            Encoding.UTF_8,
            Encoding.UTF_16_LE,
        ]:
            try:
                if target_encoding == Encoding.SHIFT_JIS:
                    # shift-jisは一旦cp932に変換してもう一度戻したので返す
                    return (
                        fbytes.decode(Encoding.SHIFT_JIS.value, errors="replace")
                        .encode(Encoding.CP932.value, errors="replace")
                        .decode(Encoding.CP932.value, errors="replace")
                    )

                # 変換できなかった文字は「?」に変換する
                return fbytes.decode(encoding=target_encoding.value, errors="replace")
            except Exception:
                pass
        return ""

    def read_MVector2D(
        self,
    ) -> MVector2D:
        """
        MVector2Dの解凍

        Parameters
        ----------
        buffer : TBuffer
            バッファ
        offset : int
            オフセット

        Returns
        -------
        tuple[MVector2D, int]
            MVector2Dデータ
            オフセット
        """
        return self.read_to_model(
            [("x", float), ("y", float)],
            MVector2D(),
        )

    def read_MVector3D(
        self,
    ) -> MVector3D:
        """
        MVector3Dの解凍

        Parameters
        ----------
        buffer : TBuffer
            バッファ
        offset : int
            オフセット

        Returns
        -------
        tuple[MVector3D, int]
            MVector3Dデータ
            オフセット
        """
        return self.read_to_model(
            [("x", float), ("y", float), ("z", float)],
            MVector3D(),
        )

    def read_MVector4D(
        self,
    ) -> MVector4D:
        """
        MVector4Dの解凍

        Parameters
        ----------
        buffer : TBuffer
            バッファ
        offset : int
            オフセット

        Returns
        -------
        tuple[MVector4D, int]
            MVector4Dデータ
            オフセット
        """
        return self.read_to_model(
            [("x", float), ("y", float), ("z", float), ("w", float)],
            MVector4D(),
        )

    def read_MQuaternion(
        self,
    ) -> MQuaternion:
        """
        MQuaternionの解凍

        Parameters
        ----------
        buffer : TBuffer
            バッファ
        offset : int
            オフセット

        Returns
        -------
        tuple[MQuaternion, int]
            MQuaternionデータ
            オフセット
        """
        return self.read_to_model(
            [("x", float), ("y", float), ("z", float), ("scalar", float)],
            MQuaternion(),
        )

    def read_to_model(
        self,
        formats: list[
            tuple[
                str,
                type,
            ]
        ],
        model: TBaseModel,
    ) -> TBaseModel:
        """
        フォーマットに沿って解凍する

        Parameters
        ----------
        buffer : TBuffer
            バッファ
        offset : int
            オフセット
        formats : list[tuple[str, type]]
            フォーマットリスト（属性名、属性クラス）
        model : TBaseModel
            設定対象モデルデータ

        Returns
        -------
        tuple[TBaseModel, int]
            解凍済みモデルデータ
            移動済みオフセット
        """
        v: Any = None
        for attr_name, format_type in formats:
            if isinstance(format_type(), BaseModel):
                submodel: TBaseModel = format_type()
                v = self.read_to_model([(attr_name, submodel.__class__)], submodel)
            else:
                v = self.read_by_format(format_type)
            model.__setattr__(attr_name, v)
        return model

    def read_byte(self) -> int:
        """byteを読み込む"""
        return int(self.read_by_format(np.byte))

    def read_ubyte(self) -> int:
        """byteを読み込む"""
        return int(self.read_by_format(np.ubyte))

    def read_short(self) -> int:
        """shortを読み込む"""
        return int(self.read_by_format(np.short))

    def read_ushort(self) -> int:
        """ushortを読み込む"""
        return int(self.read_by_format(np.ushort))

    def read_int(self) -> int:
        """intを読み込む"""
        return int(self.read_by_format(int))

    def read_uint(self) -> int:
        """uintを読み込む"""
        return int(self.read_by_format(np.uint))

    def read_int8(self) -> np.int8:
        """int8を読み込む"""
        return np.int8(self.read_by_format(np.int8))

    def read_uint8(self) -> np.uint8:
        """uint8を読み込む"""
        return np.uint8(self.read_by_format(np.uint8))

    def read_float(self) -> float:
        """floatを読み込む"""
        return float(self.read_by_format(float))

    def read_double(self) -> np.double:
        """doubleを読み込む"""
        return np.double(self.read_by_format(np.double))

    def read_str(self) -> str:
        """strを読み込む"""
        return str(self.read_by_format(str))

    def read_by_format(
        self,
        format_type: type[
            Union[
                str,
                np.byte,
                np.ubyte,
                np.short,
                np.ushort,
                int,
                np.uint,
                np.int8,
                np.uint8,
                float,
                np.double,
            ]
        ],
    ) -> Union[
        str,
        np.byte,
        np.ubyte,
        np.short,
        np.ushort,
        int,
        np.uint,
        np.int8,
        np.uint8,
        float,
        np.double,
    ]:
        """
        指定したクラスタイプに従ってバッファから解凍して該当クラスインスタンスを返す

        Parameters
        ----------
        format_type : type[ Union[ str, np.byte, np.ubyte, np.short, np.ushort,
                                    int, np.uint, np.int8, np.uint8, float, np.double, ] ]
            str:        文字列
            np.byte:    sbyte	    : 1  - 符号あり  | char
            np.ubyte:   byte	    : 1  - 符号なし  | unsigned char
            np.short:   short	    : 2  - 符号あり  | short
            np.ushort   ushort	    : 2  - 符号なし  | unsigned short
            int:        int 	    : 4  - 符号あり  | int (32bit固定)
            np.uint     uint	    : 4  - 符号なし  | unsigned int
            np.int8     longlong    : 8  - 符号あり  | long long
            np.uint8    ulonglong   : 8  - 符号なし  | unsigned long long
            float       float	    : 4  - 単精度実数 | float
            np.double   double	    : 8  - 浮動小数点数 | double

        Returns
        -------
        Union[ str, np.byte, np.ubyte, np.short, np.ushort, int, np.uint, np.int8, np.uint8, float, np.double, ]
            指定されたtypeに相当するインスタンス

        Raises
        ------
        MParseException
            対象外フォーマットタイプを指定された場合
        """
        if format_type == str:
            format_name = "s"
            return self.read_text()
        elif format_type == np.byte:
            format_name = "b"
            format_size = 1
        elif format_type == np.ubyte:
            format_name = "B"
            format_size = 1
        elif format_type == np.short:
            format_name = "h"
            format_size = 2
        elif format_type == np.ushort:
            format_name = "H"
            format_size = 2
        elif format_type == int:
            format_name = "i"
            format_size = 4
        elif format_type == np.uint:
            format_name = "I"
            format_size = 4
        elif format_type == np.int8:
            format_name = "q"
            format_size = 8
        elif format_type == np.uint8:
            format_name = "Q"
            format_size = 8
        elif format_type == float:
            format_name = "f"
            format_size = 4
        elif format_type == np.double:
            format_name = "d"
            format_size = 8
        else:
            raise MParseException(
                "Unknown read_by_format format_type: %s", [format_type]
            )

        return format_type(self.unpack(format_name, format_size))

    def unpack(
        self,
        format: str,
        format_size: int,
    ) -> Any:
        """
        バイナリを解凍

        Parameters
        ----------
        buffer : TBuffer
            バッファ
        offset : int
            オフセット
        format : str
            読み取りフォーマット

        Returns
        -------
        Any
            読み取り結果
        """
        # バイナリ読み取り
        b: tuple = struct.unpack_from(format, self.buffer, self.offset)
        # オフセット加算
        self.offset += format_size

        if b:
            return b[0]

        return None
