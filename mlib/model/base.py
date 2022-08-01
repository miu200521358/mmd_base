import hashlib
from enum import Enum
from typing import Any, Optional, TypeVar

import numpy as np
from mlib.base import BaseModel
from mlib.math import MQuaternion, MVector3D


class Switch(Enum):
    """ONOFFスイッチ"""

    OFF = 0
    ON = 1


class Pair(BaseModel):

    __slots__ = ["key", "value"]

    def __init__(self, key: Any, value: Any) -> None:
        self.key = key
        self.value = value


class IntPair(Pair):
    def __init__(self, key: str, value: int) -> None:
        super().__init__(key, value)


class BaseRotationModel(BaseModel):
    def __init__(self, v_radians: MVector3D = MVector3D()) -> None:
        super().__init__()
        self.__radians = v_radians

    @property
    def radians(self):
        return self.__radians

    @radians.setter
    def radians(self, v: MVector3D):
        self.__radians = v
        self.qq = MQuaternion.from_euler_degrees(*np.degrees(v.vector))


class BaseIndexModel(BaseModel):
    """
    INDEXを持つ基底クラス
    """

    def __init__(self, index: int = -1) -> None:
        """
        初期化

        Parameters
        ----------
        index : int, optional
            INDEX, by default -1
        """
        super().__init__()
        self.index = index


TBaseIndexModel = TypeVar("TBaseIndexModel", bound=BaseIndexModel)


class BaseIndexNameModel(BaseIndexModel):
    """
    INDEXと名前を持つ基底クラス
    """

    def __init__(self, index: int = -1, name: str = "", english_name: str = "") -> None:
        """
        初期化

        Parameters
        ----------
        index : int, optional
            INDEX, by default -1
        name : str, optional
            名前, by default ""
        english_name : str, optional
            英語名, by default ""
        """
        super().__init__()
        self.index = index
        self.name = name
        self.english_name = english_name


TBaseIndexNameModel = TypeVar("TBaseIndexNameModel", bound=BaseIndexNameModel)


class BaseIndexListModel(BaseModel):

    __slots__ = ["data"]

    def __init__(self, data: list[TBaseIndexModel] = []) -> None:
        """
        モデルリスト

        Parameters
        ----------
        data : list[TBaseIndexModel], optional
            リスト, by default []
        """
        super().__init__()
        self.data = data

    def get(self, index: int, required: bool = False) -> Optional[TBaseIndexModel]:
        """
        リストから要素を取得する

        Parameters
        ----------
        index : int
            インデックス番号
        required : bool, optional
            必須要素であるか, by default False

        Returns
        -------
        Optional[TBaseIndexModel]
            要素（必須でない場合かつ見つからなければNone）
        """
        if index >= len(self.data):
            if required:
                raise KeyError(f"Not Found: {index}")
            else:
                return None
        return self.data[index]

    def append(self, v: TBaseIndexModel) -> None:
        v.index = len(self.data)
        self.data.append(v)


class BaseIndexNameListModel(BaseModel):

    __slots__ = ["data"]

    def __init__(self, data: list[TBaseIndexNameModel] = []) -> None:
        """
        モデルリスト

        Parameters
        ----------
        data : list[TBaseIndexNameModel], optional
            リスト, by default []
        """
        super().__init__()
        self.data = data
        self.names = dict([(v.name, v.index) for v in self.data])

    def get(self, index: int, required: bool = False) -> Optional[TBaseIndexNameModel]:
        """
        リストから要素を取得する

        Parameters
        ----------
        index : int
            インデックス番号
        required : bool, optional
            必須要素であるか, by default False

        Returns
        -------
        Optional[TBaseIndexNameModel]
            要素（必須でない場合かつ見つからなければNone）
        """
        if index >= len(self.data):
            if required:
                raise KeyError(f"Not Found: {index}")
            else:
                return None
        return self.data[index]

    def get_by_name(
        self, name: str, required: bool = False
    ) -> Optional[TBaseIndexNameModel]:
        """
        リストから要素を取得する

        Parameters
        ----------
        name : str
            名前
        required : bool, optional
            必須要素であるか, by default False

        Returns
        -------
        Optional[TBaseIndexNameModel]
            要素（必須でない場合かつ見つからなければNone）
        """
        if name not in self.names:
            if required:
                raise KeyError(f"Not Found: {name}")
            else:
                return None
        return self.data[self.names[name]]

    def append(self, v: TBaseIndexNameModel) -> None:
        v.index = len(self.data)
        self.data.append(v)
        if v.name not in self.names:
            # 名前は先勝ちで保持
            self.names[v.name] = v.index


class BaseHashModel(BaseModel):
    def __init__(self, path: str = "") -> None:
        """
        ハッシュ機能付きモデル

        Parameters
        ----------
        path : str, optional
            パス, by default ""
        """
        super().__init__()
        self.path = path

    def hexdigest(self) -> str:
        sha1 = hashlib.sha1()

        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(2048 * sha1.block_size), b""):
                sha1.update(chunk)

        sha1.update(chunk)

        # ファイルパスをハッシュに含める
        sha1.update(self.path.encode("utf-8"))

        return sha1.hexdigest()
