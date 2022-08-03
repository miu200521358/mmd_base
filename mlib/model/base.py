import hashlib
from abc import ABC, abstractmethod
from enum import Enum, unique
from typing import Generic, Optional, TypeVar

import numpy as np
from mlib.base import BaseModel, Encoding
from mlib.math import MQuaternion, MVector3D


@unique
class Switch(Enum):
    """ONOFFスイッチ"""

    OFF = 0
    ON = 1


class BaseRotationModel(BaseModel):
    def __init__(self, v_radians: MVector3D = MVector3D()) -> None:
        super().__init__()
        self.__radians = MVector3D()
        self.__degrees = MVector3D()
        self.__qq = MQuaternion()
        self.radians = v_radians

    @property
    def qq(self) -> MQuaternion:
        """
        回転情報をクォータニオンとして受け取る
        """
        return self.__qq

    @property
    def radians(self) -> MVector3D:
        """
        回転情報をラジアンとして受け取る
        """
        return self.__radians

    @radians.setter
    def radians(self, v: MVector3D):
        """
        ラジアンを回転情報として設定する

        Parameters
        ----------
        v : MVector3D
            ラジアン
        """
        self.__radians = v
        self.__degrees = MVector3D(*np.degrees(v.vector))
        self.__qq = MQuaternion.from_euler_degrees(self.degrees)

    @property
    def degrees(self) -> MVector3D:
        """
        回転情報を度として受け取る
        """
        return self.__degrees

    @degrees.setter
    def degrees(self, v: MVector3D):
        """
        度を回転情報として設定する

        Parameters
        ----------
        v : MVector3D
            度
        """
        self.__degrees = v
        self.__radians = MVector3D(*np.radians(v.vector))
        self.__qq = MQuaternion.from_euler_degrees(v)


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


class BaseIndexListModel(Generic[TBaseIndexModel], BaseModel):
    """BaseIndexModelのリスト基底クラス"""

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
        self.__data = data
        self.__index = 0

    def __getitem__(self, index: int) -> Optional[TBaseIndexModel]:
        return self.get(index)

    def __setitem__(self, index: int, value: TBaseIndexModel):
        self.__data[index] = value

    def __delitem__(self, index: int):
        del self.__data[index]

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
        if index >= len(self.__data):
            if required:
                raise KeyError(f"Not Found: {index}")
            else:
                return None
        return self.__data[index]

    def append(self, v: TBaseIndexModel) -> None:
        v.index = len(self.__data)
        self.__data.append(v)

    def __len__(self) -> int:
        return len(self.__data)

    def __iter__(self):
        self.__index = -1
        return self.__data

    def __next__(self):
        self.__index += 1
        if self.__index >= len(self.__data):
            raise StopIteration
        return self.__data[self.__index]


TBaseIndexListModel = TypeVar("TBaseIndexListModel", bound=BaseIndexListModel)


class BaseIndexNameListModel(Generic[TBaseIndexNameModel], BaseModel):
    """BaseIndexNameModelのリスト基底クラス"""

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

    def __getitem__(self, index: int) -> Optional[TBaseIndexNameModel]:
        return self.get(index)

    def __setitem__(self, index: int, value: TBaseIndexNameModel):
        self.data[index] = value

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


class BaseHashModel(BaseModel, ABC):
    """
    ハッシュ機能付きモデル

    Parameters
    ----------
    path : str, optional
        パス, by default ""
    """

    def __init__(self, path: str = "") -> None:
        super().__init__()
        self.path = path
        self.digest = ""

    @abstractmethod
    def get_name(self) -> str:
        """モデル内の名前に相当する値を返す"""
        pass

    def hexdigest(self) -> str:
        """モデルデータのハッシュ値を取得する"""
        sha1 = hashlib.sha1()

        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(2048 * sha1.block_size), b""):
                sha1.update(chunk)

        sha1.update(chunk)

        # ファイルパスをハッシュに含める
        sha1.update(self.path.encode(Encoding.UTF_8.value))

        return sha1.hexdigest()


TBaseHashModel = TypeVar("TBaseHashModel", bound=BaseHashModel)
