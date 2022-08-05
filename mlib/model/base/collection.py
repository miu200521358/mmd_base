import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar, Union

from mlib.base import BaseModel, Encoding
from mlib.model.base.part import TBaseIndexModel, TBaseIndexNameModel


class BaseIndexListModel(Generic[TBaseIndexModel]):
    """BaseIndexModelのリスト基底クラス"""

    __slots__ = ["data", "__iter_index"]

    def __init__(self, data: list[TBaseIndexModel] = None) -> None:
        """
        モデルリスト

        Parameters
        ----------
        data : list[TBaseIndexModel], optional
            リスト, by default []
        """
        super().__init__()
        self.data = data or []
        self.__iter_index = 0

    def __getitem__(self, index: int) -> Optional[TBaseIndexModel]:
        return self.get(index)

    def __setitem__(self, index: int, value: TBaseIndexModel):
        self.data[index] = value

    def __delitem__(self, index: int):
        del self.data[index]

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

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        self.__iter_index = -1
        return self.data

    def __next__(self):
        self.__iter_index += 1
        if self.__iter_index >= len(self.data):
            raise StopIteration
        return self.data[self.__iter_index]


TBaseIndexListModel = TypeVar("TBaseIndexListModel", bound=BaseIndexListModel)


class BaseIndexNameListModel(Generic[TBaseIndexNameModel]):
    """BaseIndexNameModelのリスト基底クラス"""

    __slots__ = ["data", "__names"]

    def __init__(self, data: list[TBaseIndexNameModel] = None) -> None:
        """
        モデルリスト

        Parameters
        ----------
        data : list[TBaseIndexNameModel], optional
            リスト, by default []
        """
        super().__init__()
        self.data: list[TBaseIndexNameModel] = data or []
        self.__names = dict([(v.name, v.index) for v in self.data])

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
        if name not in self.__names:
            if required:
                raise KeyError(f"Not Found: {name}")
            else:
                return None
        return self.data[self.__names[name]]

    def append(self, v: TBaseIndexNameModel) -> None:
        v.index = len(self.data)
        self.data.append(v)
        if v.name not in self.__names:
            # 名前は先勝ちで保持
            self.__names[v.name] = v.index


class BaseIndexDictModel(Generic[TBaseIndexModel]):
    """BaseIndexModelの辞書基底クラス"""

    __slots__ = ["data", "__keys", "__iter_index"]

    def __init__(self, data: Dict[int, TBaseIndexModel] = None) -> None:
        """
        モデル辞書

        Parameters
        ----------
        data : Dict[TBaseIndexModel], optional
            辞書, by default {}
        """
        super().__init__()
        self.data: Dict[int, TBaseIndexModel] = data or {}
        self.__keys: list[int] = sorted(list(self.data.keys()))
        self.__iter_index: int = 0

    def __getitem__(self, index: int) -> Optional[TBaseIndexModel]:
        return self.get(index)

    def __setitem__(self, index: int, value: TBaseIndexModel):
        self.data[index] = value
        self.__keys = sorted(list(self.data.keys()))

    def __delitem__(self, index: int):
        del self.data[index]

    def append(self, value: TBaseIndexModel):
        self.data[value.index] = value

    def get(self, index: int, required: bool = False) -> Optional[TBaseIndexModel]:
        """
        辞書から要素を取得する

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
        if index not in self.data:
            if required:
                raise KeyError(f"Not Found: {index}")
            else:
                return None
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        self.__iter_index = -1
        return self.__keys

    def __next__(self):
        self.__iter_index += 1
        if self.__iter_index >= len(self.__keys):
            raise StopIteration
        return self.data[self.__keys[self.__iter_index]]


TBaseIndexDictModel = TypeVar("TBaseIndexDictModel", bound=BaseIndexDictModel)


class BaseIndexNameDictModel(Generic[TBaseIndexNameModel]):
    """BaseIndexNameModelの辞書基底クラス"""

    __slots__ = ["data", "__iter_name", "__iter_index"]

    def __init__(self, data: Dict[str, Dict[int, TBaseIndexNameModel]] = None) -> None:
        """
        モデル辞書

        Parameters
        ----------
        data : Dict[TBaseIndexNameModel], optional
            辞書, by default {}
        """
        super().__init__()
        self.data: Dict[str, Dict[int, TBaseIndexNameModel]] = data or {}
        self.__iter_name: str = ""
        self.__iter_index: int = 0

    def __getitem__(
        self, key: Any
    ) -> Optional[Union[TBaseIndexNameModel, Dict[int, TBaseIndexNameModel]]]:
        if isinstance(key, tuple):
            name, index = key
        elif isinstance(key, str):
            name = key
            index = None
        return self.get(name, index)

    def append(self, value: TBaseIndexNameModel):
        if value.name not in self.data:
            self.data[value.name] = {}
        self.data[value.name][value.index] = value

    def __delitem__(self, name: str, index: int = None):
        if index is None:
            del self.data[name]
        else:
            del self.data[name][index]

    def keys(self) -> Dict[str, list[int]]:
        return dict(
            [(k, [v for v in sorted(list(self.data[k]))]) for k in self.data.keys()]
        )

    def get(
        self, name: str, index: int = None, required: bool = False
    ) -> Optional[Union[TBaseIndexNameModel, Dict[int, TBaseIndexNameModel]]]:
        """
        辞書から要素を取得する

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
        if name not in self.data:
            if required:
                raise KeyError(f"Not Found Name: {(name, index)}")
            else:
                return None

        if index is None:
            # INDEXが未指定の場合、辞書そのものを返す
            return self.data[name]

        elif index not in self.data[name]:
            if required:
                raise KeyError(f"Not Found Index: {(name, index)}")
            else:
                return None
        return self.data[name][index]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self, name: str):
        self.__iter_index = -1
        self.__iter_name = name
        return self.data[name]

    def __next__(self):
        self.__iter_index += 1
        if self.__iter_index >= len(self.keys[self.__iter_name]):
            raise StopIteration
        return self.data[self.__iter_name][
            self.keys[self.__iter_name][self.__iter_index]
        ]


TBaseIndexNameDictModel = TypeVar(
    "TBaseIndexNameDictModel", bound=BaseIndexNameDictModel
)


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
