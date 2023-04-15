import hashlib
from bisect import bisect_left
from typing import Generic, Optional, TypeVar

from mlib.base.base import BaseModel, Encoding
from mlib.base.part import BaseIndexModel, BaseIndexNameModel


TBaseIndexModel = TypeVar("TBaseIndexModel", bound=BaseIndexModel)
TBaseIndexNameModel = TypeVar("TBaseIndexNameModel", bound=BaseIndexNameModel)


class BaseIndexDictModel(Generic[TBaseIndexModel], BaseModel):
    """BaseIndexModelのリスト基底クラス"""

    __slots__ = (
        "data",
        "indexes",
        "_iter_index",
        "_size",
    )

    def __init__(self) -> None:
        """モデルリスト"""
        super().__init__()
        self.data: dict[int, TBaseIndexModel] = {}
        self.indexes: list[int] = []
        self._iter_index = 0
        self._size = 0

    def create(self) -> "TBaseIndexModel":
        raise NotImplementedError

    def __getitem__(self, index: int) -> TBaseIndexModel:
        if 0 > index:
            # マイナス指定の場合、後ろからの順番に置き換える
            index = len(self.data) + index
            return self.data[self.indexes[index]]
        if index in self.data:
            return self.data[index]

        # なかったら追加
        self.append(self.create())
        return self.data[index]

    def __setitem__(self, index: int, v: TBaseIndexModel) -> None:
        self.data[index] = v

    def append(self, value: TBaseIndexModel, is_sort: bool = True) -> None:
        if 0 > value.index:
            value.index = len(self.data)
        self.data[value.index] = value
        if is_sort:
            self.sort_indexes()

    def sort_indexes(self) -> None:
        self.indexes = sorted(self.data.keys()) if self.data else []

    def __delitem__(self, index: int) -> None:
        del self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        self._iter_index = -1
        self._size = len(self.indexes)
        return self

    def __next__(self) -> TBaseIndexModel:
        self._iter_index += 1
        if self._iter_index >= self._size:
            raise StopIteration
        return self.data[self.indexes[self._iter_index]]

    def __contains__(self, key: int) -> bool:
        return key in self.data

    def __bool__(self) -> bool:
        return 0 < len(self.data)

    @property
    def last_index(self) -> int:
        return max(self.data.keys())


TBaseIndexDictModel = TypeVar("TBaseIndexDictModel", bound=BaseIndexDictModel)


class BaseIndexNameDictModel(Generic[TBaseIndexNameModel], BaseModel):
    """BaseIndexNameModelの辞書基底クラス"""

    __slots__ = (
        "name",
        "data",
        "cache",
        "indexes",
        "_names",
        "_iter_index",
        "_size",
    )

    def __init__(self, name: str = "") -> None:
        """モデル辞書"""
        super().__init__()
        self.name = name
        self.data: dict[int, TBaseIndexNameModel] = {}
        self.cache: dict[int, TBaseIndexNameModel] = {}
        self.indexes: list[int] = []
        self._names: dict[str, int] = {}
        self._iter_index = 0
        self._size = 0

    def __getitem__(self, key: int | str) -> TBaseIndexNameModel:
        if isinstance(key, int):
            return self.get_by_index(key)
        return self.get_by_name(key)

    def __delitem__(self, key: int | str) -> None:
        if key in self:
            if isinstance(key, int):
                del self.data[key]
            else:
                del self.data[self._names[key]]

    def __setitem__(self, index: int, v: TBaseIndexNameModel) -> None:
        self.data[index] = v

    def append(self, value: TBaseIndexNameModel, is_sort: bool = True) -> None:
        if 0 > value.index:
            value.index = len(self.data)

        if value.name and value.name not in self._names:
            # 名前は先勝ちで保持
            self._names[value.name] = value.index

        self.data[value.index] = value
        if is_sort:
            self.sort_indexes()

    @property
    def names(self) -> list[str]:
        return list(self._names.keys())

    @property
    def last_index(self) -> int:
        return max(self.data.keys()) if self.data else 0

    @property
    def last_name(self) -> str:
        if not self.data:
            return ""
        return self[-1].name

    def get_by_index(self, index: int) -> TBaseIndexNameModel:
        """
        リストから要素を取得する

        Parameters
        ----------
        index : int
            インデックス番号

        Returns
        -------
        TBaseIndexNameModel
            要素
        """
        if 0 > index:
            # マイナス指定の場合、後ろからの順番に置き換える
            index = len(self.data) + index
            return self.data[self.indexes[index]]
        return self.data[index]

    def get_by_name(self, name: str) -> TBaseIndexNameModel:
        """
        リストから要素を取得する

        Parameters
        ----------
        name : str
            名前

        Returns
        -------
        TBaseIndexNameModel
            要素
        """
        return self.data[self._names[name]]

    def sort_indexes(self) -> None:
        self.indexes = sorted(self.data.keys()) if self.data else []

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        self._iter_index = -1
        self._size = len(self.indexes)
        return self

    def __next__(self) -> TBaseIndexNameModel:
        self._iter_index += 1
        if self._iter_index >= self._size:
            raise StopIteration
        return self.data[self.indexes[self._iter_index]]

    def __contains__(self, key: int | str) -> bool:
        if isinstance(key, int):
            return key in self.data
        return key in self._names

    def __bool__(self) -> bool:
        return 0 < len(self.data)

    def range_indexes(self, index: int, indexes: Optional[list[int]] = None) -> tuple[int, int, int]:
        """
        指定されたINDEXの前後を返す

        Parameters
        ----------
        index : int
            指定INDEX

        Returns
        -------
        tuple[int, int]
            INDEXがデータ内にある場合: index, index, index
            INDEXがデータ内にない場合: 前のindex, 対象INDEXに相当する場所にあるINDEX, 次のindex
                prev_idx == idx: 指定されたINDEXが一番先頭
                idx == next_idx: 指定されたINDEXが一番最後
        """
        if not indexes:
            indexes = self.indexes
        if not indexes or index in self.data:
            return index, index, index

        # index がない場合、前後のINDEXを取得する

        idx = bisect_left(indexes, index)
        if 0 == idx:
            prev_index = 0
        else:
            prev_index = indexes[idx - 1]
        if idx == len(indexes):
            next_index = max(indexes)
        else:
            next_index = indexes[idx]

        return (
            prev_index,
            index,
            next_index,
        )


TBaseIndexNameDictModel = TypeVar("TBaseIndexNameDictModel", bound=BaseIndexNameDictModel)


class BaseIndexNameDictWrapperModel(Generic[TBaseIndexNameDictModel], BaseModel):
    """BaseIndexNameDictModelの辞書基底クラス"""

    __slots__ = (
        "data",
        "cache",
        "_names",
        "_iter_index",
        "_size",
    )

    def __init__(self) -> None:
        """モデル辞書"""
        super().__init__()
        self.data: dict[str, TBaseIndexNameDictModel] = {}
        self.cache: dict[str, TBaseIndexNameDictModel] = {}
        self._names: list[str] = []
        self._iter_index = 0
        self._size = 0

    def create(self, key: str) -> TBaseIndexNameDictModel:
        raise NotImplementedError

    def __getitem__(self, key: str) -> TBaseIndexNameDictModel:
        if key not in self.data:
            self.append(self.create(key), name=key)
        return self.data[key]

    def filter(self, *keys: str) -> dict[str, TBaseIndexNameDictModel]:
        return dict([(k, v.copy()) for k, v in self.data.items() if k in keys])

    def __delitem__(self, key: str) -> None:
        if key in self:
            del self.data[key]

    def __setitem__(self, v: TBaseIndexNameDictModel) -> None:
        self.data[v.name] = v

    def append(self, value: TBaseIndexNameDictModel, name: Optional[str] = None) -> None:
        if not name:
            name = value.last_name

        if name not in self._names:
            self._names.append(name)
        self.data[name] = value

    @property
    def names(self) -> list[str]:
        return self._names

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        self._iter_index = -1
        self._size = len(self.data)
        return self

    def __next__(self) -> TBaseIndexNameDictModel:
        self._iter_index += 1
        if self._iter_index >= self._size:
            raise StopIteration
        return self.data[self._names[self._iter_index]]

    def __contains__(self, key: str) -> bool:
        return key in self._names

    def __bool__(self) -> bool:
        return 0 < len(self.data)


TBaseIndexNameDictWrapperModel = TypeVar("TBaseIndexNameDictWrapperModel", bound=BaseIndexNameDictWrapperModel)


class BaseHashModel(BaseModel):
    """
    ハッシュ機能付きモデル

    Parameters
    ----------
    path : str, optional
        パス, by default ""
    """

    __slots__ = ("path", "digest")

    def __init__(self, path: str = "") -> None:
        super().__init__()
        self.path = path
        self.digest = ""

    @property
    def name(self) -> str:
        """モデル内の名前に相当する値を返す"""
        raise NotImplementedError()

    def update_digest(self) -> None:
        sha1 = hashlib.sha1()

        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(2048 * sha1.block_size), b""):
                sha1.update(chunk)

        sha1.update(chunk)

        # ファイルパスをハッシュに含める
        sha1.update(self.path.encode(Encoding.UTF_8.value))

        self.digest = sha1.hexdigest()

    def delete(self):
        """削除する準備"""
        pass

    def __bool__(self) -> bool:
        # パスが定義されていたら、中身入り
        return 0 < len(self.path)
