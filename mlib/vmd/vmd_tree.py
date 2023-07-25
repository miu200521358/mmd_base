from typing import Iterator

from mlib.base.math import MMatrix4x4, MVector3D


class VmdBoneFrameTree:
    __slots__ = (
        "fno",
        "bone_index",
        "bone_name",
        "global_matrix",
        "local_matrix",
        "position",
    )

    def __init__(
        self,
        fno: int,
        bone_index: int,
        bone_name: str,
        global_matrix: MMatrix4x4,
        local_matrix: MMatrix4x4,
        position: MVector3D,
    ) -> None:
        self.fno = fno
        self.bone_index = bone_index
        self.bone_name = bone_name
        self.global_matrix = global_matrix
        self.local_matrix = local_matrix
        self.position = position


class VmdBoneFrameTrees:
    __slots__ = (
        "_indexes",
        "data",
    )

    def __init__(self) -> None:
        self._indexes: dict[tuple[int, int], tuple[int, str]] = {}
        self.data: dict[tuple[int, str], VmdBoneFrameTree] = {}

    def append(
        self,
        fno: int,
        bone_index: int,
        bone_name: str,
        global_matrix: MMatrix4x4,
        local_matrix: MMatrix4x4,
        position: MVector3D,
    ):
        """
        ボーン変形結果追加

        Parameters
        ----------
        fno: キーフレ
        bone_index: ボーンINDEX
        bone_name: ボーン名
        global_matrix : 自身のボーン位置を加味した行列
        local_matrix : 自身のボーン位置を加味しない行列
        position : ボーン変形後のグローバル位置
        """
        self.data[(fno, bone_name)] = VmdBoneFrameTree(fno, bone_index, bone_name, global_matrix, local_matrix, position)
        self._indexes[(fno, bone_index)] = (fno, bone_name)

    def __getitem__(self, key) -> VmdBoneFrameTree:
        return self.data[key]

    def exists(self, fno: int, bone_name: str) -> bool:
        """既に該当ボーンの情報が登録されているか"""
        return (fno, bone_name) in self.data

    def __len__(self) -> int:
        return len(self._indexes)

    def __iter__(self) -> Iterator[VmdBoneFrameTree]:
        return iter([self.data[k] for k in list(self.data.keys())])
