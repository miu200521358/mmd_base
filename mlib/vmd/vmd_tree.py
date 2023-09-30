from typing import Iterator, Optional

import numpy as np

from mlib.core.math import MMatrix4x4, MQuaternion, MVector3D


class VmdBoneFrameTree:
    __slots__ = (
        "fno",
        "bone_index",
        "bone_name",
        "global_matrix_ary",
        "local_matrix_ary",
        "frame_position_matrix_ary",
        "frame_rotation_matrix_ary",
        "cache_global_matrix",
        "cache_local_matrix",
        "cache_global_matrix_no_scale",
        "cache_local_matrix_no_scale",
        "cache_position",
        "cache_frame_position",
        "cache_frame_rotation",
    )

    def __init__(
        self,
        fno: int,
        bone_index: int,
        bone_name: str,
        global_matrix_ary: np.ndarray,
        local_matrix_ary: np.ndarray,
        frame_position_matrix_ary: np.ndarray,
        frame_rotation_matrix_ary: np.ndarray,
    ) -> None:
        self.fno = fno
        self.bone_index = bone_index
        self.bone_name = bone_name
        self.global_matrix_ary = global_matrix_ary
        self.local_matrix_ary = local_matrix_ary
        self.frame_position_matrix_ary = frame_position_matrix_ary
        self.frame_rotation_matrix_ary = frame_rotation_matrix_ary
        self.cache_global_matrix: Optional[MMatrix4x4] = None
        self.cache_local_matrix: Optional[MMatrix4x4] = None
        self.cache_global_matrix_no_scale: Optional[MMatrix4x4] = None
        self.cache_local_matrix_no_scale: Optional[MMatrix4x4] = None
        self.cache_position: Optional[MVector3D] = None
        self.cache_frame_position: Optional[MVector3D] = None
        self.cache_frame_rotation: Optional[MQuaternion] = None

    @property
    def global_matrix(self) -> MMatrix4x4:
        if self.cache_global_matrix is not None:
            return self.cache_global_matrix
        self.cache_global_matrix = MMatrix4x4(self.global_matrix_ary)
        return self.cache_global_matrix

    @property
    def local_matrix(self) -> MMatrix4x4:
        if self.cache_local_matrix is not None:
            return self.cache_local_matrix
        self.cache_local_matrix = MMatrix4x4(self.local_matrix_ary)
        return self.cache_local_matrix

    @property
    def global_matrix_no_scale(self) -> MMatrix4x4:
        if self.cache_global_matrix_no_scale is not None:
            return self.cache_global_matrix_no_scale

        global_matrix = self.global_matrix

        rot = global_matrix.to_quaternion()
        pos = global_matrix.to_position()

        no_scale_mat = MMatrix4x4()
        no_scale_mat.translate(pos)
        no_scale_mat.rotate(rot)
        self.cache_global_matrix_no_scale = no_scale_mat
        return self.cache_global_matrix_no_scale

    @property
    def local_matrix_no_scale(self) -> MMatrix4x4:
        if self.cache_local_matrix_no_scale is not None:
            return self.cache_local_matrix_no_scale

        local_matrix = self.local_matrix

        rot = local_matrix.to_quaternion()
        pos = local_matrix.to_position()

        no_scale_mat = MMatrix4x4()
        no_scale_mat.translate(pos)
        no_scale_mat.rotate(rot)
        self.cache_local_matrix_no_scale = no_scale_mat
        return self.cache_local_matrix_no_scale

    @property
    def position(self) -> MVector3D:
        if self.cache_position is not None:
            return self.cache_position
        self.cache_position = MVector3D(*self.global_matrix_ary[:3, 3])
        return self.cache_position

    @property
    def frame_position(self) -> MVector3D:
        if self.cache_frame_position is not None:
            return self.cache_frame_position
        self.cache_frame_position = MVector3D(*self.frame_position_matrix_ary[:3, 3])
        return self.cache_frame_position

    @property
    def frame_rotation(self) -> MQuaternion:
        if self.cache_frame_rotation is not None:
            return self.cache_frame_rotation

        self.cache_frame_rotation = MMatrix4x4(self.frame_rotation_matrix_ary).to_quaternion()
        return self.cache_frame_rotation


class VmdBoneFrameTrees:
    __slots__ = (
        "_names",
        "_indexes",
        "data",
    )

    def __init__(self) -> None:
        self._names: list[str] = []
        self._indexes: list[int] = []
        self.data: dict[tuple[int, str], VmdBoneFrameTree] = {}

    def append(
        self,
        fno: int,
        bone_index: int,
        bone_name: str,
        global_matrix: np.ndarray,
        local_matrix: np.ndarray,
        frame_position_matrix: np.ndarray,
        frame_rotation_matrix: np.ndarray,
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

        self.data[(fno, bone_name)] = VmdBoneFrameTree(
            fno,
            bone_index,
            bone_name,
            global_matrix,
            local_matrix,
            frame_position_matrix,
            frame_rotation_matrix,
        )
        if bone_name not in self._names:
            self._names.append(bone_name)
        if fno not in self._indexes:
            self._indexes.append(fno)

    def __getitem__(self, key) -> VmdBoneFrameTree:
        return self.data[key]

    def exists(self, fno: int, bone_name: str) -> bool:
        """既に該当ボーンの情報が登録されているか"""
        return (fno, bone_name) in self.data

    def __len__(self) -> int:
        return len(self._indexes)

    def __iter__(self) -> Iterator[VmdBoneFrameTree]:
        return iter([self.data[k] for k in list(self.data.keys())])

    @property
    def indexes(self) -> list[int]:
        return sorted(self._indexes)

    @property
    def names(self) -> list[str]:
        return self._names
