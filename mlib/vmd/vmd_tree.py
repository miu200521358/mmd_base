from mlib.base.math import MMatrix4x4, MVector3D


class VmdBoneFrameTree:
    def __init__(
        self,
        global_matrix: MMatrix4x4,
        local_matrix: MMatrix4x4,
        position: MVector3D,
    ) -> None:
        self.global_matrix = global_matrix
        self.local_matrix = local_matrix
        self.position = position


class VmdBoneFrameTrees:
    LAST_NAME = "-1"

    def __init__(self) -> None:
        self.data: dict[tuple[int, str], VmdBoneFrameTree] = {}

    def append(
        self,
        fno: int,
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
        bone_name: ボーン名
        global_matrix : 自身のボーン位置を加味した行列
        local_matrix : 自身のボーン位置を加味しない行列
        position : ボーン変形後のグローバル位置
        """
        self.data[(fno, bone_name)] = VmdBoneFrameTree(global_matrix, local_matrix, position)

    def __getitem__(self, key) -> VmdBoneFrameTree:
        return self.data[key]

    def exists(self, fno: int, bone_name: str) -> bool:
        """既に該当ボーンの情報が登録されているか"""
        return (fno, bone_name) in self.data
