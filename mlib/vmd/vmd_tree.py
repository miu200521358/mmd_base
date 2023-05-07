from mlib.base.math import MMatrix4x4, MVector3D


class VmdBoneFrameTree:
    def __init__(self, matrix: MMatrix4x4, position: MVector3D, tail_position: MVector3D) -> None:
        self.matrix = matrix
        self.position = position
        self.tail_position = tail_position


class VmdBoneFrameTrees:
    LAST_NAME = "-1"

    def __init__(self) -> None:
        self.data: dict[tuple[int, str], VmdBoneFrameTree] = {}

    def append(self, fno: int, bone_name: str, matrix: MMatrix4x4, position: MVector3D, tail_position: MVector3D):
        """
        ボーン変形結果追加

        Parameters
        ----------
        fno: キーフレ
        bone_name: ボーン名
        matrix : 親ボーンから見た行列
        position : ボーン変形後のグローバル位置
        """
        self.data[(fno, bone_name)] = VmdBoneFrameTree(matrix, position, tail_position)

    def __getitem__(self, key) -> VmdBoneFrameTree:
        return self.data[key]

    def exists(self, fno: int, bone_name: str) -> bool:
        """既に該当ボーンの情報が登録されているか"""
        return (fno, bone_name) in self.data
