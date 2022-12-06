import numpy as np

from mlib.base.bezier import evaluate
from mlib.base.collection import (
    BaseHashModel,
    BaseIndexDictModel,
    BaseIndexNameDictInnerModel,
    BaseIndexNameDictModel,
)
from mlib.base.math import MMatrix4x4, MMatrix4x4List, MQuaternion
from mlib.pmx.collection import BoneTrees
from mlib.vmd.part import (
    VmdBoneFrame,
    VmdCameraFrame,
    VmdLightFrame,
    VmdMorphFrame,
    VmdShadowFrame,
    VmdShowIkFrame,
)


class VmdBoneNameFrames(BaseIndexNameDictInnerModel[VmdBoneFrame]):
    """
    ボーン名別キーフレ辞書
    """

    def __init__(self, name: str):
        super().__init__(name=name)

    def __getitem__(self, index: int) -> VmdBoneFrame:
        if not self.data:
            # まったくデータがない場合、生成
            return VmdBoneFrame(name=self.name, index=index)

        indices = self.indices()
        if index not in indices:
            # キーフレがない場合、生成したのを返す（保持はしない）
            prev_index, middle_index, next_index = self.range_indexes(index)
            # prevとnextの範囲内である場合、補間曲線ベースで求め直す
            return self.calc(prev_index, middle_index, next_index, index)
        return self.data[index]

    def calc(
        self, prev_index: int, middle_index: int, next_index: int, index: int
    ) -> VmdBoneFrame:
        if index in self.data:
            return self.data[index]

        bf = VmdBoneFrame(name=self.name, index=index)

        if prev_index == middle_index:
            # prevと等しい場合、指定INDEX以前がないので、その次のをコピーして返す
            bf.position = self[next_index].position.copy()
            bf.rotation = self[next_index].rotation.copy()
            bf.interpolations = self[next_index].interpolations.copy()
            return bf
        elif next_index == middle_index:
            # nextと等しい場合は、その前のをコピーして返す
            bf.position = self[prev_index].position.copy()
            bf.rotation = self[prev_index].rotation.copy()
            bf.interpolations = self[prev_index].interpolations.copy()
            return bf

        prev_bf = self[prev_index]
        next_bf = self[next_index]

        _, ry, _ = evaluate(
            next_bf.interpolations.rotation, prev_index, index, next_index
        )
        bf.rotation = MQuaternion.slerp(prev_bf.rotation, next_bf.rotation, ry)

        _, xy, _ = evaluate(
            next_bf.interpolations.translation_x, prev_index, index, next_index
        )
        bf.position.x = (
            prev_bf.position.x + (next_bf.position.x - prev_bf.position.x) * xy
        )

        _, yy, _ = evaluate(
            next_bf.interpolations.translation_y, prev_index, index, next_index
        )
        bf.position.y = (
            prev_bf.position.y + (next_bf.position.y - prev_bf.position.y) * yy
        )

        _, zy, _ = evaluate(
            next_bf.interpolations.translation_z, prev_index, index, next_index
        )
        bf.position.z = (
            prev_bf.position.z + (next_bf.position.z - prev_bf.position.z) * zy
        )

        return bf


class VmdBoneFrames(BaseIndexNameDictModel[VmdBoneFrame, VmdBoneNameFrames]):
    """
    ボーンキーフレ辞書
    """

    def __init__(self):
        super().__init__()

    def create_inner(self, name: str):
        return VmdBoneNameFrames(name=name)

    def get_matrix_by_indexes(
        self, fnos: list[int], bone_trees: BoneTrees
    ) -> dict[int, MMatrix4x4]:
        """
        指定されたキーフレ番号の行列計算結果を返す

        Parameters
        ----------
        fnos : list[int]
            キーフレ番号のリスト
        bone_trees: BoneTrees
            ボーンツリー

        Returns
        -------
        行列辞書（キー：キーフレ番号、値：行列リスト）
        """
        positions = []
        for fno in fnos:
            for bone_tree in bone_trees.values():
                for bone in bone_tree:
                    pass

    def get(self, name: str) -> VmdBoneNameFrames:
        if name not in self.data:
            self.data[name] = self.create_inner(name)

        return self.data[name]


class VmdMorphFrames(
    BaseIndexNameDictModel[VmdMorphFrame, BaseIndexNameDictInnerModel[VmdMorphFrame]]
):
    """
    モーフキーフレリスト
    """

    def __init__(self):
        super().__init__()


class VmdCameraFrames(BaseIndexDictModel[VmdCameraFrame]):
    """
    カメラキーフレリスト
    """

    def __init__(self):
        super().__init__()


class VmdLightFrames(BaseIndexDictModel[VmdLightFrame]):
    """
    照明キーフレリスト
    """

    def __init__(self):
        super().__init__()


class VmdShadowFrames(BaseIndexDictModel[VmdShadowFrame]):
    """
    照明キーフレリスト
    """

    def __init__(self):
        super().__init__()


class VmdShowIkFrames(BaseIndexDictModel[VmdShowIkFrame]):
    """
    IKキーフレリスト
    """

    def __init__(self):
        super().__init__()


class VmdMotion(BaseHashModel):
    """
    VMDモーション

    Parameters
    ----------
    path : str, optional
        パス, by default None
    signature : str, optional
        パス, by default None
    model_name : str, optional
        パス, by default None
    bones : VmdBoneFrames
        ボーンキーフレリスト, by default []
    morphs : VmdMorphFrames
        モーフキーフレリスト, by default []
    morphs : VmdMorphFrames
        モーフキーフレリスト, by default []
    cameras : VmdCameraFrames
        カメラキーフレリスト, by default []
    lights : VmdLightFrames
        照明キーフレリスト, by default []
    shadows : VmdShadowFrames
        セルフ影キーフレリスト, by default []
    show_iks : VmdShowIkFrames
        IKキーフレリスト, by default []
    """

    def __init__(
        self,
        path: str = None,
    ):
        super().__init__(path=path or "")
        self.signature: str = ""
        self.model_name: str = ""
        self.bones: VmdBoneFrames = VmdBoneFrames()
        self.morphs: VmdMorphFrames = VmdMorphFrames()
        self.cameras: VmdCameraFrames = VmdCameraFrames()
        self.lights: VmdLightFrames = VmdLightFrames()
        self.shadows: VmdShadowFrames = VmdShadowFrames()
        self.show_iks: VmdShowIkFrames = VmdShowIkFrames()

    def get_bone_count(self) -> int:
        return int(np.sum([len(bfs) for bfs in self.bones]))

    def get_name(self) -> str:
        return self.model_name

    def init_matrix(self):
        for named_bfs in self.bones:
            for bf in named_bfs:
                bf.init_matrix()
