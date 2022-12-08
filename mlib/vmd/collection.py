from math import acos, pi

import numpy as np

from mlib.base.bezier import evaluate
from mlib.base.collection import (
    BaseHashModel,
    BaseIndexDictModel,
    BaseIndexNameDictInnerModel,
    BaseIndexNameDictModel,
)
from mlib.base.math import MMatrix4x4, MMatrix4x4List, MQuaternion, MVector3D
from mlib.pmx.collection import BoneTree, BoneTrees, PmxModel
from mlib.pmx.part import Bone, BoneFlg
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


class VmdBoneFrameTree:
    def __init__(self, matrix: np.ndarray, position: np.ndarray) -> None:
        self.matrix = MMatrix4x4(matrix)
        self.position = MVector3D(*position)


class VmdBoneFrames(BaseIndexNameDictModel[VmdBoneFrame, VmdBoneNameFrames]):
    """
    ボーンキーフレ辞書
    """

    def __init__(self):
        super().__init__()

    def create_inner(self, name: str):
        return VmdBoneNameFrames(name=name)

    def get_matrix_by_indexes(
        self, fnos: list[int], bone_trees: BoneTrees, model: PmxModel
    ) -> dict[str, dict[int, dict[str, VmdBoneFrameTree]]]:
        """
        指定されたキーフレ番号の行列計算結果を返す

        Parameters
        ----------
        fnos : list[int]
            キーフレ番号のリスト
        bone_trees: BoneTrees
            ボーンツリーリスト

        Returns
        -------
        行列辞書（キー：ボーンツリーリストの最後のボーン名、値：行列リスト）
        """
        bone_matrixes: dict[str, dict[int, dict[str, VmdBoneFrameTree]]] = {}
        for bone_tree in bone_trees:
            row = len(fnos)
            col = len(bone_tree)
            poses = np.full((row, col), MVector3D())
            qqs = np.full((row, col), MQuaternion())
            for n, fno in enumerate(fnos):
                for m, bone in enumerate(bone_tree):
                    # ボーンの親から見た相対位置を求める
                    poses[n, m] = self.get_position(bone, fno, bone_tree, m)
                    # FK(捩り) > IK(捩り) > 付与親(捩り)
                    qqs[n, m] = self.get_rotation(bone, fno, model)
            matrixes = MMatrix4x4List(row, col)
            matrixes.translate(poses.tolist())
            matrixes.rotate(qqs.tolist())
            result_mats = matrixes.matmul_cols()
            positions = result_mats.to_positions()
            bone_matrixes[bone_tree.last_name()] = {}

            for n, fno in enumerate(fnos):
                bone_matrixes[bone_tree.last_name()][fno] = {}
                for m, bone in enumerate(bone_tree):
                    bone_matrixes[bone_tree.last_name()][fno][
                        bone.name
                    ] = VmdBoneFrameTree(
                        matrix=result_mats.vector[n, m], position=positions[n, m]
                    )

        return bone_matrixes

    def get(self, name: str) -> VmdBoneNameFrames:
        if name not in self.data:
            self.data[name] = self.create_inner(name)

        return self.data[name]

    def get_position(
        self, bone: Bone, fno: int, bone_tree: BoneTree, idx: int
    ) -> MVector3D:
        # TODO 付与親
        return (
            self[bone.name][fno].position
            + bone.position
            - (MVector3D() if idx == 0 else bone_tree[idx - 1].position)
        )

    def get_rotation(self, bone: Bone, fno: int, model: PmxModel) -> MQuaternion:
        # FK(捩り) > IK(捩り) > 付与親(捩り)
        fk_qq = self.get_fix_rotation(bone, self[bone.name][fno].rotation)

        effect_qq = self.get_effect_rotation(bone, fno, fk_qq, model)

        return effect_qq.normalized()

    def get_effect_rotation(
        self, bone: Bone, fno: int, qq: MQuaternion, model: PmxModel
    ) -> MQuaternion:
        """
        付与親を加味した回転を求める

        Parameters
        ----------
        bone : Bone
            計算対象ボーン
        fno : int
            計算対象キーフレ
        qq : MQuaternion
            計算対象クォータニオン
        model : PmxModel
            計算対象モデル

        Returns
        -------
        MQuaternion
            計算結果
        """
        if (
            BoneFlg.IS_EXTERNAL_ROTATION in bone.bone_flg
            and bone.effect_index in model.bones
        ):
            if bone.effect_factor == 0:
                # 付与率が0の場合、常に0になる
                return MQuaternion()
            else:
                # 付与親の回転量を取得する（それが付与持ちなら更に遡る）
                effect_bone = model.bones[bone.effect_index]
                effect_qq = self.get_rotation(effect_bone, fno, model)
                if bone.effect_factor > 0:
                    # 正の付与親
                    qq = qq * effect_qq.multiply_factor(bone.effect_factor)
                else:
                    # 負の付与親の場合、逆回転
                    qq = (
                        qq
                        * (effect_qq.multiply_factor(abs(bone.effect_factor))).inverse()
                    )
        return qq

    def get_fix_rotation(self, bone: Bone, qq: MQuaternion) -> MQuaternion:
        """
        軸制限回転を求める

        Parameters
        ----------
        bone : Bone
            対象ボーン
        qq : MQuaternion
            計算対象回転

        Returns
        -------
        MQuaternion
            軸制限された回転
        """
        if BoneFlg.HAS_FIXED_AXIS in bone.bone_flg:
            qq_axis = MVector3D(qq.x, qq.y, qq.z)
            theta = acos(max(-1, min(1, bone.fixed_axis.dot(qq_axis))))

            fixed_qq_axis: MVector3D = (
                bone.fixed_axis * qq_axis.length() * (1 if theta < pi / 2 else -1)
            )
            return MQuaternion(
                qq.scalar, fixed_qq_axis.x, fixed_qq_axis.y, fixed_qq_axis.z
            ).normalized()

        return qq


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
