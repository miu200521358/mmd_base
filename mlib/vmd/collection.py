from math import acos, degrees, pi

import numpy as np

from mlib.base.bezier import evaluate
from mlib.base.collection import (
    BaseHashModel,
    BaseIndexDictModel,
    BaseIndexNameDictInnerModel,
    BaseIndexNameDictModel,
)
from mlib.base.math import MMatrix4x4, MMatrix4x4List, MQuaternion, MVector3D
from mlib.pmx.collection import BoneTree, PmxModel
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

        if index not in self.indices():
            # キーフレがない場合、生成したのを返す（保持はしない）
            prev_index, middle_index, next_index = self.range_indexes(index)
            ik_prev_index, ik_middle_index, ik_next_index = self.range_indexes(
                index,
                indices=sorted([v.index for v in self.data.values() if v.ik_rotation]),
            )
            # prevとnextの範囲内である場合、補間曲線ベースで求め直す
            return self.calc(
                prev_index,
                middle_index,
                next_index,
                ik_prev_index,
                ik_middle_index,
                ik_next_index,
                index,
            )
        return self.data[index]

    def indices(self):
        # 登録対象のキーのみを検出対象とする
        return sorted([v.index for v in self.data.values() if v.register])

    def calc(
        self,
        prev_index: int,
        middle_index: int,
        next_index: int,
        ik_prev_index: int,
        ik_middle_index: int,
        ik_next_index: int,
        index: int,
    ) -> VmdBoneFrame:
        if index in self.data:
            return self.data[index]

        bf = VmdBoneFrame(name=self.name, index=index)

        if prev_index == middle_index:
            # prevと等しい場合、指定INDEX以前がないので、その次のをコピーして返す
            bf.position = self[next_index].position.copy()
            bf.rotation = self[next_index].rotation.copy()
            bf.ik_rotation = (self[ik_next_index].ik_rotation or MQuaternion()).copy()
            bf.interpolations = self[next_index].interpolations.copy()
            return bf
        elif next_index == middle_index:
            # nextと等しい場合は、その前のをコピーして返す
            bf.position = self[prev_index].position.copy()
            bf.rotation = self[prev_index].rotation.copy()
            bf.ik_rotation = (self[ik_prev_index].ik_rotation or MQuaternion()).copy()
            bf.interpolations = self[prev_index].interpolations.copy()
            return bf

        prev_bf = self[prev_index]
        next_bf = self[next_index]

        prev_ik_qq = (
            prev_bf.ik_rotation
            if prev_bf.ik_rotation
            else self[ik_prev_index].ik_rotation
            if self[ik_prev_index].ik_rotation
            else MQuaternion()
        )
        next_ik_qq = (
            next_bf.ik_rotation
            if next_bf.ik_rotation
            else self[ik_next_index].ik_rotation
            if self[ik_next_index].ik_rotation
            else MQuaternion()
        )

        # FK用回転
        _, ry, _ = evaluate(
            next_bf.interpolations.rotation, prev_index, index, next_index
        )
        bf.rotation = MQuaternion.slerp(prev_bf.rotation, next_bf.rotation, ry)
        # IK用回転
        bf.ik_rotation = MQuaternion.slerp(prev_ik_qq, next_ik_qq, ry)

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
        self,
        fnos: list[int],
        bone_trees: list[BoneTree],
        model: PmxModel,
    ) -> dict[str, dict[int, dict[str, VmdBoneFrameTree]]]:
        """
        指定されたキーフレ番号の行列計算結果を返す

        Parameters
        ----------
        fnos : list[int]
            キーフレ番号のリスト
        bone_trees: list[BoneTree]
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
                    poses[n, m] = self.get_position(bone, fno, model)
                    # FK(捩り) > IK(捩り) > 付与親(捩り)
                    qqs[n, m] = self.get_rotation(bone, fno, model, append_ik=True)
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

    def get_position(self, bone: Bone, fno: int, model: PmxModel) -> MVector3D:
        # TODO 付与親
        return (
            self[bone.name][fno].position
            + bone.position
            - (
                MVector3D()
                if bone.index < 0 or bone.parent_index not in model.bones
                else model.bones[bone.parent_index].position
            )
        )

    def get_rotation(
        self, bone: Bone, fno: int, model: PmxModel, append_ik: bool
    ) -> MQuaternion:
        # FK(捩り) > IK(捩り) > 付与親(捩り)
        bf = self[bone.name][fno]
        qq = bf.rotation
        if bf.ik_rotation:
            # IK用回転を持っている場合、追加
            qq *= bf.ik_rotation

        fk_qq = self.get_fix_rotation(bone, qq)

        # IKを加味した回転
        ik_qq = self.get_ik_rotation(bone, fno, fk_qq, model) if append_ik else fk_qq

        # 付与親を加味した回転
        effect_qq = self.get_effect_rotation(bone, fno, ik_qq, model)

        return effect_qq.normalized()

    def get_effect_rotation(
        self,
        bone: Bone,
        fno: int,
        qq: MQuaternion,
        model: PmxModel,
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
                effect_qq = self.get_rotation(effect_bone, fno, model, append_ik=True)
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

    def get_ik_rotation(
        self,
        bone: Bone,
        fno: int,
        qq: MQuaternion,
        model: PmxModel,
    ) -> MQuaternion:
        """
        IKを加味した回転を求める

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
        ik_bones : PmxModel
            IK計算用キーフレ

        Returns
        -------
        MQuaternion
            計算結果
        """
        # 影響ボーン移動辞書
        bone_positions: dict[int, MVector3D] = {}

        for ik_target_bone_idx in bone.ik_target_indices:
            # IKボーン自身の位置
            ik_bone = model.bones[ik_target_bone_idx]

            if ik_target_bone_idx not in model.bones or not ik_bone.ik:
                continue

            ik_matrixes = self.get_matrix_by_indexes(
                [fno], [model.bone_trees[ik_bone.index]], model
            )
            global_result_pos = ik_matrixes[ik_bone.name][fno][ik_bone.name].position

            # IKターゲットボーンツリー
            effector_bone = model.bones[ik_bone.ik.bone_index]
            effector_bone_tree = model.bone_trees[effector_bone.index]

            # IKリンクボーンツリー
            ik_link_bone_trees: dict[int, BoneTree] = {}
            for ik_link in ik_bone.ik.links:
                if ik_link.bone_index not in model.bones:
                    continue
                ik_link_bone_trees[ik_link.bone_index] = model.bone_trees[
                    ik_link.bone_index
                ]

            for _ in range(ik_bone.ik.loop_count):
                for ik_link in ik_bone.ik.links:
                    # ikLink は末端から並んでる
                    for axis in ["z", "x", "y"]:
                        # ZXY の順番でIKを回す
                        if ik_link.bone_index not in model.bones:
                            continue

                        # 現在のIKターゲットボーンのグローバル位置を取得
                        col = len(effector_bone_tree)
                        poses = np.full((1, col), MVector3D())
                        qqs = np.full((1, col), MQuaternion())
                        for m, it_bone in enumerate(effector_bone_tree):
                            # ボーンの親から見た相対位置を求める
                            if it_bone.index not in bone_positions:
                                bone_positions[it_bone.index] = self.get_position(
                                    it_bone, fno, model
                                )
                            poses[0, m] = bone_positions[it_bone.index]
                            # ボーンの回転
                            qqs[0, m] = self.get_rotation(
                                it_bone, fno, model, append_ik=False
                            )
                        matrixes = MMatrix4x4List(1, col)
                        matrixes.translate(poses.tolist())
                        matrixes.rotate(qqs.tolist())
                        effector_result_mats = matrixes.matmul_cols()
                        global_effector_pos = MVector3D(
                            *effector_result_mats.to_positions()[0, -1]
                        )

                        # 処理対象IKボーン
                        link_bone = model.bones[ik_link.bone_index]
                        link_bone_tree = ik_link_bone_trees[link_bone.index]

                        # 処理対象IKボーンのグローバル位置と行列を取得
                        col = len(link_bone_tree)
                        poses = np.full((1, col), MVector3D())
                        qqs = np.full((1, col), MQuaternion())
                        for m, it_bone in enumerate(link_bone_tree):
                            # ボーンの親から見た相対位置を求める
                            if it_bone.index not in bone_positions:
                                bone_positions[it_bone.index] = self.get_position(
                                    it_bone, fno, model
                                )
                            poses[0, m] = bone_positions[it_bone.index]
                            # ボーンの回転
                            qqs[0, m] = self.get_rotation(
                                it_bone, fno, model, append_ik=False
                            )
                        matrixes = MMatrix4x4List(1, col)
                        matrixes.translate(poses.tolist())
                        matrixes.rotate(qqs.tolist())
                        link_result_mats = matrixes.matmul_cols()

                        # 注目ノード（実際に動かすボーン）
                        link_matrix = MMatrix4x4(*link_result_mats.vector[-1])

                        # ワールド座標系から注目ノードの局所座標系への変換
                        link_inverse_matrix = link_matrix.inverse()

                        # 注目ノードを起点とした、エフェクタのローカル位置
                        local_effector_pos = link_inverse_matrix * global_effector_pos
                        # 注目ノードを起点とした、IK目標のローカル位置
                        local_result_pos = link_inverse_matrix * global_result_pos

                        if (
                            local_effector_pos - local_result_pos
                        ).length_squared() < 0.0001:
                            # 位置の差がほとんどない場合、終了
                            return qq

                        #  (1) 基準関節→エフェクタ位置への方向ベクトル
                        norm_effector_pos = local_effector_pos.normalized()
                        #  (2) 基準関節→目標位置への方向ベクトル
                        norm_ik_target_pos = local_result_pos.normalized()

                        # ベクトル (1) を (2) に一致させるための最短回転量（Axis-Angle）
                        # 回転角
                        rotation_dot = norm_effector_pos.dot(norm_ik_target_pos)
                        # 回転角度
                        rotation_radian = acos(max(-1, min(1, rotation_dot)))

                        if abs(rotation_radian) < 0.0001:
                            # ほとんど回らない場合、スルー
                            continue

                        # 回転軸
                        rotation_axis = norm_effector_pos.cross(
                            norm_ik_target_pos
                        ).normalized()
                        # 回転角度(制限角で最大変位量を制限する)
                        rotation_degree = min(
                            degrees(rotation_radian), ik_bone.ik.unit_rotation.degrees.x
                        )

                        # 補正関節回転量
                        correct_qq = MQuaternion.from_axis_angles(
                            rotation_axis, rotation_degree
                        )

                        # 軸制限をかけた回転
                        mat = MMatrix4x4()
                        mat.rotate(qq)
                        match axis:
                            case "x":
                                mat.rotate_x(correct_qq)
                            case "y":
                                mat.rotate_y(correct_qq)
                            case "z":
                                mat.rotate_z(correct_qq)
                        qq = mat.to_quaternion()

                        # リンクボーンの角度を保持
                        self[link_bone.name][fno].ik_rotation = (
                            self[link_bone.name][fno].ik_rotation or MQuaternion()
                        ) * qq

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
