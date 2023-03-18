from math import acos, degrees, pi
from typing import Optional

import numpy as np

from mlib.base.bezier import evaluate
from mlib.base.collection import (
    BaseHashModel,
    BaseIndexDictModel,
    BaseIndexNameDictInnerModel,
    BaseIndexNameDictModel,
)
from mlib.base.math import MMatrix4x4, MMatrix4x4List, MQuaternion, MVector3D
from mlib.pmx.pmx_collection import BoneTree, PmxModel
from mlib.pmx.pmx_part import Bone
from mlib.vmd.vmd_part import (
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
        self.__ik_indices: list[int] = []

    def __getitem__(self, index: int) -> VmdBoneFrame:
        if not self.data:
            # まったくデータがない場合、生成
            return VmdBoneFrame(name=self.name, index=index)

        if index in self.data.keys():
            return self.data[index]

        # キーフレがない場合、生成したのを返す（保持はしない）
        prev_index, middle_index, next_index = self.range_indexes(index)

        # IK用キーはIK回転情報があるキーフレのみ対象とする
        if self.__ik_indices:
            ik_prev_index, ik_middle_index, ik_next_index = self.range_indexes(
                index,
                indices=self.__ik_indices,
            )
        else:
            ik_prev_index = ik_middle_index = ik_next_index = 0

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

    def append(self, value: VmdBoneFrame):
        if value.ik_rotation is not None:
            self.__ik_indices.append(value.index)
        return super().append(value)

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

        if prev_index == middle_index == next_index and ik_prev_index == ik_middle_index == ik_next_index:
            # 全くキーフレがない場合、そのまま返す
            return bf
        if prev_index == middle_index and middle_index != next_index:
            # FKのprevと等しい場合、指定INDEX以前がないので、その次のをコピーして返す
            bf.position = self[next_index].position.copy()
            bf.rotation = self[next_index].rotation.copy()
            bf.ik_rotation = (self[ik_next_index].ik_rotation or MQuaternion()).copy()
            bf.interpolations = self[next_index].interpolations.copy()
            return bf
        elif prev_index != middle_index and next_index == middle_index:
            # FKのnextと等しい場合は、その前のをコピーして返す
            bf.position = self[prev_index].position.copy()
            bf.rotation = self[prev_index].rotation.copy()
            bf.ik_rotation = (self[ik_prev_index].ik_rotation or MQuaternion()).copy()
            bf.interpolations = self[prev_index].interpolations.copy()
            return bf

        prev_bf = self[prev_index] if prev_index in self else VmdBoneFrame(name=self.name, index=prev_index)
        next_bf = self[next_index] if next_index in self else VmdBoneFrame(name=self.name, index=next_index)

        ik_prev_bf = self[ik_prev_index] if ik_prev_index in self else VmdBoneFrame(name=self.name, index=ik_prev_index)
        ik_next_bf = self[ik_next_index] if ik_next_index in self else VmdBoneFrame(name=self.name, index=ik_next_index)

        # 補間結果Yは、FKキーフレ内で計算する
        _, ry, _ = evaluate(next_bf.interpolations.rotation, prev_index, index, next_index)
        # IK用回転
        bf.ik_rotation = MQuaternion.slerp(
            (ik_prev_bf.ik_rotation or MQuaternion()),
            (ik_next_bf.ik_rotation or MQuaternion()),
            ry,
        )
        # FK用回転
        bf.rotation = MQuaternion.slerp(prev_bf.rotation, next_bf.rotation, ry)

        _, xy, _ = evaluate(next_bf.interpolations.translation_x, prev_index, index, next_index)
        bf.position.x = prev_bf.position.x + (next_bf.position.x - prev_bf.position.x) * xy

        _, yy, _ = evaluate(next_bf.interpolations.translation_y, prev_index, index, next_index)
        bf.position.y = prev_bf.position.y + (next_bf.position.y - prev_bf.position.y) * yy

        _, zy, _ = evaluate(next_bf.interpolations.translation_z, prev_index, index, next_index)
        bf.position.z = prev_bf.position.z + (next_bf.position.z - prev_bf.position.z) * zy

        return bf


class VmdBoneFrameTree:
    def __init__(self, global_matrix: np.ndarray, local_matrix: np.ndarray, bone_matrix: np.ndarray, position: np.ndarray) -> None:
        """
        ボーン変形結果

        Parameters
        ----------
        global_matrix : ワールド座標行列
        local_matrix : 親ボーンから見たローカル座標行列
        bone_matrix : ボーン変形行列
        position : ボーン変形後のグローバル位置
        """
        self.global_matrix = MMatrix4x4(global_matrix)
        self.local_matrix = MMatrix4x4(local_matrix)
        self.bone_matrix = MMatrix4x4(bone_matrix)
        self.position = MVector3D(*position)


class VmdBoneFrames(BaseIndexNameDictModel[VmdBoneFrame, VmdBoneNameFrames]):
    """
    ボーンキーフレ辞書
    """

    def __init__(self) -> None:
        super().__init__()

    def create_inner(self, name: str):
        return VmdBoneNameFrames(name=name)

    def get_matrix_by_indexes(self, fnos: list[int], bone_trees: list[BoneTree], model: PmxModel) -> dict[int, dict[str, VmdBoneFrameTree]]:
        """
        指定されたキーフレ番号の行列計算結果を返す

        Parameters
        ----------
        fnos : list[int]
            キーフレ番号のリスト
        bone_trees: list[BoneTree]
            ボーンツリーリスト
        model: PmxModel
            モデルデータ

        Returns
        -------
        行列辞書（キー: fno,ボーン名、値：行列リスト）
        """
        bone_matrixes: dict[int, dict[str, VmdBoneFrameTree]] = {}
        for bone_tree in bone_trees:
            row = len(fnos)
            col = len(bone_tree)
            poses = np.full((row, col), MVector3D())
            qqs = np.full((row, col), MQuaternion())
            for n, fno in enumerate(fnos):
                for m, bone in enumerate(bone_tree.data.values()):
                    # ボーンの親から見た相対位置
                    poses[n, m] = model.bones.get_parent_relative_position(bone.index).vector
                    poses[n, m] += self.get_position(bone, fno, model).vector
                    # FK(捩り) > IK(捩り) > 付与親(捩り)
                    qqs[n, m] = self.get_rotation(bone, fno, model, append_ik=True).to_matrix4x4().vector
            # 親ボーンから見たローカル座標行列
            matrixes = MMatrix4x4List(row, col)
            matrixes.translate(poses.tolist())
            matrixes.rotate(qqs.tolist())
            # グローバル座標行列
            global_mats = matrixes.matmul_cols()
            # グローバル位置
            positions = global_mats.to_positions()
            # ボーン変形行列
            bone_mats = matrixes.vector @ global_mats.inverse().vector

            for n, fno in enumerate(fnos):
                if fno not in bone_matrixes:
                    bone_matrixes[fno] = {}
                for m, bone in enumerate(bone_tree.data.values()):
                    bone_matrixes[fno][bone.name] = VmdBoneFrameTree(
                        global_matrix=global_mats.vector[n, m], local_matrix=matrixes.vector[n, m], bone_matrix=bone_mats[n, m], position=positions[n, m]
                    )

        return bone_matrixes

    def get(self, name: str) -> VmdBoneNameFrames:
        if name not in self.data:
            self.data[name] = self.create_inner(name)

        return self.data[name]

    def get_position(self, bone: Bone, fno: int, model: PmxModel) -> MVector3D:
        """
        該当キーフレにおけるボーンの移動位置

        Parameters
        ----------
        bone : Bone
            計算対象ボーン
        fno : int
            計算対象キーフレ
        model : PmxModel
            計算対象モデル

        Returns
        -------
        MVector3D
            相対位置
        """
        # 自身の位置
        pos = self[bone.name][fno].position

        # 付与親を加味して返す
        return self.get_effect_position(bone, fno, pos, model)

    def get_effect_position(
        self,
        bone: Bone,
        fno: int,
        pos: MVector3D,
        model: PmxModel,
    ) -> MVector3D:
        """
        付与親を加味した移動を求める

        Parameters
        ----------
        bone : Bone
            計算対象ボーン
        fno : int
            計算対象キーフレ
        pos : MVector3D
            計算対象移動
        model : PmxModel
            計算対象モデル

        Returns
        -------
        MVector3D
            計算結果
        """
        if bone.is_external_translation and bone.effect_index in model.bones:
            if bone.effect_factor == 0:
                # 付与率が0の場合、常に0になる
                return MVector3D()
            else:
                # 付与親の回転量を取得する（それが付与持ちなら更に遡る）
                effect_bone = model.bones[bone.effect_index]
                effect_pos = model.bones.get_parent_relative_position(bone.effect_index)
                effect_pos += self.get_position(effect_bone, fno, model)
                pos *= effect_pos

        return pos

    def get_rotation(self, bone: Bone, fno: int, model: PmxModel, append_ik: bool = False) -> MQuaternion:
        """
        該当キーフレにおけるボーンの相対位置

        Parameters
        ----------
        bone : Bone
            計算対象ボーン
        fno : int
            計算対象キーフレ
        model : PmxModel
            計算対象モデル
        append_ik : bool
            IKを計算するか(循環してしまう場合があるので、デフォルトFalse)

        Returns
        -------
        MQuaternion
            該当キーフレにおけるボーンの回転量
        """
        # FK(捩り) > IK(捩り) > 付与親(捩り)
        bf = self[bone.name][fno]
        qq = bf.rotation.copy()
        if bf.ik_rotation is not None:
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
        if bone.is_external_rotation and bone.effect_index in model.bones:
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
                    qq = qq * (effect_qq.multiply_factor(abs(bone.effect_factor))).inverse()
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

        if not bone.ik_link_indices:
            return qq

        # 影響ボーン移動辞書
        bone_positions: dict[int, MVector3D] = {}

        for ik_target_bone_idx in bone.ik_link_indices:
            # IKボーン自身の位置
            ik_bone = model.bones[ik_target_bone_idx]

            if ik_target_bone_idx not in model.bones or not ik_bone.ik:
                continue

            ik_matrixes = self.get_matrix_by_indexes([fno], [model.bone_trees[ik_bone.index]], model)
            global_target_pos = ik_matrixes[fno][ik_bone.name].position

            # IKターゲットボーンツリー
            effector_bone = model.bones[ik_bone.ik.bone_index]
            effector_bone_tree = model.bone_trees[effector_bone.index]

            # IKの角度をターゲットのIK角度に設定する
            ik_bf = self[ik_bone.name][fno]

            effector_bf = (
                self.data[effector_bone.name].data[fno]
                if effector_bone.name in self.data and fno in self.data[effector_bone.name].data
                else VmdBoneFrame(name=effector_bone.name, index=fno)
            )
            effector_bf.ik_target_rotation = ik_bf.rotation.copy()
            self.append(effector_bf)

            # IKリンクボーンツリー
            ik_link_bone_trees: dict[int, BoneTree] = {}
            for ik_link in ik_bone.ik.links:
                if ik_link.bone_index not in model.bones:
                    continue
                ik_link_bone_trees[ik_link.bone_index] = model.bone_trees[ik_link.bone_index]

            for i in range(ik_bone.ik.loop_count):
                is_break = False
                for ik_link in ik_bone.ik.links:
                    # ikLink は末端から並んでる
                    if ik_link.bone_index not in model.bones:
                        continue

                    # 現在のIKターゲットボーンのグローバル位置を取得
                    col = len(effector_bone_tree)
                    poses = np.full((1, col), MVector3D())
                    qqs = np.full((1, col), MQuaternion())
                    for m, it_bone in enumerate(effector_bone_tree):
                        # ボーンの親から見た相対位置を求める
                        if it_bone.index not in bone_positions:
                            bone_positions[it_bone.index] = model.bones.get_parent_relative_position(it_bone.index)
                            bone_positions[it_bone.index] += self.get_position(it_bone, fno, model)
                        poses[0, m] = bone_positions[it_bone.index].vector
                        # ボーンの回転
                        qqs[0, m] = self.get_rotation(it_bone, fno, model, append_ik=False).to_matrix4x4().vector
                    matrixes = MMatrix4x4List(1, col)
                    matrixes.translate(poses.tolist())
                    matrixes.rotate(qqs.tolist())
                    effector_result_mats = matrixes.matmul_cols()
                    global_effector_pos = MVector3D(*effector_result_mats.to_positions()[0, -1])

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
                            bone_positions[it_bone.index] = model.bones.get_parent_relative_position(it_bone.index)
                            bone_positions[it_bone.index] += self.get_position(it_bone, fno, model)
                        poses[0, m] = bone_positions[it_bone.index].vector
                        # ボーンの回転
                        qqs[0, m] = self.get_rotation(it_bone, fno, model, append_ik=False).to_matrix4x4().vector
                    matrixes = MMatrix4x4List(1, col)
                    matrixes.translate(poses.tolist())
                    matrixes.rotate(qqs.tolist())
                    link_target_mats = matrixes.matmul_cols()

                    # 注目ノード（実際に動かすボーン）
                    link_matrix = MMatrix4x4(link_target_mats.vector[0, -1])

                    # ワールド座標系から注目ノードの局所座標系への変換
                    link_inverse_matrix = link_matrix.inverse()

                    # 注目ノードを起点とした、エフェクタのローカル位置
                    local_effector_pos = link_inverse_matrix * global_effector_pos
                    # 注目ノードを起点とした、IK目標のローカル位置
                    local_target_pos = link_inverse_matrix * global_target_pos

                    if (local_effector_pos - local_target_pos).length_squared() < 0.00001:
                        # 位置の差がほとんどない場合、スルー
                        is_break = True
                        break

                    #  (1) 基準関節→エフェクタ位置への方向ベクトル
                    norm_effector_pos = local_effector_pos.normalized()
                    #  (2) 基準関節→目標位置への方向ベクトル
                    norm_target_pos = local_target_pos.normalized()

                    # ベクトル (1) を (2) に一致させるための最短回転量（Axis-Angle）
                    # 回転角
                    rotation_dot = norm_effector_pos.dot(norm_target_pos)
                    # 回転角度
                    rotation_radian = acos(max(-1, min(1, rotation_dot)))

                    if abs(1 - rotation_dot) < 0.00001:
                        # ほとんど回らない場合、スルー
                        is_break = True
                        break

                    # 回転軸
                    rotation_axis = norm_effector_pos.cross(norm_target_pos).normalized()
                    # 回転角度
                    rotation_degree = degrees(rotation_radian)

                    # 制限角で最大変位量を制限する
                    rotation_degree = min(rotation_degree, ik_bone.ik.unit_rotation.degrees.x)

                    # 補正関節回転量
                    correct_qq = MQuaternion.from_axis_angles(rotation_axis, rotation_degree)

                    # リンクボーンの角度を保持
                    link_bf = self[link_bone.name][fno]

                    # 軸制限をかけた回転
                    mat = MMatrix4x4()
                    mat.rotate(link_bf.ik_rotation or MQuaternion())

                    if link_bone.has_local_coordinate:
                        # ローカル軸を向く
                        mat.rotate(
                            MQuaternion.from_axes(
                                link_bone.local_x_vector,
                                link_bone.local_y_vector,
                                link_bone.correct_local_z_vector,
                            )
                        )

                    # 補正角度を軸に沿ったオイラー角度に分解する
                    euler_degrees = correct_qq.separate_euler_degrees()
                    if ik_link.angle_limit:
                        # 角度制限が入ってる場合
                        euler_degrees.x = max(
                            min(
                                euler_degrees.x,
                                ik_link.max_angle_limit.degrees.x,
                            ),
                            ik_link.min_angle_limit.degrees.x,
                        )
                        euler_degrees.y = max(
                            min(
                                euler_degrees.y,
                                ik_link.max_angle_limit.degrees.y,
                            ),
                            ik_link.min_angle_limit.degrees.y,
                        )
                        euler_degrees.z = max(
                            min(
                                euler_degrees.z,
                                ik_link.max_angle_limit.degrees.z,
                            ),
                            ik_link.min_angle_limit.degrees.z,
                        )
                        correct_qq = MQuaternion.from_euler_degrees(euler_degrees)
                    mat.rotate(correct_qq)

                    link_bf.ik_rotation = mat.to_quaternion()
                    self.append(link_bf)

                if is_break:
                    break

        # IKの計算結果の回転を加味して返す
        bf = self[bone.name][fno]
        return bf.rotation * (bf.ik_rotation or MQuaternion())

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
        if bone.has_fixed_axis:
            qq_axis = MVector3D(qq.x, qq.y, qq.z)
            theta = acos(max(-1, min(1, bone.fixed_axis.dot(qq_axis))))

            fixed_qq_axis: MVector3D = bone.fixed_axis * qq_axis.length() * (1 if theta < pi / 2 else -1)
            return MQuaternion(qq.scalar, fixed_qq_axis.x, fixed_qq_axis.y, fixed_qq_axis.z).normalized()

        return qq


class VmdMorphNameFrames(BaseIndexNameDictInnerModel[VmdMorphFrame]):
    """
    モーフ名別キーフレ辞書
    """

    def __init__(self, name: str):
        super().__init__(name=name)
        self.__ik_indices: list[int] = []

    def __getitem__(self, index: int) -> VmdMorphFrame:
        if not self.data:
            # まったくデータがない場合、生成
            return VmdMorphFrame(name=self.name, index=index)

        if index in self.data.keys():
            return self.data[index]

        # キーフレがない場合、生成したのを返す（保持はしない）
        prev_index, middle_index, next_index = self.range_indexes(index)

        # prevとnextの範囲内である場合、補間曲線ベースで求め直す
        return self.calc(
            prev_index,
            middle_index,
            next_index,
            index,
        )

    def append(self, value: VmdMorphFrame):
        return super().append(value)

    def calc(
        self,
        prev_index: int,
        middle_index: int,
        next_index: int,
        index: int,
    ) -> VmdMorphFrame:
        if index in self.data:
            return self.data[index]

        mf = VmdMorphFrame(name=self.name, index=index)

        if prev_index == middle_index == next_index:
            # 全くキーフレがない場合、そのまま返す
            return mf
        if prev_index == middle_index and middle_index != next_index:
            # FKのprevと等しい場合、指定INDEX以前がないので、その次のをコピーして返す
            mf.ratio = float(self[next_index].ratio)
            return mf
        elif prev_index != middle_index and next_index == middle_index:
            # FKのnextと等しい場合は、その前のをコピーして返す
            mf.ratio = float(self[prev_index].ratio)
            return mf

        prev_mf = self[prev_index]
        next_mf = self[next_index]

        # モーフは補間なし
        ry = (next_index - index) / (next_index - prev_index)
        mf.ratio = prev_mf.ratio + (next_mf.ratio - prev_mf.ratio) * ry

        return mf


class VmdMorphFrames(BaseIndexNameDictModel[VmdMorphFrame, VmdMorphNameFrames]):
    """
    モーフキーフレ辞書
    """

    def __init__(self) -> None:
        super().__init__()

    def create_inner(self, name: str):
        return VmdMorphNameFrames(name=name)


class VmdCameraFrames(BaseIndexDictModel[VmdCameraFrame]):
    """
    カメラキーフレリスト
    """

    def __init__(self) -> None:
        super().__init__()


class VmdLightFrames(BaseIndexDictModel[VmdLightFrame]):
    """
    照明キーフレリスト
    """

    def __init__(self) -> None:
        super().__init__()


class VmdShadowFrames(BaseIndexDictModel[VmdShadowFrame]):
    """
    照明キーフレリスト
    """

    def __init__(self) -> None:
        super().__init__()


class VmdShowIkFrames(BaseIndexDictModel[VmdShowIkFrame]):
    """
    IKキーフレリスト
    """

    def __init__(self) -> None:
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
        path: Optional[str] = None,
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

    @property
    def bone_count(self) -> int:
        return int(np.sum([len(bfs) for bfs in self.bones]))

    @property
    def name(self) -> str:
        return self.model_name
