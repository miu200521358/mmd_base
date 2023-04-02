from functools import lru_cache
import logging
from bisect import bisect_left
from math import acos, degrees, pi
from typing import Optional

import numpy as np

from mlib.base.collection import BaseHashModel, BaseIndexNameDictModel, BaseIndexNameDictWrapperModel
from mlib.base.logger import MLogger
from mlib.base.math import MMatrix4x4, MMatrix4x4List, MQuaternion, MVector3D
from mlib.pmx.pmx_collection import BoneTree, PmxModel
from mlib.pmx.pmx_part import Bone, BoneMorphOffset, GroupMorphOffset, MorphType, VertexMorphOffset
from mlib.vmd.vmd_part import VmdBoneFrame, VmdCameraFrame, VmdLightFrame, VmdMorphFrame, VmdShadowFrame, VmdShowIkFrame

logger = MLogger(__name__, logging.DEBUG)


class VmdBoneNameFrames(BaseIndexNameDictModel[VmdBoneFrame]):
    """
    ボーン名別キーフレ辞書
    """

    __slots__ = ["data", "name", "cache", "_names", "_indexes", "_iter_index", "_ik_indexes"]

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self.__ik_indexes: list[int] = []

    def __getitem__(self, key: int | str) -> VmdBoneFrame:
        if isinstance(key, str):
            return VmdBoneFrame(name=key, index=0)

        if key in self.data:
            return self.get_by_index(key)

        # キーフレがない場合、生成したのを返す（保持はしない）
        prev_index, middle_index, next_index = self.range_indexes(key)

        # prevとnextの範囲内である場合、補間曲線ベースで求め直す
        return self.calc(
            prev_index,
            middle_index,
            next_index,
        )

    def append(self, value: VmdBoneFrame, is_sort: bool = True):
        if value.ik_rotation is not None and value.index not in self.__ik_indexes:
            self.__ik_indexes.append(value.index)
            self.__ik_indexes.sort()
        super().append(value, is_sort)

    def calc(self, prev_index: int, index: int, next_index: int) -> VmdBoneFrame:
        if index in self.data:
            return self.data[index]

        if index in self.cache:
            bf = self.cache[index]
        else:
            bf = VmdBoneFrame(name=self.name, index=index)
            self.cache[index] = bf

        if prev_index == next_index:
            if next_index == index:
                # 全くキーフレがない場合、そのまま返す
                return bf

            # FKのprevと等しい場合、指定INDEX以前がないので、その次のをコピーして返す
            bf.position = self.data[next_index].position.copy()
            bf.rotation = self.data[next_index].rotation.copy()
            return bf

        prev_bf = self.data[prev_index] if prev_index in self else VmdBoneFrame(name=self.name, index=prev_index)
        next_bf = self.data[next_index] if next_index in self else VmdBoneFrame(name=self.name, index=next_index)

        slice_idx = bisect_left(self.__ik_indexes, index)
        prev_ik_indexes = self.__ik_indexes[:slice_idx]
        next_ik_indexes = self.__ik_indexes[slice_idx:]

        prev_ik_index = prev_ik_indexes[-1] if prev_ik_indexes else prev_index
        prev_ik_rotation = self.data[prev_ik_index].ik_rotation or MQuaternion()

        next_ik_index = next_ik_indexes[0] if next_ik_indexes else next_index
        next_ik_rotation = self.data[next_ik_index].ik_rotation or prev_ik_rotation

        # 補間結果Yは、FKキーフレ内で計算する
        ry, xy, yy, zy = next_bf.interpolations.evaluate(prev_index, index, next_index)

        # IK用回転
        bf.ik_rotation = MQuaternion.slerp(prev_ik_rotation, next_ik_rotation, ry)

        # FK用回転
        bf.rotation = MQuaternion.slerp(prev_bf.rotation, next_bf.rotation, ry)

        # 移動
        bf.position = MVector3D.calc_by_ratio(prev_bf.position, next_bf.position, xy, yy, zy)

        return bf


class VmdBoneFrameTree:
    def __init__(self, global_matrix: np.ndarray, local_matrix: np.ndarray, position: np.ndarray) -> None:
        """
        ボーン変形結果

        Parameters
        ----------
        global_matrix : ワールド座標行列
        local_matrix : 親ボーンから見たローカル座標行列
        position : ボーン変形後のグローバル位置
        """
        self.global_matrix = MMatrix4x4(*global_matrix.flatten())
        self.local_matrix = MMatrix4x4(*local_matrix.flatten())
        self.position = MVector3D(*position)


class VmdBoneFrames(BaseIndexNameDictWrapperModel[VmdBoneNameFrames]):
    """
    ボーンキーフレ辞書
    """

    __slots__ = ["data", "cache", "cache_ratios", "cache_qqs", "_names", "_iter_index"]

    def __init__(self) -> None:
        super().__init__()
        self.cache_ratios: dict[tuple[int, str, str], MVector3D] = {}
        self.cache_qqs: dict[tuple[int, str, str], MQuaternion] = {}

    def create(self, key: str) -> VmdBoneNameFrames:
        return VmdBoneNameFrames(name=key)

    def clear(self) -> None:
        self.cache_ratios = {}
        self.cache_qqs = {}

    @property
    def max_fno(self) -> int:
        return max([max(self[bname].indexes + [0]) for bname in self.names] + [0])

    def get_matrix_by_indexes(
        self,
        fnos: list[int],
        bone_trees: dict[str, BoneTree],
        model: PmxModel,
        append_ik: bool = True,
    ) -> dict[int, dict[str, VmdBoneFrameTree]]:
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

        # IK回転を事前に求めておく
        for fno in fnos:
            self.calc_ik_rotations(fno, model, [bone_tree.last_name for bone_tree in bone_trees.values()])

        bone_matrixes: dict[int, dict[str, VmdBoneFrameTree]] = {}
        for bone_tree in bone_trees.values():
            row = len(fnos)
            col = len(bone_tree) + 1
            poses = np.full((row, col, 3), np.zeros(3))
            qqs = np.full((row, col, 4, 4), np.eye(4))
            for n, fno in enumerate(fnos):
                for m, bone in enumerate(bone_tree):
                    # ボーンの親から見た相対位置
                    poses[n, m] = bone.parent_relative_position.vector

                    if (fno, model.hexdigest, bone.name) in self.cache_ratios:
                        poses[n, m] += self.cache_ratios[(fno, model.hexdigest, bone.name)].vector
                    else:
                        pos = self.get_position(bone, fno, model)
                        self.cache_ratios[(fno, model.hexdigest, bone.name)] = pos
                        poses[n, m] += pos.vector

                    # FK(捩り) > IK(捩り) > 付与親(捩り)
                    if (fno, model.hexdigest, bone.name) in self.cache_qqs:
                        qqs[n, m] = self.cache_qqs[(fno, model.hexdigest, bone.name)].to_matrix4x4().vector
                    else:
                        qq = self.get_rotation(bone, fno, model, append_ik=append_ik)
                        self.cache_qqs[(fno, model.hexdigest, bone.name)] = qq
                        qqs[n, m] = qq.to_matrix4x4().vector

                # 末端ボーン表示先の位置を計算
                poses[n, -1] = bone_tree[-1].tail_relative_position.vector
                qqs[n, -1] = np.eye(4)
            # 親ボーンから見たローカル座標行列
            matrixes = MMatrix4x4List(row, col)
            matrixes.translate(poses.tolist())
            matrixes.rotate(qqs.tolist())
            # グローバル座標行列
            global_mats = matrixes.matmul_cols()
            # グローバル位置
            positions = global_mats.to_positions()

            for i, fno in enumerate(fnos):
                if fno not in bone_matrixes:
                    bone_matrixes[fno] = {}
                for j, bone in enumerate(bone_tree):
                    bone_matrixes[fno][bone.name] = VmdBoneFrameTree(
                        global_matrix=global_mats.vector[i, j],
                        local_matrix=matrixes.vector[i, j],
                        position=positions[i, j],
                    )
                bone_matrixes[fno]["-1"] = VmdBoneFrameTree(
                    global_matrix=global_mats.vector[i, -1],
                    local_matrix=matrixes.vector[i, -1],
                    position=positions[i, -1],
                )

        return bone_matrixes

    def animate_bone_matrixes(self, fno: int, model: PmxModel, bone_names: Optional[list[str]] = None) -> tuple[np.ndarray, np.ndarray]:
        row = 1
        col = len(model.bones)
        poses = np.full((row, col, 3), np.zeros(3))
        qqs = np.full((row, col, 4, 4), np.eye(4))
        bone_indexes: list[int] = []

        if not bone_names:
            bone_names = model.bones.tail_bone_names

        # IK回転を事前に求めておく
        self.calc_ik_rotations(fno, model)

        for bone_name in bone_names:
            for bone in model.bone_trees[bone_name]:
                if bone.index not in bone_indexes:
                    # モーションによる移動量
                    if (fno, model.hexdigest, bone.name) in self.cache_ratios:
                        poses[0, bone.index] = self.cache_ratios[(fno, model.hexdigest, bone.name)].gl.vector
                    else:
                        pos = self.get_position(bone, fno, model)
                        poses[0, bone.index] = pos.gl.vector
                        self.cache_ratios[(fno, model.hexdigest, bone.name)] = pos

                    # FK(捩り) > IK(捩り) > 付与親(捩り)
                    if (fno, model.hexdigest, bone.name) in self.cache_qqs:
                        qqs[0, bone.index] = self.cache_qqs[(fno, model.hexdigest, bone.name)].gl.to_matrix4x4().vector
                    else:
                        qq = self.get_rotation(bone, fno, model, append_ik=True)
                        self.cache_qqs[(fno, model.hexdigest, bone.name)] = qq
                        qqs[0, bone.index] = qq.gl.to_matrix4x4().vector
                    # 計算済みボーンとして登録
                    bone_indexes.append(bone.index)

        return poses, qqs

    def calc_ik_rotations(self, fno: int, model: PmxModel, bone_names: Optional[list[str]] = None):
        # IK関係の末端ボーン名
        ik_last_bone_names: set[str] = {model.bones[0].name}
        if bone_names:
            target_last_bone_names = {model.bone_trees[bname].last_name for bname in bone_names}
        else:
            target_last_bone_names = set(model.bones.names)
        for bone in model.bones:
            if bone.is_ik and bone.ik:
                # IKリンクボーン・ターゲットボーンのボーンツリーをすべてチェック対象とする
                ik_last_bone_names |= {model.bone_trees[bone.index].last_name}
                ik_last_bone_names |= {model.bone_trees[bone.ik.bone_index].last_name}
                for link_bone in bone.ik.links:
                    ik_last_bone_names |= {model.bone_trees[link_bone.bone_index].last_name}
        ik_last_bone_names &= target_last_bone_names
        if not ik_last_bone_names:
            # IK計算対象がない場合はそのまま終了
            return
        # モーション内のキーフレリストから前の変化キーフレと次の変化キーフレを抽出する
        prev_frame_indexes: set[int] = {0}
        next_frame_indexes: set[int] = {self.max_fno}
        for ik_last_bone_name in [bone.name for bone in model.bones if bone.name in ik_last_bone_names]:
            if ik_last_bone_name in self:
                prev_frame_indexes |= {i for i in self[ik_last_bone_name].indexes if fno > i}
                next_frame_indexes |= {i for i in self[ik_last_bone_name].indexes if fno < i}
                prev_fno = max(list(prev_frame_indexes))
                next_fno = min(list(next_frame_indexes))

                is_calc_prev = False
                is_calc_next = False
                for bone in model.bone_trees[ik_last_bone_name]:
                    if bone.ik_link_indexes or bone.ik_target_indexes:
                        if prev_fno not in self[bone.name] or not self[bone.name][prev_fno].ik_rotation:
                            is_calc_prev = True
                        if next_fno not in self[bone.name] or not self[bone.name][next_fno].ik_rotation:
                            is_calc_next = True
                if is_calc_prev and (prev_fno, model.hexdigest, ik_last_bone_name) not in self.cache_qqs:
                    self.get_rotation(model.bones[ik_last_bone_name], prev_fno, model, append_ik=True)
                if is_calc_next and (next_fno, model.hexdigest, ik_last_bone_name) not in self.cache_qqs:
                    self.get_rotation(model.bones[ik_last_bone_name], next_fno, model, append_ik=True)

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
        pos = self[bone.name][fno].position.copy()

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
        if not (bone.is_external_translation and bone.effect_index in model.bones):
            return pos

        if bone.effect_factor == 0:
            # 付与率が0の場合、常に0になる
            return MVector3D()

        # 付与親の回転量を取得する（それが付与持ちなら更に遡る）
        effect_bone = model.bones[bone.effect_index]
        # effect_pos = model.bones.get_parent_relative_position(bone.effect_index)
        effect_pos = self.get_position(effect_bone, fno, model)
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

        return effect_qq

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
        if not (bone.is_external_rotation and bone.effect_index in model.bones):
            return qq

        if bone.effect_factor == 0:
            # 付与率が0の場合、常に0になる
            return MQuaternion()

        # 付与親の回転量を取得する（それが付与持ちなら更に遡る）
        effect_bone = model.bones[bone.effect_index]
        effect_qq = self.get_rotation(effect_bone, fno, model, append_ik=True)
        if bone.effect_factor > 0:
            # 正の付与親
            qq *= effect_qq.multiply_factor(bone.effect_factor)
        else:
            # 負の付与親の場合、逆回転
            qq *= (effect_qq.multiply_factor(abs(bone.effect_factor))).inverse()

        return qq.normalized()

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

        if not bone.ik_link_indexes:
            return qq

        # 影響ボーン移動辞書
        bone_positions: dict[int, MVector3D] = {}

        for ik_target_bone_idx in bone.ik_link_indexes:
            # IKボーン自身の位置
            ik_bone = model.bones[ik_target_bone_idx]

            if ik_target_bone_idx not in model.bones or not ik_bone.ik:
                continue

            ik_matrixes = self.get_matrix_by_indexes([fno], model.bone_trees.filter(ik_bone.name), model, append_ik=False)
            global_target_pos = ik_matrixes[fno][ik_bone.name].position

            # IKターゲットボーンツリー
            effector_bone = model.bones[ik_bone.ik.bone_index]
            effector_bone_tree = model.bone_trees[effector_bone.name]

            # IKリンクボーンツリー
            ik_link_bone_trees: dict[int, BoneTree] = {ik_bone.index: model.bone_trees[ik_bone.name]}
            for ik_link in ik_bone.ik.links:
                if ik_link.bone_index not in model.bones:
                    continue
                ik_link_bone_trees[ik_link.bone_index] = model.bone_trees[model.bones[ik_link.bone_index].name]

            is_break = False
            for loop in range(ik_bone.ik.loop_count):
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

                    # リンクボーンの角度を保持
                    link_bf = self[link_bone.name][fno]
                    if "左" in link_bf.name:
                        logger.debug(
                            f"- ik_rotation: name[{link_bf.name}], index[{link_bf.index}], loop[{loop}] "
                            + f"rot[{link_bf.ik_rotation.to_euler_degrees_mmd() if link_bf.ik_rotation else '-'}]"
                        )

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
                    link_matrix = MMatrix4x4(*link_target_mats.vector[0, -1].flatten())

                    # ワールド座標系から注目ノードの局所座標系への変換
                    link_inverse_matrix = link_matrix.inverse()

                    # 注目ノードを起点とした、エフェクタのローカル位置
                    local_effector_pos = link_inverse_matrix * global_effector_pos
                    # 注目ノードを起点とした、IK目標のローカル位置
                    local_target_pos = link_inverse_matrix * global_target_pos

                    logger.debug(
                        f"- ik_rotation: name[{link_bf.name}], index[{link_bf.index}], loop[{loop}], "
                        + f"global_target_pos: {global_target_pos}, global_effector_pos: {global_effector_pos}"
                    )

                    if (local_effector_pos - local_target_pos).length_squared() < 1e-5:
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

                    # 回転軸
                    rotation_axis = norm_effector_pos.cross(norm_target_pos).normalized()
                    # 回転角度
                    rotation_degree = degrees(rotation_radian)

                    # if abs(1 - rotation_dot) < 1e-6:
                    #     # ほとんど回らない場合、スルー
                    #     continue

                    # 制限角で最大変位量を制限する
                    if loop > 0:
                        rotation_degree = min(rotation_degree, ik_bone.ik.unit_rotation.degrees.x)

                    # 補正関節回転量
                    correct_qq = MQuaternion.from_axis_angles(rotation_axis, rotation_degree)
                    ik_qq = (link_bf.ik_rotation or MQuaternion()) * correct_qq

                    if ik_link.angle_limit:
                        # 角度制限が入ってる場合、オイラー角度に分解する
                        euler_degrees = ik_qq.separate_euler_degrees()

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
                        ik_qq = MQuaternion.from_euler_degrees(euler_degrees)

                    link_bf.ik_rotation = ik_qq
                    if "左" in link_bf.name:
                        logger.debug(
                            f"-- ik_rotation: name[{link_bf.name}], index[{link_bf.index}], loop[{loop}], "
                            + f"rot[{link_bf.ik_rotation.to_euler_degrees_mmd() if link_bf.ik_rotation else '-'}]"
                        )
                    self[link_bf.name].append(link_bf)

                if is_break:
                    break
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


@lru_cache(maxsize=None)
def calc_morph_ratio(prev: float, next: float, ratio: float) -> float:
    return prev + (next - prev) * ratio


class VmdMorphNameFrames(BaseIndexNameDictModel[VmdMorphFrame]):
    """
    モーフ名別キーフレ辞書
    """

    def __getitem__(self, key: int | str) -> VmdMorphFrame:
        if isinstance(key, str):
            return VmdMorphFrame(name=key, index=0)

        if key in self.data:
            return self.get_by_index(key)

        # キーフレがない場合、生成したのを返す（保持はしない）
        prev_index, middle_index, next_index = self.range_indexes(key)

        # prevとnextの範囲内である場合、補間曲線ベースで求め直す
        return self.calc(
            prev_index,
            middle_index,
            next_index,
        )

    def calc(self, prev_index: int, index: int, next_index: int) -> VmdMorphFrame:
        if index in self.data:
            return self.data[index]

        if index in self.cache:
            mf = self.cache[index]
        else:
            mf = VmdMorphFrame(name=self.name, index=index)
            self.cache[index] = mf

        if prev_index == next_index:
            if next_index == index:
                # 全くキーフレがない場合、そのまま返す
                return mf

            # FKのprevと等しい場合、指定INDEX以前がないので、その次のをコピーして返す
            mf.ratio = self.data[next_index].ratio
            return mf

        prev_mf = self.data[prev_index] if prev_index in self else VmdMorphFrame(name=self.name, index=prev_index)
        next_mf = self.data[next_index] if next_index in self else VmdMorphFrame(name=self.name, index=next_index)

        # モーフは補間なし
        ry = (index - prev_index) / (next_index - prev_index)
        mf.ratio = calc_morph_ratio(prev_mf.ratio, next_mf.ratio, ry)

        return mf


class VmdMorphFrames(BaseIndexNameDictWrapperModel[VmdMorphNameFrames]):
    """
    モーフキーフレ辞書
    """

    def __init__(self) -> None:
        super().__init__()

    def create(self, key: str) -> VmdMorphNameFrames:
        return VmdMorphNameFrames(name=key)

    @property
    def max_fno(self) -> int:
        return max([max(self[fname].indexes + [0]) for fname in self.names] + [0])

    def animate_vertex_morphs(self, fno: int, model: PmxModel) -> np.ndarray:
        row = len(model.vertices)
        poses = np.full((row, 3), np.zeros(3))

        for morph in model.morphs.filter_by_type(MorphType.VERTEX):
            mf = self[morph.name][fno]
            if not mf.ratio:
                continue

            # モーションによる頂点モーフ変動量
            for offset in morph.offsets:
                if type(offset) is VertexMorphOffset and offset.vertex_index < row:
                    ratio_pos: MVector3D = offset.position_offset * mf.ratio
                    poses[offset.vertex_index] += ratio_pos.gl.vector

        return np.array(poses)

    def animate_bone_morphs(self, fno: int, model: PmxModel) -> tuple[list[str], VmdBoneFrames]:
        bone_morph_names: set[str] = set([])
        bone_morphs = VmdBoneFrames()

        for morph in model.morphs.filter_by_type(MorphType.BONE):
            mf = self[morph.name][fno]
            if not mf.ratio:
                continue

            # モーションによるボーンモーフ変動量
            for offset in morph.offsets:
                if type(offset) is BoneMorphOffset and offset.bone_index in model.bones:
                    bone = model.bones[offset.bone_index]
                    bone_morph_names |= {bone.name}

                    bf = VmdBoneFrame(name=bone.name, index=fno)
                    bf.position += offset.position * mf.ratio
                    bf.rotation *= MQuaternion.from_euler_degrees(offset.rotation.degrees * mf.ratio)
                    bone_morphs[bone.name].append(bf)

        return list(bone_morph_names), bone_morphs

    def animate_group_morphs(self, fno: int, model: PmxModel) -> tuple[np.ndarray, list[str], VmdBoneFrames]:
        group_vertex_poses = np.full((len(model.vertices), 3), np.zeros(3))
        group_bone_morphs = VmdBoneFrames()
        group_bone_morph_names: set[str] = set([])

        for morph in model.morphs.filter_by_type(MorphType.GROUP):
            mf = self[morph.name][fno]
            if not mf.ratio:
                continue

            # モーションによるボーンモーフ変動量
            for group_offset in morph.offsets:
                if type(group_offset) is GroupMorphOffset and group_offset.morph_index in model.morphs:
                    part_morph = model.morphs[group_offset.morph_index]
                    mf_factor = mf.ratio * group_offset.morph_factor
                    if not mf_factor:
                        continue

                    for offset in part_morph.offsets:
                        if type(offset) is VertexMorphOffset and offset.vertex_index < group_vertex_poses.shape[0]:
                            ratio_pos: MVector3D = offset.position_offset * mf_factor
                            group_vertex_poses[offset.vertex_index] += ratio_pos.gl.vector
                        elif type(offset) is BoneMorphOffset and offset.bone_index in model.bones:
                            bone = model.bones[offset.bone_index]
                            group_bone_morph_names |= {bone.name}

                            bf = VmdBoneFrame(name=bone.name, index=fno)
                            bf.position += offset.position * mf_factor
                            bf.rotation *= MQuaternion.from_euler_degrees(offset.rotation.degrees * mf_factor)
                            group_bone_morphs[bone.name].append(bf)

        return group_vertex_poses, list(group_bone_morph_names), group_bone_morphs


class VmdCameraFrames(BaseIndexNameDictModel[VmdCameraFrame]):
    """
    カメラキーフレリスト
    """

    def __init__(self) -> None:
        super().__init__()


class VmdLightFrames(BaseIndexNameDictModel[VmdLightFrame]):
    """
    照明キーフレリスト
    """

    def __init__(self) -> None:
        super().__init__()


class VmdShadowFrames(BaseIndexNameDictModel[VmdShadowFrame]):
    """
    照明キーフレリスト
    """

    def __init__(self) -> None:
        super().__init__()


class VmdShowIkFrames(BaseIndexNameDictModel[VmdShowIkFrame]):
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
    def max_fno(self) -> int:
        return max(self.bones.max_fno, self.morphs.max_fno)

    @property
    def name(self) -> str:
        return self.model_name

    def animate(self, fno: int, model: PmxModel) -> tuple[np.ndarray, np.ndarray]:
        # ボーン操作
        bone_poses, bone_qqs = self.bones.animate_bone_matrixes(fno, model)

        # 頂点モーフ
        vertex_morph_poses = self.morphs.animate_vertex_morphs(fno, model)
        # ボーンモーフ
        bone_morph_names, bone_morphs = self.morphs.animate_bone_morphs(fno, model)
        bone_morph_poses, bone_morph_qqs = bone_morphs.animate_bone_matrixes(fno, model, bone_morph_names)
        # グループモーフ
        group_vertex_morph_poses, group_bone_morph_names, group_bone_morphs = self.morphs.animate_group_morphs(fno, model)
        group_bone_morph_poses, group_bone_morph_qqs = group_bone_morphs.animate_bone_matrixes(fno, model, group_bone_morph_names)

        # ボーン変形行列
        matrixes = MMatrix4x4List(bone_poses.shape[0], bone_poses.shape[1])
        matrixes.translate(bone_morph_poses.tolist())
        matrixes.rotate(bone_morph_qqs.tolist())
        matrixes.translate(group_bone_morph_poses.tolist())
        matrixes.rotate(group_bone_morph_qqs.tolist())
        matrixes.translate(bone_poses.tolist())
        matrixes.rotate(bone_qqs.tolist())

        bone_matrixes: list[np.ndarray] = []
        for bone in model.bones:
            # 全体のボーン変形行列を求める
            matrix = model.bones.get_mesh_gl_matrix(matrixes, bone.index, np.eye(4))

            # BOf行列: 自身のボーンのボーンオフセット行列
            matrix = matrix @ bone.offset_matrix.copy().vector

            bone_matrixes.append(matrix.T)

        return (np.array(bone_matrixes), vertex_morph_poses + group_vertex_morph_poses)
