import os
from bisect import bisect_left
from functools import lru_cache
from math import acos, degrees
from typing import Optional
from itertools import product

import numpy as np
from numpy.linalg import inv

from mlib.base.collection import BaseHashModel, BaseIndexNameDictModel, BaseIndexNameDictWrapperModel
from mlib.base.logger import MLogger
from mlib.base.math import MMatrix4x4, MMatrix4x4List, MQuaternion, MVector3D, MVector4D
from mlib.pmx.pmx_collection import BoneTree, PmxModel
from mlib.pmx.pmx_part import (
    Bone,
    BoneMorphOffset,
    GroupMorphOffset,
    Material,
    MaterialMorphCalcMode,
    MaterialMorphOffset,
    MorphType,
    ShaderMaterial,
    UvMorphOffset,
    VertexMorphOffset,
)
from mlib.pmx.shader import MShader
from mlib.vmd.vmd_part import VmdBoneFrame, VmdCameraFrame, VmdLightFrame, VmdMorphFrame, VmdShadowFrame, VmdShowIkFrame
from mlib.vmd.vmd_tree import VmdBoneFrameTrees

logger = MLogger(os.path.basename(__file__), level=1)


class VmdBoneNameFrames(BaseIndexNameDictModel[VmdBoneFrame]):
    """
    ボーン名別キーフレ辞書
    """

    __slots__ = (
        "data",
        "name",
        "cache",
        "_names",
        "_indexes",
        "_iter_index",
        "_ik_indexes",
        "_size",
    )

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self._ik_indexes: list[int] = []

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
        if value.ik_rotation is not None and value.index not in self._ik_indexes:
            self._ik_indexes.append(value.index)
            self._ik_indexes.sort()
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
            next_bf = self.data[next_index]
            bf.position = next_bf.position.copy()
            bf.local_position = next_bf.local_position.copy()
            bf.rotation = next_bf.rotation.copy()
            bf.local_rotation = next_bf.local_rotation.copy()
            bf.scale = next_bf.scale.copy()
            bf.local_scale = next_bf.local_scale.copy()
            return bf

        prev_bf = self.data[prev_index] if prev_index in self else VmdBoneFrame(name=self.name, index=prev_index)
        next_bf = self.data[next_index] if next_index in self else VmdBoneFrame(name=self.name, index=next_index)

        slice_idx = bisect_left(self._ik_indexes, index)
        prev_ik_indexes = self._ik_indexes[:slice_idx]
        next_ik_indexes = self._ik_indexes[slice_idx:]

        prev_ik_index = prev_ik_indexes[-1] if prev_ik_indexes else prev_index
        prev_ik_rotation = self.data[prev_ik_index].ik_rotation or MQuaternion() if prev_ik_index in self.data else MQuaternion()

        next_ik_index = next_ik_indexes[0] if next_ik_indexes else next_index
        next_ik_rotation = self.data[next_ik_index].ik_rotation or MQuaternion() if next_ik_index in self.data else prev_ik_rotation

        # 補間結果Yは、FKキーフレ内で計算する
        ry, xy, yy, zy = next_bf.interpolations.evaluate(prev_index, index, next_index)

        # IK用回転
        bf.ik_rotation = MQuaternion.slerp(prev_ik_rotation, next_ik_rotation, ry)

        # FK用回転
        bf.rotation = MQuaternion.slerp(prev_bf.rotation, next_bf.rotation, ry)

        # 移動
        bf.position = MVector3D.calc_by_ratio(prev_bf.position, next_bf.position, xy, yy, zy)

        # スケール
        bf.scale = MVector3D.calc_by_ratio(prev_bf.scale, next_bf.scale, xy, yy, zy)

        # ローカル回転
        bf.local_rotation = MQuaternion.slerp(prev_bf.local_rotation, next_bf.local_rotation, ry)

        # ローカル移動
        bf.local_position = MVector3D.calc_by_ratio(prev_bf.local_position, next_bf.local_position, xy, yy, zy)

        # ローカルスケール
        bf.local_scale = MVector3D.calc_by_ratio(prev_bf.local_scale, next_bf.local_scale, xy, yy, zy)

        return bf


class VmdBoneFrames(BaseIndexNameDictWrapperModel[VmdBoneNameFrames]):
    """
    ボーンキーフレ辞書
    """

    def __init__(self) -> None:
        super().__init__()

    def create(self, key: str) -> VmdBoneNameFrames:
        return VmdBoneNameFrames(name=key)

    @property
    def max_fno(self) -> int:
        return max([max(self[bname].indexes + [0]) for bname in self.names] + [0])

    def animate_bone_matrixes(
        self,
        fnos: list[int],
        model: PmxModel,
        morph_bone_frames: Optional["VmdBoneFrames"] = None,
        bone_names: list[str] = [],
        append_ik: bool = True,
    ) -> VmdBoneFrameTrees:
        # モーフボーン操作
        if morph_bone_frames is not None:
            (
                morph_bone_poses,
                morph_bone_qqs,
                morph_bone_scales,
                morph_bone_local_poses,
                morph_bone_local_qqs,
                morph_bone_local_scales,
            ) = morph_bone_frames.get_bone_matrixes(fnos, model, bone_names, append_ik=False)
            # logger.debug(f"-- ボーンアニメーション[{model.name}]: モーフボーン操作")
        else:
            morph_row = len(fnos)
            morph_col = len(model.bones)
            morph_bone_poses = np.full((morph_row, morph_col, 4, 4), np.eye(4))
            morph_bone_qqs = np.full((morph_row, morph_col, 4, 4), np.eye(4))
            morph_bone_scales = np.full((morph_row, morph_col, 4, 4), np.eye(4))
            morph_bone_local_poses = np.full((morph_row, morph_col, 4, 4), np.eye(4))
            morph_bone_local_qqs = np.full((morph_row, morph_col, 4, 4), np.eye(4))
            morph_bone_local_scales = np.full((morph_row, morph_col, 4, 4), np.eye(4))

        # モーションボーン操作
        (
            motion_bone_poses,
            motion_bone_qqs,
            motion_bone_scales,
            motion_bone_local_poses,
            motion_bone_local_qqs,
            motion_bone_local_scales,
        ) = self.get_bone_matrixes(fnos, model, bone_names, append_ik=append_ik)
        # logger.debug(f"-- ボーンアニメーション[{model.name}]: モーションボーン操作")

        # ボーン変形行列
        matrixes = MMatrix4x4List(motion_bone_poses.shape[0], motion_bone_poses.shape[1])

        # モーフの適用
        matrixes.matmul(morph_bone_poses)
        matrixes.matmul(morph_bone_local_poses)
        matrixes.matmul(morph_bone_qqs)
        matrixes.matmul(morph_bone_local_qqs)
        matrixes.matmul(morph_bone_scales)
        matrixes.matmul(morph_bone_local_scales)

        # モーションの適用
        matrixes.matmul(motion_bone_poses)
        matrixes.matmul(motion_bone_local_poses)
        matrixes.matmul(motion_bone_qqs)
        matrixes.matmul(motion_bone_local_qqs)
        matrixes.matmul(motion_bone_scales)
        matrixes.matmul(motion_bone_local_scales)

        # 各ボーンごとのボーン変形行列結果と逆BOf行列(初期姿勢行列)の行列積
        relative_matrixes = np.array([np.array([bone.parent_revert_matrix @ matrixes[fidx, bone.index] for bone in model.bones]) for fidx in range(len(fnos))])

        # ボーンツリーINDEXリストごとのボーン変形行列リスト(子どもから親に遡る)
        tree_relative_matrixes = [[relative_matrixes[fidx, list(reversed(bone.tree_indexes))] for bone in model.bones] for fidx in range(len(fnos))]

        bone_indexes = model.bones.indexes
        if bone_names:
            bone_indexes = sorted(set([bone_index for bone_name in bone_names for bone_index in model.bones[bone_name].tree_indexes]))

        # 行列積ボーン変形行列結果
        result_matrixes = np.full(relative_matrixes.shape, np.eye(4))

        for fidx, bone_index in product(list(range(len(fnos))), bone_indexes):
            result_matrixes[fidx, bone_index] = model.bones[bone_index].offset_matrix.copy()
            for matrix in tree_relative_matrixes[fidx][bone_index]:
                result_matrixes[fidx, bone_index] = matrix @ result_matrixes[fidx, bone_index]

        bone_matrixes = VmdBoneFrameTrees()
        for fidx, fno in enumerate(fnos):
            for bone in model.bones:
                local_matrix = MMatrix4x4()
                local_matrix.vector = result_matrixes[fidx, bone.index]

                pos_mat = np.eye(4)
                pos_mat[:3, 3] = bone.position.vector

                # グローバル行列は最後にボーン位置に移動させる
                global_matrix = MMatrix4x4()
                global_matrix.vector = result_matrixes[fidx, bone.index] @ pos_mat

                bone_matrixes.append(fno, bone.name, global_matrix, local_matrix, global_matrix.to_position())

        return bone_matrixes

    def get_bone_matrixes(
        self,
        fnos: list[int],
        model: PmxModel,
        bone_names: list[str] = [],
        append_ik: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ボーン変形行列を求める"""

        row = len(fnos)
        col = len(model.bones)
        poses = np.full((row, col, 4, 4), np.eye(4))
        qqs = np.full((row, col, 4, 4), np.eye(4))
        scales = np.full((row, col, 4, 4), np.eye(4))
        local_poses = np.full((row, col, 4, 4), np.eye(4))
        local_qqs = np.full((row, col, 4, 4), np.eye(4))
        local_scales = np.full((row, col, 4, 4), np.eye(4))

        if not bone_names:
            target_bone_names = model.bones.names
        else:
            target_bone_names = [
                model.bones[bone_index].name
                for bone_index in sorted(set([bone_index for bone_name in bone_names for bone_index in model.bones[bone_name].relative_bone_indexes]))
            ]

        if append_ik:
            # IK回転を事前に求めておく
            for fno in fnos:
                self.calc_ik_rotations(fno, model, target_bone_names)

        for i, fno in enumerate(fnos):
            fno_poses: dict[int, MVector3D] = {}
            fno_scales: dict[int, MVector3D] = {}
            fno_local_poses: dict[int, MVector3D] = {}
            fno_local_qqs: dict[int, MQuaternion] = {}
            fno_local_scales: dict[int, MVector3D] = {}

            is_valid_local_pos = False
            is_valid_local_rot = False
            is_valid_local_scale = False

            for bone_name in target_bone_names:
                bone = model.bones[bone_name]
                if bone.index in fno_poses:
                    continue
                bf = self[bone.name][fno]
                fno_poses[bone.index] = bf.position
                fno_scales[bone.index] = bf.scale
                fno_local_poses[bone.index] = bf.local_position
                fno_local_qqs[bone.index] = bf.local_rotation
                fno_local_scales[bone.index] = bf.local_scale

                is_valid_local_pos = is_valid_local_pos or bool(bf.local_position)
                is_valid_local_rot = is_valid_local_rot or bool(bf.local_rotation)
                is_valid_local_scale = is_valid_local_scale or bool(bf.local_scale)

            for bone_name in target_bone_names:
                bone = model.bones[bone_name]

                is_parent_bone_not_local_cancels: list[bool] = []
                parent_local_poses: list[MVector3D] = []
                parent_local_qqs: list[MQuaternion] = []
                parent_local_scales: list[MVector3D] = []
                parent_local_axises: list[MVector3D] = []

                for parent_index in bone.tree_indexes[:-1]:
                    parent_bone = model.bones[parent_index]
                    if parent_bone.index not in fno_local_poses:
                        parent_bf = self[parent_bone.name][fno]
                        fno_local_poses[parent_bone.index] = parent_bf.local_position
                        fno_local_qqs[parent_bone.index] = parent_bf.local_rotation
                        fno_local_scales[parent_bone.index] = parent_bf.local_scale
                        fno_local_poses[parent_bone.index] = parent_bf.local_position
                    is_parent_bone_not_local_cancels.append(model.bones.is_bone_not_local_cancels[parent_bone.index])
                    parent_local_axises.append(model.bones.local_axises[parent_bone.index])
                    parent_local_poses.append(fno_local_poses[parent_bone.index])
                    parent_local_qqs.append(fno_local_qqs[parent_bone.index])
                    parent_local_scales.append(fno_local_scales[parent_bone.index])

                poses[i, bone.index] = self.get_position(fno_poses, bone, model)

                # モーションによるローカル移動量
                if is_valid_local_pos:
                    local_pos_mat = self.get_local_position(
                        fno_local_poses,
                        bone,
                        tuple(is_parent_bone_not_local_cancels),
                        tuple(parent_local_poses),
                        tuple(parent_local_axises),
                    )
                    local_poses[i, bone.index] = local_pos_mat

                # FK(捩り) > IK(捩り) > 付与親(捩り)
                qq = self.get_rotation(fno, bone, model, append_ik=append_ik)
                qqs[i, bone.index] = qq.to_matrix4x4().vector

                # ローカル回転
                if is_valid_local_rot:
                    local_rot_mat = self.get_local_rotation(
                        fno_local_qqs,
                        bone,
                        tuple(is_parent_bone_not_local_cancels),
                        tuple(parent_local_qqs),
                        tuple(parent_local_axises),
                    )
                    local_qqs[i, bone.index] = local_rot_mat

                # モーションによるスケール変化
                scale_mat = self.get_scale(fno_scales, bone, model)
                scales[i, bone.index] = scale_mat

                # ローカルスケール
                if is_valid_local_scale:
                    local_scale_mat = self.get_local_scale(
                        fno_local_scales,
                        bone,
                        tuple(is_parent_bone_not_local_cancels),
                        tuple(parent_local_scales),
                        tuple(parent_local_axises),
                    )
                    local_scales[i, bone.index] = local_scale_mat

        return poses, qqs, scales, local_poses, local_qqs, local_scales

    def get_position(self, fno_poses: dict[int, MVector3D], bone: Bone, model: PmxModel) -> np.ndarray:
        """
        該当キーフレにおけるボーンの移動位置
        """
        # 自身の位置
        mat = np.eye(4)
        mat[:3, 3] = fno_poses[bone.index].vector

        # 付与親を加味して返す
        return mat @ self.get_effect_position(fno_poses, bone, model)

    def get_effect_position(self, fno_poses: dict[int, MVector3D], bone: Bone, model: PmxModel) -> np.ndarray:
        """
        付与親を加味した移動を求める
        """
        if not (bone.is_external_translation and bone.effect_index in model.bones):
            return np.eye(4)

        if 0 == bone.effect_factor:
            # 付与率が0の場合、常に0になる
            return np.eye(4)

        # 付与親の移動量を取得する（それが付与持ちなら更に遡る）
        effect_bone = model.bones[bone.effect_index]
        effect_pos_mat = self.get_position(fno_poses, effect_bone, model)
        # 付与率を加味する
        effect_pos_mat[:3, 3] *= bone.effect_factor

        return effect_pos_mat

    def get_local_position(
        self,
        fno_local_poses: dict[int, MVector3D],
        bone: Bone,
        is_parent_bone_not_local_cancels: tuple[bool],
        parent_local_poses: tuple[MVector3D],
        parent_local_axises: tuple[MVector3D],
    ) -> np.ndarray:
        """
        該当キーフレにおけるボーンのローカル位置
        """
        # 自身のローカル移動量
        local_pos = fno_local_poses[bone.index]

        return calc_local_position(
            bone.is_not_local_cancel,
            local_pos,
            bone.tail_relative_position,
            is_parent_bone_not_local_cancels,
            parent_local_poses,
            parent_local_axises,
        )

    def calc_ik_rotations(self, fno: int, model: PmxModel, target_bone_names: list[str]):
        """IK関連ボーンの事前計算"""
        ik_target_names = [model.bone_trees[target_bone_name].last_name for target_bone_name in target_bone_names if model.bones[target_bone_name].is_ik]
        if not ik_target_names:
            # IK計算対象がない場合はそのまま終了
            return

        ik_relative_bone_names = [
            model.bones[bone_index].name
            for bone_index in sorted(
                set([relative_bone_index for ik_target_name in ik_target_names for relative_bone_index in model.bones[ik_target_name].relative_bone_indexes])
            )
        ]

        # モーション内のキーフレリストから前の変化キーフレと次の変化キーフレを抽出する
        prev_frame_indexes: set[int] = {0}
        next_frame_indexes: set[int] = {self.max_fno}
        for relative_bone_name in ik_relative_bone_names:
            if relative_bone_name not in self:
                continue
            (prev_fno, _, next_fno) = self[relative_bone_name].range_indexes(fno)
            if prev_fno != fno:
                prev_frame_indexes |= {prev_fno}
            if next_fno != fno:
                next_frame_indexes |= {next_fno}

        for fno in (max(prev_frame_indexes), min(next_frame_indexes)):
            fno_qqs: dict[int, MQuaternion] = {}
            fno_ik_qqs: dict[int, Optional[MQuaternion]] = {}

            for relative_bone_name in ik_relative_bone_names:
                bone = model.bones[relative_bone_name]
                bf = self[relative_bone_name][fno]
                fno_qqs[bone.index] = bf.rotation
                fno_ik_qqs[bone.index] = bf.ik_rotation

            for relative_bone_name in ik_relative_bone_names:
                self.get_rotation(fno, bone, model, append_ik=True)

    def get_scale(self, fno_scales: dict[int, MVector3D], bone: Bone, model: PmxModel) -> np.ndarray:
        """
        該当キーフレにおけるボーンの縮尺
        """
        # 自身のスケール
        scale_mat = np.eye(4)
        scale_mat[:3, :3] += np.diag(fno_scales[bone.index].vector)

        # 付与親を加味して返す
        return self.get_effect_scale(scale_mat, fno_scales, bone, model)

    def get_effect_scale(self, scale_mat: np.ndarray, fno_scales: dict[int, MVector3D], bone: Bone, model: PmxModel) -> np.ndarray:
        """
        付与親を加味した縮尺を求める
        """
        if not (bone.is_external_translation and bone.effect_index in model.bones):
            return scale_mat

        if 0 == bone.effect_factor:
            # 付与率が0の場合、常に1になる
            return np.eye(4)

        # 付与親の回転量を取得する（それが付与持ちなら更に遡る）
        effect_bone = model.bones[bone.effect_index]
        effect_scale_mat = self.get_scale(fno_scales, effect_bone, model)

        return scale_mat @ effect_scale_mat

    def get_local_scale(
        self,
        fno_local_scales: dict[int, MVector3D],
        bone: Bone,
        is_parent_bone_not_local_cancels: tuple[bool],
        parent_local_scales: tuple[MVector3D],
        parent_local_axises: tuple[MVector3D],
    ) -> np.ndarray:
        """
        該当キーフレにおけるボーンのローカル縮尺
        """
        # 自身のローカルスケール
        local_scale = fno_local_scales[bone.index]

        return calc_local_scale(
            bone.is_not_local_cancel,
            local_scale,
            bone.tail_relative_position,
            is_parent_bone_not_local_cancels,
            parent_local_scales,
            parent_local_axises,
        )

    def get_rotation(
        self,
        fno: int,
        bone: Bone,
        model: PmxModel,
        append_ik: bool = False,
    ) -> MQuaternion:
        """
        該当キーフレにおけるボーンの相対位置
        append_ik : IKを計算するか(循環してしまう場合があるので、デフォルトFalse)
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
        effect_qq = self.get_effect_rotation(fno, bone, ik_qq, model, append_ik)

        return effect_qq

    def get_effect_rotation(
        self,
        fno: int,
        bone: Bone,
        qq: MQuaternion,
        model: PmxModel,
        append_ik: bool,
    ) -> MQuaternion:
        """
        付与親を加味した回転を求める
        """
        if not (bone.is_external_rotation and bone.effect_index in model.bones):
            return qq

        if 0 == bone.effect_factor:
            # 付与率が0の場合、常に0になる
            return MQuaternion()

        # 付与親の回転量を取得する（それが付与持ちなら更に遡る）
        effect_bone = model.bones[bone.effect_index]
        effect_qq = self.get_rotation(fno, effect_bone, model, append_ik=append_ik)
        if 0 < bone.effect_factor:
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
        """

        if not bone.ik_link_indexes:
            return qq

        for ik_target_bone_idx in bone.ik_link_indexes:
            # IKボーン自身の位置
            ik_bone = model.bones[ik_target_bone_idx]

            if ik_target_bone_idx not in model.bones or not ik_bone.ik:
                continue

            ik_matrixes = self.animate_bone_matrixes([fno], model, bone_names=[ik_bone.name], append_ik=False)
            global_target_pos = ik_matrixes[fno, ik_bone.name].position

            # IKターゲットボーンツリー
            effector_bone = model.bones[ik_bone.ik.bone_index]

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
                    effector_matrixes = self.animate_bone_matrixes([fno], model, bone_names=[effector_bone.name], append_ik=False)
                    global_effector_pos = effector_matrixes[fno, effector_bone.name].position

                    # 処理対象IKボーン
                    link_bone = model.bones[ik_link.bone_index]

                    # リンクボーンの角度を保持
                    link_bf = self[link_bone.name][fno]

                    # 処理対象IKボーンのグローバル位置と行列を取得
                    link_matrixes = self.animate_bone_matrixes([fno], model, bone_names=[link_bone.name], append_ik=False)
                    # 注目ノード（実際に動かすボーン）
                    link_matrix = link_matrixes[fno, link_bone.name].global_matrix

                    # ワールド座標系から注目ノードの局所座標系への変換
                    link_inverse_matrix = link_matrix.inverse()

                    # 注目ノードを起点とした、エフェクタのローカル位置
                    local_effector_pos = link_inverse_matrix * global_effector_pos
                    # 注目ノードを起点とした、IK目標のローカル位置
                    local_target_pos = link_inverse_matrix * global_target_pos

                    if 1e-5 > (local_effector_pos - local_target_pos).length_squared():
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
                    rotation_axis = norm_effector_pos.cross(norm_target_pos)
                    # 回転角度
                    rotation_degree = degrees(rotation_radian)

                    # 制限角で最大変位量を制限する
                    if 0 < loop:
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
        """
        if bone.has_fixed_axis:
            return qq.to_fixed_axis_quaternion(bone.corrected_fixed_axis)

        return qq

    def get_local_rotation(
        self,
        fno_local_qqs: dict[int, MQuaternion],
        bone: Bone,
        is_parent_bone_not_local_cancels: tuple[bool],
        parent_local_qqs: tuple[MQuaternion],
        parent_local_axises: tuple[MVector3D],
    ) -> np.ndarray:
        """
        該当キーフレにおけるボーンのローカル回転
        """
        # 自身のローカル回転量
        local_qq = fno_local_qqs[bone.index]

        return calc_local_rotation(
            bone.is_not_local_cancel,
            local_qq,
            bone.tail_relative_position,
            is_parent_bone_not_local_cancels,
            parent_local_qqs,
            parent_local_axises,
        )


@lru_cache(maxsize=None)
def calc_local_position(
    is_bone_not_local_cancel: bool,
    local_pos: MVector3D,
    local_axis: MVector3D,
    is_parent_bone_not_local_cancels: tuple[bool],
    parent_local_poses: tuple[MVector3D],
    parent_local_axises: tuple[MVector3D],
) -> np.ndarray:
    local_parent_matrix = np.eye(4)

    # 親を辿る
    if not is_bone_not_local_cancel:
        for n in range(1, len(parent_local_axises) + 1):
            local_parent_matrix = local_parent_matrix @ calc_local_position(
                is_parent_bone_not_local_cancels[-n],
                parent_local_poses[-n],
                parent_local_axises[-n],
                tuple(is_parent_bone_not_local_cancels[:-n]),
                tuple(parent_local_poses[:-n]),
                tuple(parent_local_axises[:-n]),
            )

    # ローカル軸に沿った回転行列
    rotation_matrix = local_axis.to_local_matrix4x4().vector

    local_pos_mat = np.eye(4)
    local_pos_mat[:3, 3] = local_pos.vector

    # ローカル軸に合わせた移動行列を作成する(親はキャンセルする)
    return inv(local_parent_matrix) @ inv(rotation_matrix) @ local_pos_mat @ rotation_matrix


@lru_cache(maxsize=None)
def calc_local_rotation(
    is_bone_not_local_cancel: bool,
    local_qq: MQuaternion,
    local_axis: MVector3D,
    is_parent_bone_not_local_cancels: tuple[bool],
    parent_local_qqs: tuple[MQuaternion],
    parent_local_axises: tuple[MVector3D],
) -> np.ndarray:
    local_parent_matrix = np.eye(4)

    # 親を辿る
    if not is_bone_not_local_cancel:
        for n in range(1, len(parent_local_axises) + 1):
            local_parent_matrix = local_parent_matrix @ calc_local_rotation(
                is_parent_bone_not_local_cancels[-n],
                parent_local_qqs[-n],
                parent_local_axises[-n],
                tuple(is_parent_bone_not_local_cancels[:-n]),
                tuple(parent_local_qqs[:-n]),
                tuple(parent_local_axises[:-n]),
            )

    # ローカル軸に沿った回転行列
    rotation_matrix = local_axis.to_local_matrix4x4().vector

    local_rot_mat = local_qq.to_matrix4x4().vector

    # ローカル軸に合わせた移動行列を作成する(親はキャンセルする)
    return inv(local_parent_matrix) @ inv(rotation_matrix) @ local_rot_mat @ rotation_matrix


@lru_cache(maxsize=None)
def calc_local_scale(
    is_bone_not_local_cancel: bool,
    local_scale: MVector3D,
    local_axis: MVector3D,
    is_parent_bone_not_local_cancels: tuple[bool],
    parent_local_scales: tuple[MVector3D],
    parent_local_axises: tuple[MVector3D],
) -> np.ndarray:
    local_parent_matrix = np.eye(4)

    # 親を辿る
    if not is_bone_not_local_cancel:
        for n in range(1, len(parent_local_axises) + 1):
            local_parent_matrix = local_parent_matrix @ calc_local_scale(
                is_parent_bone_not_local_cancels[-n],
                parent_local_scales[-n],
                parent_local_axises[-n],
                tuple(is_parent_bone_not_local_cancels[:-n]),
                tuple(parent_local_scales[:-n]),
                tuple(parent_local_axises[:-n]),
            )

    # ローカル軸に沿った回転行列
    rotation_matrix = local_axis.to_local_matrix4x4().vector

    local_scale_mat = np.eye(4)
    local_scale_mat[:3, :3] += np.diag(local_scale.vector)

    # ローカル軸に合わせた移動行列を作成する(親はキャンセルする)
    return inv(local_parent_matrix) @ inv(rotation_matrix) @ local_scale_mat @ rotation_matrix


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
        mf.ratio = prev_mf.ratio + (next_mf.ratio - prev_mf.ratio) * ry

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
            if morph.name not in self.data:
                # モーフそのものの定義がなければスルー
                continue
            mf = self[morph.name][fno]
            if not mf.ratio:
                continue

            # モーションによる頂点モーフ変動量
            for offset in morph.offsets:
                if type(offset) is VertexMorphOffset and offset.vertex_index < row:
                    ratio_pos: MVector3D = offset.position_offset * mf.ratio
                    poses[offset.vertex_index] += ratio_pos.gl.vector

        return np.array(poses)

    def animate_uv_morphs(self, fno: int, model: PmxModel, uv_index: int) -> np.ndarray:
        row = len(model.vertices)
        poses = np.full((row, 4), np.zeros(4))

        target_uv_type = MorphType.UV if 0 == uv_index else MorphType.EXTENDED_UV1
        for morph in model.morphs.filter_by_type(target_uv_type):
            if morph.name not in self.data:
                # モーフそのものの定義がなければスルー
                continue
            mf = self[morph.name][fno]
            if not mf.ratio:
                continue

            # モーションによるUVモーフ変動量
            for offset in morph.offsets:
                if type(offset) is UvMorphOffset and offset.vertex_index < row:
                    ratio_pos: MVector4D = offset.uv * mf.ratio
                    poses[offset.vertex_index] += ratio_pos.vector

        # UVのYは 1 - y で求め直しておく
        poses[:, 1] = 1 - poses[:, 1]

        return np.array(poses)

    def animate_bone_morphs(self, fno: int, model: PmxModel) -> VmdBoneFrames:
        bone_frames = VmdBoneFrames()
        for morph in model.morphs.filter_by_type(MorphType.BONE):
            if morph.name not in self.data:
                # モーフそのものの定義がなければスルー
                continue
            mf = self[morph.name][fno]
            if not mf.ratio:
                continue

            # モーションによるボーンモーフ変動量
            for offset in morph.offsets:
                if type(offset) is BoneMorphOffset and offset.bone_index in model.bones:
                    bf = bone_frames[model.bones[offset.bone_index].name][fno]
                    bf = self.animate_bone_morph_frame(fno, model, bf, offset, mf.ratio)
                    bone_frames[bf.name][fno] = bf

        return bone_frames

    def animate_bone_morph_frame(self, fno: int, model: PmxModel, bf: VmdBoneFrame, offset: BoneMorphOffset, ratio: float) -> VmdBoneFrame:
        bf.position += offset.position * ratio
        bf.local_position += offset.local_position * ratio
        bf.rotation *= MQuaternion.from_euler_degrees(offset.rotation.degrees * ratio)
        bf.local_rotation *= MQuaternion.from_euler_degrees(offset.local_rotation.degrees * ratio)
        bf.scale += offset.scale * ratio
        bf.local_scale += offset.local_scale * ratio
        return bf

    def animate_group_morphs(self, fno: int, model: PmxModel, materials: list[ShaderMaterial]) -> tuple[np.ndarray, VmdBoneFrames, list[ShaderMaterial]]:
        group_vertex_poses = np.full((len(model.vertices), 3), np.zeros(3))
        bone_frames = VmdBoneFrames()

        # デフォルトの材質情報を保持（シェーダーに合わせて一部入れ替え）
        for morph in model.morphs.filter_by_type(MorphType.GROUP):
            if morph.name not in self.data:
                # モーフそのものの定義がなければスルー
                continue
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
                            bf = bone_frames[model.bones[offset.bone_index].name][fno]
                            bf = self.animate_bone_morph_frame(fno, model, bf, offset, mf_factor)
                            bone_frames[bf.name][fno] = bf
                        elif type(offset) is MaterialMorphOffset and offset.material_index in model.materials:
                            materials = self.animate_material_morph_frame(model, offset, mf_factor, materials, MShader.LIGHT_AMBIENT4)

        return group_vertex_poses, bone_frames, materials

    def animate_material_morph_frame(
        self, model: PmxModel, offset: MaterialMorphOffset, ratio: float, materials: list[ShaderMaterial], light_ambient: MVector4D
    ) -> list[ShaderMaterial]:
        if 0 > offset.material_index:
            # 0の場合、全材質を対象とする
            material_indexes = model.materials.indexes
        else:
            # 特定材質の場合、材質固定
            material_indexes = [offset.material_index]
        # 指定材質を対象として変動量を割り当てる
        for target_calc_mode in [MaterialMorphCalcMode.MULTIPLICATION, MaterialMorphCalcMode.ADDITION]:
            # 先に乗算を計算した後に加算を加味する
            for material_index in material_indexes:
                # 元々の材質情報をコピー
                mat = model.materials[material_index]

                # オフセットに合わせた材質情報
                material = Material(
                    mat.index,
                    mat.name,
                    mat.english_name,
                )
                material.diffuse = offset.diffuse
                material.ambient = offset.ambient
                material.specular = offset.specular
                material.edge_color = offset.edge_color
                material.edge_size = offset.edge_size

                material_offset = ShaderMaterial(
                    material,
                    light_ambient,
                    offset.texture_factor,
                    offset.toon_texture_factor,
                    offset.sphere_texture_factor,
                )

                # オフセットに合わせた材質情報
                material_offset *= ratio
                if offset.calc_mode == target_calc_mode:
                    if offset.calc_mode == MaterialMorphCalcMode.ADDITION:
                        # 加算
                        materials[material_index] += material_offset
                    else:
                        # 乗算
                        materials[material_index] *= material_offset

        return materials

    def animate_material_morphs(self, fno: int, model: PmxModel) -> list[ShaderMaterial]:
        # デフォルトの材質情報を保持（シェーダーに合わせて一部入れ替え）
        materials = [ShaderMaterial(m, MShader.LIGHT_AMBIENT4) for m in model.materials]

        for morph in model.morphs.filter_by_type(MorphType.MATERIAL):
            if morph.name not in self.data:
                # モーフそのものの定義がなければスルー
                continue
            mf = self[morph.name][fno]
            if not mf.ratio:
                continue

            # モーションによる材質モーフ変動量
            for offset in morph.offsets:
                if type(offset) is MaterialMorphOffset and (offset.material_index in model.materials or 0 > offset.material_index):
                    materials = self.animate_material_morph_frame(model, offset, mf.ratio, materials, MShader.LIGHT_AMBIENT4)

        return materials


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

    __slots__ = (
        "path",
        "digest",
        "signature",
        "model_name",
        "bones",
        "morphs",
        "cameras",
        "lights",
        "shadows",
        "show_iks",
    )

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

    def animate(self, fno: int, model: PmxModel) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[ShaderMaterial]]:
        logger.debug(f"-- スキンメッシュアニメーション[{model.name}][{fno:04d}]: 開始")

        # 頂点モーフ
        vertex_morph_poses = self.morphs.animate_vertex_morphs(fno, model)
        logger.debug(f"-- スキンメッシュアニメーション[{model.name}][{fno:04d}]: 頂点モーフ")

        # UVモーフ
        uv_morph_poses = self.morphs.animate_uv_morphs(fno, model, 0)
        logger.debug(f"-- スキンメッシュアニメーション[{model.name}][{fno:04d}]: UVモーフ")

        # 追加UVモーフ1
        uv1_morph_poses = self.morphs.animate_uv_morphs(fno, model, 1)
        logger.debug(f"-- スキンメッシュアニメーション[{model.name}][{fno:04d}]: 追加UVモーフ1")

        # 追加UVモーフ2-4は無視

        # 材質モーフ
        material_morphs = self.morphs.animate_material_morphs(fno, model)
        logger.debug(f"-- スキンメッシュアニメーション[{model.name}][{fno:04d}]: 材質モーフ")

        # グループモーフ
        group_vertex_morph_poses, group_morph_bone_frames, group_materials = self.morphs.animate_group_morphs(fno, model, material_morphs)
        logger.debug(f"-- スキンメッシュアニメーション[{model.name}][{fno:04d}]: グループモーフ")

        bone_matrixes = self.animate_bone([fno], model)

        # OpenGL座標系に変換

        gl_matrixes = np.array([matrix.local_matrix.vector.T for matrix in bone_matrixes.data.values()])
        # gl_matrixes = np.array(bone_matrixes)

        gl_matrixes[..., 0, 1:3] *= -1
        gl_matrixes[..., 1:3, 0] *= -1
        gl_matrixes[..., 3, 0] *= -1

        logger.debug(f"-- スキンメッシュアニメーション[{model.name}][{fno:04d}]: OpenGL座標系変換")

        return gl_matrixes, vertex_morph_poses + group_vertex_morph_poses, uv_morph_poses, uv1_morph_poses, group_materials

    def animate_bone(self, fnos: list[int], model: PmxModel, bone_names: list[str] = [], append_ik: bool = True) -> VmdBoneFrameTrees:
        all_morph_bone_frames = VmdBoneFrames()

        for fno in fnos:
            logger.debug(f"-- ボーンアニメーション[{model.name}][{fno:04d}]: 開始")

            # 材質モーフ
            material_morphs = self.morphs.animate_material_morphs(fno, model)
            logger.debug(f"-- ボーンアニメーション[{model.name}][{fno:04d}]: 材質モーフ")

            # ボーンモーフ
            morph_bone_frames = self.morphs.animate_bone_morphs(fno, model)
            logger.debug(f"-- ボーンアニメーション[{model.name}][{fno:04d}]: ボーンモーフ")

            for bfs in morph_bone_frames:
                bf = bfs[fno]
                mbf = all_morph_bone_frames[bf.name][bf.index]
                all_morph_bone_frames[bf.name][bf.index] = mbf + bf

            # グループモーフ
            _, group_morph_bone_frames, _ = self.morphs.animate_group_morphs(fno, model, material_morphs)
            logger.debug(f"-- ボーンアニメーション[{model.name}][{fno:04d}]: グループモーフ")

            for bfs in group_morph_bone_frames:
                bf = bfs[fno]
                mbf = all_morph_bone_frames[bf.name][bf.index]
                all_morph_bone_frames[bf.name][bf.index] = mbf + bf

            logger.debug(f"-- ボーンアニメーション[{model.name}][{fno:04d}]: モーフキーフレ加算")

        # ボーン変形行列操作
        bone_matrixes = self.bones.animate_bone_matrixes(fnos, model, all_morph_bone_frames, bone_names, append_ik=append_ik)
        logger.debug(f"-- ボーンアニメーション[{model.name}]: ボーン変形行列操作")

        return bone_matrixes
