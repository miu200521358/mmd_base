import os
from glob import glob
from typing import Optional

import numpy as np
import OpenGL.GL as gl

from mlib.base.collection import BaseHashModel, BaseIndexDictModel, BaseIndexNameDictModel, BaseIndexNameDictWrapperModel
from mlib.base.exception import MViewerException
from mlib.base.logger import MLogger
from mlib.base.math import MMatrix4x4, MMatrix4x4List, MVector3D
from mlib.pmx.mesh import IBO, VAO, VBO, Mesh
from mlib.pmx.pmx_part import (
    STANDARD_BONE_NAMES,
    Bone,
    BoneMorphOffset,
    DisplaySlot,
    DisplayType,
    DrawFlg,
    Face,
    Joint,
    Material,
    Morph,
    MorphType,
    RigidBody,
    ShaderMaterial,
    Texture,
    TextureType,
    ToonSharing,
    Vertex,
)
from mlib.pmx.shader import MShader, ProgramType, VsLayout

logger = MLogger(os.path.basename(__file__))
__ = logger.get_text


class Vertices(BaseIndexDictModel[Vertex]):
    """
    頂点リスト
    """

    def __init__(self) -> None:
        super().__init__()


class Faces(BaseIndexDictModel[Face]):
    """
    面リスト
    """

    def __init__(self) -> None:
        super().__init__()


class Textures(BaseIndexNameDictModel[Texture]):
    """
    テクスチャリスト
    """

    def __init__(self) -> None:
        super().__init__()


class ToonTextures(BaseIndexNameDictModel[Texture]):
    """
    共有テクスチャ辞書
    """

    def __init__(self) -> None:
        super().__init__()


class Materials(BaseIndexNameDictModel[Material]):
    """
    材質リスト
    """

    def __init__(self) -> None:
        super().__init__()


class BoneTree(BaseIndexNameDictModel[Bone]):
    """ボーンリンク"""

    def __init__(self, name: str = "") -> None:
        super().__init__(name)

    def get_relative_position(self, key: int) -> MVector3D:
        """
        該当ボーンの相対位置を取得

        Parameters
        ----------
        key : int
            ボーンINDEX

        Returns
        -------
        ボーンの親ボーンから見た相対位置
        """
        if key not in self:
            return MVector3D()

        bone = self[key]
        if bone.parent_index not in self:
            return bone.position

        return bone.position - self[bone.parent_index]

    def filter(self, start_bone_name: str, end_bone_name: Optional[str] = None) -> "BoneTree":
        start_index = [i for i, b in enumerate(self.data.values()) if b.name == start_bone_name][0]
        if end_bone_name:
            end_index = [i for i, b in enumerate(self.data.values()) if b.name == end_bone_name][0]
        else:
            end_index = self.last_index
        new_tree = BoneTree(end_bone_name)
        for i, t in enumerate(self):
            if start_index <= i <= end_index:
                new_tree.append(t, is_sort=False)
        return new_tree


class BoneTrees(BaseIndexNameDictWrapperModel[BoneTree]):
    """
    BoneTreeリスト
    """

    def __init__(self) -> None:
        """モデル辞書"""
        super().__init__()

    def create(self, key: str) -> BoneTree:
        return BoneTree(key)

    def is_in_standard(self, name: str) -> bool:
        """準標準までのボーンツリーに含まれるボーンであるか否か"""
        # チェック対象ボーンが含まれるボーンツリー
        if name in STANDARD_BONE_NAMES:
            return True

        for bone_tree in [bt for bt in self.data.values() if name in bt.names and name != bt.last_name]:
            bone_find_index = [i for i, b in enumerate(bone_tree) if b.name == name][0]
            is_parent_standard = False
            is_child_standard = False
            # 親系統、子系統どちらにも準標準ボーンが含まれている場合、TRUE
            for parent_name in bone_tree.names[:bone_find_index]:
                if parent_name in STANDARD_BONE_NAMES:
                    is_parent_standard = True
                    break
            for child_name in bone_tree.names[bone_find_index + 1 :]:
                if child_name in STANDARD_BONE_NAMES:
                    is_child_standard = True
                    break
            if is_parent_standard and is_child_standard:
                return True

        return False

    def is_standard_tail(self, name: str) -> bool:
        """準標準までのボーンツリーに含まれるボーンの表示先であるか否か"""
        if name in STANDARD_BONE_NAMES:
            # そもそも自分が準標準ボーンならばFalse
            return False

        bone_tree = self.data[name]
        bone_find_index = [i for i, b in enumerate(bone_tree) if b.name == name][0]
        if 1 > bone_find_index:
            # そもそも子でない場合、表示先にはなり得ない
            return False

        bone = bone_tree[bone_tree.names[bone_find_index]]
        parent_bone = bone_tree[bone_tree.names[bone_find_index - 1]]

        if not self.is_in_standard(parent_bone.name):
            # そもそも親ボーンが準標準ボーンの範囲外ならばFalse
            return False

        for bt in self.data.values():
            if bt[bt.last_name].parent_index == bone.index:
                # 自分が親ボーンとして登録されている場合、False
                return False

        # 自分が準標準ボーンの範囲内の親を持ち、子ボーンがいない場合、表示先とみなす
        return True


class Bones(BaseIndexNameDictModel[Bone]):
    """
    ボーンリスト
    """

    def __init__(self) -> None:
        super().__init__()

    def get_by_index(self, index: int) -> Bone:
        if index == -1:
            return Bone()
        return super().get_by_index(index)

    def writable(self) -> list[Bone]:
        """出力対象となるボーン一覧を取得する"""
        bones: list[Bone] = []
        for b in self:
            if b.is_system:
                continue
            bones.append(b)
        return bones

    def create_bone_trees(self) -> BoneTrees:
        """
        ボーンツリー一括生成

        Returns
        -------
        BoneTrees
        """
        # ボーンツリー
        bone_trees = BoneTrees()
        total_index_count = len(self)

        # 計算ボーンリスト
        for i, end_bone in enumerate(self):
            # レイヤー込みのINDEXリスト取得を末端ボーンをキーとして保持
            bone_tree = BoneTree(name=end_bone.name)
            for _, bidx in sorted(self.create_bone_link_indexes(end_bone.index)):
                bone_tree.append(self.data[bidx].copy(), is_sort=False)
            bone_trees.append(bone_tree, name=end_bone.name)

            logger.count(
                "モデルセットアップ：ボーンツリー",
                index=i,
                total_index_count=total_index_count,
                display_block=500,
            )

        return bone_trees

    @property
    def tail_bone_names(self) -> list[str]:
        """
        親ボーンとして登録されていないボーン名リストを取得する
        """
        tail_bone_names = []
        parent_bone_indexes = []
        for end_bone in self:
            parent_bone_indexes.append(end_bone.parent_index)

        for end_bone in self:
            if end_bone.index not in parent_bone_indexes:
                tail_bone_names.append(end_bone.name)

        return tail_bone_names

    def create_bone_link_indexes(self, child_idx: int, bone_link_indexes=None) -> list[tuple[int, int]]:
        """
        指定ボーンの親ボーンを繋げてく

        Parameters
        ----------
        child_idx : int
            指定ボーンINDEX
        bone_link_indexes : _type_, optional
            既に構築済みの親ボーンリスト, by default None

        Returns
        -------
        親ボーンリスト
        """
        # 階層＞リスト順（＞FK＞IK＞付与）
        if not bone_link_indexes:
            bone_link_indexes = [(self.data[child_idx].layer, self.data[child_idx].index)]

        for b in reversed(self.data.values()):
            if b.index == self.data[child_idx].parent_index:
                bone_link_indexes.append((b.layer, b.index))
                return self.create_bone_link_indexes(b.index, bone_link_indexes)

        return bone_link_indexes

    def get_tail_relative_position(self, bone_index: int) -> MVector3D:
        """
        末端位置を取得

        Parameters
        ----------
        bone_index : int
            ボーンINDEX

        Returns
        -------
        ボーンの末端位置（グローバル位置）
        """
        if bone_index not in self:
            return MVector3D()

        bone = self[bone_index]
        to_pos = MVector3D()

        from_pos = bone.position
        if bone.name in STANDARD_BONE_NAMES and isinstance(STANDARD_BONE_NAMES[bone.name]["tail"], MVector3D):
            # 表示先位置が指定されてる場合、そのまま使用
            return STANDARD_BONE_NAMES[bone.name]["tail"]
        elif bone.name in STANDARD_BONE_NAMES:
            # 表示先ボーンが指定されており、いずれかある場合、そのまま使用
            for tail_bone_name in STANDARD_BONE_NAMES[bone.name]["tail"]:
                if tail_bone_name in self:
                    return self[tail_bone_name].position - from_pos

        # 合致するのがなければ通常の表示先から検出
        if bone.is_tail_bone and 0 <= bone.tail_index and bone.tail_index in self:
            # 表示先が指定されているの場合、保持
            to_pos = self[bone.tail_index].position
        elif not bone.is_tail_bone:
            # 表示先が相対パスの場合、保持
            to_pos = from_pos + bone.tail_position
        else:
            # 表示先がない場合、とりあえず親ボーンからの向きにする
            from_pos = self[bone.parent_index].position
            to_pos = self[bone_index].position

        return to_pos - from_pos

    def get_parent_relative_position(self, bone_index: int) -> MVector3D:
        """親ボーンから見た相対位置"""
        bone = self[bone_index]
        return bone.position - (MVector3D() if bone.index < 0 or bone.parent_index not in self else self[bone.parent_index].position)

    def get_mesh_matrix(self, matrixes: MMatrix4x4List, bone_index: int, matrix: np.ndarray) -> np.ndarray:
        """
        スキンメッシュアニメーション用ボーン変形行列を作成する

        Parameters
        ----------
        matrixes : MMatrix4x4List
            座標変換行列
        bone_index : int
            処理対象ボーンINDEX
        matrix : Optional[MMatrix4x4], optional
            計算元行列, by default None

        Returns
        -------
        ボーン変形行列
        """
        bone = self[bone_index]

        # 自身の姿勢をかける
        # 座標変換行列
        matrix = matrixes.vector[0, bone_index] @ matrix
        # 逆BOf行列(初期姿勢行列)
        matrix = bone.parent_revert_matrix.vector @ matrix

        if 0 <= bone.index and bone.parent_index in self:
            # 親ボーンがある場合、遡る
            matrix = self.get_mesh_matrix(matrixes, bone.parent_index, matrix)

        return matrix


class Morphs(BaseIndexNameDictModel[Morph]):
    """
    モーフリスト
    """

    def __init__(self) -> None:
        super().__init__()

    def filter_by_type(self, *keys: MorphType) -> list[Morph]:
        return [v for v in self.data.values() if v.morph_type in keys]

    def writable(self) -> list[Morph]:
        """出力対象となるモーフ一覧を取得する"""
        morphs: list[Morph] = []
        for m in self:
            if m.is_system:
                continue
            morphs.append(m)
        return morphs


class DisplaySlots(BaseIndexNameDictModel[DisplaySlot]):
    """
    表示枠リスト
    """

    def __init__(self):
        super().__init__()


class RigidBodies(BaseIndexNameDictModel[RigidBody]):
    """
    剛体リスト
    """

    def __init__(self) -> None:
        super().__init__()

    def get_by_index(self, index: int) -> RigidBody:
        if index == -1:
            return RigidBody()
        return super().get_by_index(index)


class Joints(BaseIndexNameDictModel[Joint]):
    """
    ジョイントリスト
    """

    def __init__(self) -> None:
        super().__init__()

    def get_by_index(self, index: int) -> Joint:
        if index == -1:
            return Joint()
        return super().get_by_index(index)


class PmxModel(BaseHashModel):
    """
    Pmxモデルデータ

    Parameters
    ----------
    path : str, optional
        パス, by default ""
    signature : str, optional
        signature, by default ""
    version : float, optional
        バージョン, by default 0.0
    extended_uv_count : int, optional
        追加UV数, by default 0
    vertex_count : int, optional
        頂点数, by default 0
    texture_count : int, optional
        テクスチャ数, by default 0
    material_count : int, optional
        材質数, by default 0
    bone_count : int, optional
        ボーン数, by default 0
    morph_count : int, optional
        モーフ数, by default 0
    rigidbody_count : int, optional
        剛体数, by default 0
    name : str, optional
        モデル名, by default ""
    english_name : str, optional
        モデル名英, by default ""
    comment : str, optional
        コメント, by default ""
    english_comment : str, optional
        コメント英, by default ""
    json_data : dict, optional
        JSONデータ（vroidデータ用）, by default {}
    """

    __slots__ = (
        "path",
        "digest",
        "signature",
        "version",
        "extended_uv_count",
        "vertex_count",
        "texture_count",
        "material_count",
        "bone_count",
        "morph_count",
        "rigidbody_count",
        "model_name",
        "english_name",
        "comment",
        "english_comment",
        "vertices",
        "faces",
        "textures" "toon_textures",
        "materials",
        "bones",
        "bone_trees",
        "morphs",
        "display_slots",
        "rigidbodies",
        "joints",
        "for_draw",
        "meshes",
        "textures",
        "toon_textures",
    )

    def __init__(
        self,
        path: str = "",
    ):
        super().__init__(path=path or "")
        self.signature: str = ""
        self.version: float = 0.0
        self.extended_uv_count: int = 0
        self.vertex_count: int = 0
        self.texture_count: int = 0
        self.material_count: int = 0
        self.bone_count: int = 0
        self.morph_count: int = 0
        self.rigidbody_count: int = 0
        self.model_name: str = ""
        self.english_name: str = ""
        self.comment: str = ""
        self.english_comment: str = ""
        self.vertices: Vertices = Vertices()
        self.faces: Faces = Faces()
        self.textures: Textures = Textures()
        self.toon_textures: ToonTextures = ToonTextures()
        self.materials: Materials = Materials()
        self.bones: Bones = Bones()
        self.bone_trees: BoneTrees = BoneTrees()
        self.morphs: Morphs = Morphs()
        self.display_slots: DisplaySlots = DisplaySlots()
        self.rigidbodies: RigidBodies = RigidBodies()
        self.joints: Joints = Joints()
        self.for_draw = False
        self.meshes: Optional[Meshes] = None

    @property
    def name(self) -> str:
        return self.model_name

    def get_weighted_vertex_scale(self) -> dict[int, dict[int, MVector3D]]:
        vertex_bone_scales: dict[int, dict[int, MVector3D]] = {}
        total_index_count = len(self.vertices)
        for vertex in self.vertices:
            indexes = vertex.deform.get_indexes()
            weights = vertex.deform.get_weights()
            for bone_index in indexes:
                if bone_index not in vertex_bone_scales:
                    vertex_bone_scales[bone_index] = {}
                vertex_bone_scales[bone_index][vertex.index] = MVector3D(*(vertex.normal.vector * weights[indexes == bone_index][0]))

                logger.count(
                    "ウェイトボーン分布",
                    index=vertex.index,
                    total_index_count=total_index_count,
                    display_block=5000,
                )
        return vertex_bone_scales

    def get_vertices_by_bone(self) -> dict[int, list[int]]:
        """ボーン別頂点INDEXリスト+ウェイトの取得"""
        vertices_bones: dict[int, list[int]] = {}
        total_index_count = len(self.vertices)
        for vertex in self.vertices:
            for bone_index in vertex.deform.get_indexes():
                if bone_index not in vertices_bones:
                    vertices_bones[bone_index] = []
                vertices_bones[bone_index].append(vertex.index)

                logger.count(
                    "ウェイトボーン分布",
                    index=vertex.index,
                    total_index_count=total_index_count,
                    display_block=5000,
                )
        return vertices_bones

    def get_vertices_by_material(self) -> dict[int, list[int]]:
        """材質別頂点INDEXリストの取得"""
        prev_face_count = 0
        vertices_by_materials: dict[int, list[int]] = {}
        for material in self.materials:
            vertices: list[int] = []
            face_count = material.vertices_count // 3
            for face_index in range(prev_face_count, prev_face_count + face_count):
                vertices.extend(self.faces[face_index].vertices)
            vertices_by_materials[material.index] = list(set(vertices))
            prev_face_count += face_count
        return vertices_by_materials

    def init_draw(self, shader: MShader):
        if self.for_draw:
            # 既にフラグが立ってたら描画初期化済み
            return

        # 共有Toon読み込み
        for tidx, tpath in enumerate(
            glob(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..",
                    "resources",
                    "share_toon",
                    "*.bmp",
                )
            )
        ):
            self.toon_textures.append(Texture(tidx, os.path.abspath(tpath)))

        self.meshes = Meshes(shader, self)
        # 描画初期化
        self.for_draw = True

    def delete_draw(self):
        if not self.for_draw or not self.meshes:
            # 描画初期化してなければスルー
            return
        self.meshes.delete_draw()
        self.for_draw = False

    def draw(
        self,
        bone_matrixes: np.ndarray,
        vertex_morph_poses: np.ndarray,
        uv_morph_poses: np.ndarray,
        uv1_morph_poses: np.ndarray,
        material_morphs: list[ShaderMaterial],
        is_alpha: bool,
    ):
        if not self.for_draw or not self.meshes:
            return
        self.meshes.draw(bone_matrixes, vertex_morph_poses, uv_morph_poses, uv1_morph_poses, material_morphs, is_alpha)

    def draw_bone(
        self,
        bone_matrixes: np.ndarray,
        bone_color: np.ndarray,
    ):
        if not self.for_draw or not self.meshes:
            return
        self.meshes.draw_bone(bone_matrixes, bone_color)

    def setup(self) -> None:
        total_index_count = len(self.bones)

        for bone in self.bones:
            # IKのリンクとターゲット
            if bone.is_ik and bone.ik:
                # IKボーンの場合
                for link in bone.ik.links:
                    if link.bone_index in self.bones:
                        # リンクボーンにフラグを立てる
                        self.bones[link.bone_index].ik_link_indexes.append(bone.index)
                if bone.ik.bone_index in self.bones:
                    # ターゲットボーンにもフラグを立てる
                    self.bones[bone.ik.bone_index].ik_target_indexes.append(bone.index)

            bone.parent_relative_position = self.bones.get_parent_relative_position(bone.index)
            bone.tail_relative_position = self.bones.get_tail_relative_position(bone.index)
            # 各ボーンのローカル軸
            bone.local_axis = bone.tail_relative_position.normalized()

            # 逆オフセット行列は親ボーンからの相対位置分を戻す
            bone.parent_revert_matrix = MMatrix4x4()
            bone.parent_revert_matrix.translate(bone.parent_relative_position)

            # オフセット行列は自身の位置を原点に戻す行列
            bone.offset_matrix = MMatrix4x4()
            bone.offset_matrix.translate(-bone.position)

            logger.count(
                "モデルセットアップ：ボーン",
                index=bone.index,
                total_index_count=total_index_count,
                display_block=500,
            )

        # システム用ボーン追加
        if "右腕" in self.bones and "左腕" in self.bones and "上半身" in self.bones:
            neck_root_bone = Bone(name="首根元", index=len(self.bones))
            if "上半身2" in self.bones:
                neck_root_bone.parent_index = self.bones["上半身2"].index
            else:
                neck_root_bone.parent_index = self.bones["上半身"].index
            neck_root_bone.position = (self.bones["右腕"].position + self.bones["左腕"].position) / 2
            neck_root_bone.is_system = True
            self.bones.append(neck_root_bone)

        if "右足" in self.bones and "左足" in self.bones and "下半身" in self.bones:
            leg_root_bone = Bone(name="足中心", index=len(self.bones))
            leg_root_bone.parent_index = self.bones["下半身"].index
            leg_root_bone.position = (self.bones["右足"].position + self.bones["左足"].position) / 2
            leg_root_bone.is_system = True
            self.bones.append(leg_root_bone)

        logger.info("-- モデルセットアップ：システム用ボーン")

        # ボーンツリー生成
        self.bone_trees = self.bones.create_bone_trees()

        # 距離が離れている親ボーンINDEXの取得
        for bone_tree in self.bone_trees:
            last_bone = self.bones[bone_tree.last_name]
            for bone_name in reversed(bone_tree.names[:-1]):
                if np.isclose([last_bone.position.distance(self.bones[bone_name].position)], 0, atol=0.01, rtol=0.01).any():
                    # 同じ位置のはスルー
                    continue
                last_bone.far_parent_index = self.bones[bone_name].index
                break

        logger.info("-- モデルセットアップ：ボーンツリー")

    def insert_bone(self, bone: Bone):
        """ボーンの追加に伴う諸々のボーンINDEXの置き換え"""
        replaced_map = self.bones.insert(bone)

        for v in self.vertices:
            for before_bone_index, after_bone_index in replaced_map.items():
                v.deform.indexes = np.where(v.deform.indexes == before_bone_index, after_bone_index, v.deform.indexes)

        for b in self.bones:
            if b.parent_index in replaced_map:
                b.parent_index = replaced_map[b.parent_index]
            if b.tail_index in replaced_map:
                b.tail_index = replaced_map[b.tail_index]
            if b.effect_index in replaced_map:
                b.effect_index = replaced_map[b.effect_index]
            if b.is_ik:
                if b.ik.bone_index in replaced_map:
                    b.ik.bone_index = replaced_map[b.ik.bone_index]
                for link in b.ik.links:
                    if link.bone_index in replaced_map:
                        link.bone_index = replaced_map[link.bone_index]

        for m in self.morphs:
            if m.morph_type == MorphType.BONE:
                for offset in m.offsets:
                    bone_offset: BoneMorphOffset = offset
                    if bone_offset.bone_index in replaced_map:
                        bone_offset.bone_index = replaced_map[bone_offset.bone_index]

        for d in self.display_slots:
            for r in d.references:
                if r.display_type == DisplayType.BONE:
                    if r.display_index in replaced_map:
                        r.display_index = replaced_map[r.display_index]

        for r in self.rigidbodies:
            if r.bone_index in replaced_map:
                r.bone_index = replaced_map[r.bone_index]


class Meshes(BaseIndexDictModel[Mesh]):
    """
    メッシュリスト
    """

    __slots__ = (
        "data",
        "indexes",
        "_iter_index",
        "_size",
        "shader",
        "model",
        "vertices",
        "faces",
        "vao",
        "vbo_components",
        "morph_pos_comps",
        "morph_uv_comps",
        "morph_uv1_comps",
        "vbo_vertices",
        "ibo_faces",
        "bones",
        "bone_hierarchies",
        "bone_vao",
        "bone_vbo_components",
        "bone_vbo_vertices",
        "bone_ibo_faces",
    )

    def __init__(self, shader: MShader, model: PmxModel) -> None:
        super().__init__()

        self.shader = shader
        self.model = model

        # 頂点情報
        self.vertices = np.array(
            [
                np.fromiter(
                    [
                        *v.position.gl.vector,
                        *v.normal.gl.vector,
                        v.uv.x,
                        1 - v.uv.y,
                        v.extended_uvs[0].x if 0 < len(v.extended_uvs) else 0.0,
                        1 - v.extended_uvs[0].y if 0 < len(v.extended_uvs) else 0.0,
                        v.edge_factor,
                        *v.deform.normalized_deform,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    dtype=np.float32,
                    count=30,
                )
                for v in model.vertices
            ],
        )

        face_dtype: type = np.uint8 if model.vertex_count == 1 else np.uint16 if model.vertex_count == 2 else np.uint32

        # 面情報
        self.faces: np.ndarray = np.array(
            [
                np.fromiter(
                    [f.vertices[2], f.vertices[1], f.vertices[0]],
                    dtype=face_dtype,
                    count=3,
                )
                for f in model.faces
            ],
        )

        prev_vertices_count = 0
        for material in model.materials:
            texture: Optional[Texture] = None
            if 0 <= material.texture_index:
                texture = model.textures[material.texture_index]
                texture.init_draw(model.path, TextureType.TEXTURE)

            toon_texture: Optional[Texture] = None
            if ToonSharing.SHARING == material.toon_sharing_flg:
                # 共有Toon
                toon_texture = model.toon_textures[material.toon_texture_index]
                toon_texture.init_draw(model.path, TextureType.TOON, is_individual=False)
            elif ToonSharing.INDIVIDUAL == material.toon_sharing_flg and 0 <= material.toon_texture_index:
                # 個別Toon
                toon_texture = model.textures[material.toon_texture_index]
                toon_texture.init_draw(model.path, TextureType.TOON)

            sphere_texture: Optional[Texture] = None
            if 0 <= material.sphere_texture_index:
                sphere_texture = model.textures[material.sphere_texture_index]
                sphere_texture.init_draw(model.path, TextureType.SPHERE)

            self.append(
                Mesh(
                    material,
                    texture,
                    toon_texture,
                    sphere_texture,
                    prev_vertices_count,
                    face_dtype,
                )
            )

            prev_vertices_count += material.vertices_count

        # ---------------------

        # 頂点VAO
        self.vao = VAO()
        self.vbo_components = {
            VsLayout.POSITION_ID.value: {"size": 3, "offset": 0},
            VsLayout.NORMAL_ID.value: {"size": 3, "offset": 3},
            VsLayout.UV_ID.value: {"size": 2, "offset": 6},
            VsLayout.EXTEND_UV_ID.value: {"size": 2, "offset": 8},
            VsLayout.EDGE_ID.value: {"size": 1, "offset": 10},
            VsLayout.BONE_ID.value: {"size": 4, "offset": 11},
            VsLayout.WEIGHT_ID.value: {"size": 4, "offset": 15},
            VsLayout.MORPH_POS_ID.value: {"size": 3, "offset": 19},
            VsLayout.MORPH_UV_ID.value: {"size": 4, "offset": 22},
            VsLayout.MORPH_UV1_ID.value: {"size": 4, "offset": 26},
        }
        self.morph_pos_comps = self.vbo_components[VsLayout.MORPH_POS_ID.value]
        self.morph_uv_comps = self.vbo_components[VsLayout.MORPH_UV_ID.value]
        self.morph_uv1_comps = self.vbo_components[VsLayout.MORPH_UV1_ID.value]
        self.vbo_vertices = VBO(
            self.vertices,
            self.vbo_components,
        )
        self.ibo_faces = IBO(self.faces)

        # ----------

        # ボーン位置
        self.bones = np.array(
            [
                np.fromiter(
                    [
                        *b.position.gl.vector,
                        b.index / len(model.bones),
                    ],
                    dtype=np.float32,
                    count=4,
                )
                for b in model.bones
            ],
        )

        bone_face_dtype: type = np.uint8 if 256 > len(model.bones) else np.uint16 if 65536 > len(model.bones) else np.uint32

        # ボーン親子関係
        self.bone_hierarchies: np.ndarray = np.array(
            [
                np.fromiter(
                    [
                        b.index,
                        b.parent_index,
                    ],
                    dtype=bone_face_dtype,
                    count=2,
                )
                for b in model.bones
                if 0 <= b.parent_index
            ],
        )

        # ボーンVAO
        self.bone_vao = VAO()
        self.bone_vbo_components = {
            0: {"size": 3, "offset": 0},
            1: {"size": 1, "offset": 3},
        }
        self.bone_vbo_vertices = VBO(
            self.bones,
            self.bone_vbo_components,
        )
        self.bone_ibo_faces = IBO(self.bone_hierarchies)

    def draw(
        self,
        bone_matrixes: np.ndarray,
        vertex_morph_poses: np.ndarray,
        uv_morph_poses: np.ndarray,
        uv1_morph_poses: np.ndarray,
        material_morphs: list[ShaderMaterial],
        is_alpha: bool,
    ):
        # 隠面消去
        # https://learnopengl.com/Advanced-OpenGL/Depth-testing
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)

        # 頂点モーフ変動量を上書き設定してからバインド
        self.vbo_vertices.data[:, self.morph_pos_comps["offset"] : (self.morph_pos_comps["offset"] + self.morph_pos_comps["size"])] = vertex_morph_poses
        self.vbo_vertices.data[:, self.morph_uv_comps["offset"] : (self.morph_uv_comps["offset"] + self.morph_uv_comps["size"])] = uv_morph_poses
        self.vbo_vertices.data[:, self.morph_uv1_comps["offset"] : (self.morph_uv1_comps["offset"] + self.morph_uv1_comps["size"])] = uv1_morph_poses

        for mesh in self:
            self.vao.bind()
            self.vbo_vertices.bind()
            self.vbo_vertices.set_slot(VsLayout.POSITION_ID)
            self.vbo_vertices.set_slot(VsLayout.NORMAL_ID)
            self.vbo_vertices.set_slot(VsLayout.UV_ID)
            self.vbo_vertices.set_slot(VsLayout.EXTEND_UV_ID)
            self.vbo_vertices.set_slot(VsLayout.EDGE_ID)
            self.vbo_vertices.set_slot(VsLayout.BONE_ID)
            self.vbo_vertices.set_slot(VsLayout.WEIGHT_ID)
            self.vbo_vertices.set_slot(VsLayout.MORPH_POS_ID)
            self.vbo_vertices.set_slot(VsLayout.MORPH_UV_ID)
            self.vbo_vertices.set_slot(VsLayout.MORPH_UV1_ID)
            self.ibo_faces.bind()

            material_morph = material_morphs[mesh.material.index]

            if 0.0 >= material_morph.material.diffuse.w:
                # 非表示材質の場合、常に描写しない
                continue

            if is_alpha and 1.0 <= material_morph.material.diffuse.w:
                # 半透明描写かつ非透過度が1.0以上の場合、スルー
                continue
            elif not is_alpha and 1.0 > material_morph.material.diffuse.w:
                # 不透明描写かつ非透過度が1.0未満の場合スルー
                continue

            # アルファテストを有効にする
            gl.glEnable(gl.GL_ALPHA_TEST)

            # ブレンディングを有効にする
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

            # モデル描画
            self.shader.use(ProgramType.MODEL)
            mesh.draw_model(bone_matrixes, material_morph, self.shader, self.ibo_faces)
            self.shader.unuse()

            if DrawFlg.DRAWING_EDGE in mesh.material.draw_flg and 0 < material_morph.material.diffuse.w:
                # エッジ描画
                self.shader.use(ProgramType.EDGE)
                mesh.draw_edge(bone_matrixes, material_morph, self.shader, self.ibo_faces)
                self.shader.unuse()

            # ---------------

            self.ibo_faces.unbind()
            self.vbo_vertices.unbind()
            self.vao.unbind()

            gl.glDisable(gl.GL_BLEND)
            gl.glDisable(gl.GL_ALPHA_TEST)

        gl.glDisable(gl.GL_DEPTH_TEST)

    def draw_bone(self, bone_matrixes: np.ndarray, bone_color: np.ndarray):
        # ボーンをモデルメッシュの前面に描画するために深度テストを無効化
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_ALWAYS)

        # アルファテストを有効にする
        gl.glEnable(gl.GL_ALPHA_TEST)

        # ブレンディングを有効にする
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.bone_vao.bind()
        self.bone_vbo_vertices.bind()
        self.bone_vbo_vertices.set_slot_by_value(0)
        self.bone_vbo_vertices.set_slot_by_value(1)
        self.bone_ibo_faces.bind()

        self.shader.use(ProgramType.BONE)

        gl.glUniform4f(self.shader.edge_color_uniform[ProgramType.BONE.value], *bone_color)
        gl.glUniform1i(self.shader.bone_count_uniform[ProgramType.BONE.value], len(self.model.bones))

        self.model.meshes[0].bind_bone_matrixes(bone_matrixes, self.shader, ProgramType.BONE)

        try:
            gl.glDrawElements(
                gl.GL_LINES,
                self.bone_hierarchies.size,
                self.bone_ibo_faces.dtype,
                gl.ctypes.c_void_p(0),
            )
        except Exception as e:
            raise MViewerException("Meshes draw_bone Failure", e)

        error_code = gl.glGetError()
        if error_code != gl.GL_NO_ERROR:
            raise MViewerException(f"Meshes draw_bone Failure\n{error_code}")

        self.model.meshes[0].unbind_bone_matrixes()

        self.bone_ibo_faces.unbind()
        self.bone_vbo_vertices.unbind()
        self.bone_vao.unbind()
        self.shader.unuse()

        gl.glDisable(gl.GL_BLEND)
        gl.glDisable(gl.GL_ALPHA_TEST)
        gl.glDisable(gl.GL_DEPTH_TEST)

    def delete_draw(self):
        for material in self.model.materials:
            texture: Optional[Texture] = None
            if 0 <= material.texture_index:
                texture = self.model.textures[material.texture_index]
                texture.delete_draw()

            toon_texture: Optional[Texture] = None
            if ToonSharing.SHARING == material.toon_sharing_flg:
                # 共有Toon
                toon_texture = self.model.toon_textures[material.toon_texture_index]
                toon_texture.delete_draw()
            elif ToonSharing.INDIVIDUAL == material.toon_sharing_flg and 0 <= material.toon_texture_index:
                # 個別Toon
                toon_texture = self.model.textures[material.toon_texture_index]
                toon_texture.delete_draw()

            sphere_texture: Optional[Texture] = None
            if 0 <= material.sphere_texture_index:
                sphere_texture = self.model.textures[material.sphere_texture_index]
                sphere_texture.delete_draw()
        del self.vao
        del self.vbo_components
        del self.morph_pos_comps
        del self.morph_uv_comps
        del self.morph_uv1_comps
        del self.vbo_vertices
        del self.ibo_faces
