import os
from glob import glob
from typing import Optional

import numpy as np

from mlib.base.collection import BaseHashModel, BaseIndexDictModel, BaseIndexNameDictModel, BaseIndexNameDictWrapperModel
from mlib.base.math import MMatrix4x4, MMatrix4x4List, MVector3D
from mlib.pmx.mesh import IBO, VAO, VBO, Mesh
from mlib.pmx.pmx_part import Bone, DisplaySlot, DrawFlg, Face, Joint, Material, Morph, RigidBody, Texture, TextureType, ToonSharing, Vertex
from mlib.pmx.shader import MShader, VsLayout


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


class BoneTrees(BaseIndexNameDictWrapperModel[BoneTree]):
    """
    BoneTreeリスト
    """

    __slots__ = ["data", "_names", "_indexes", "_iter_index"]

    def __init__(self) -> None:
        """モデル辞書"""
        super().__init__()

    def create(self, key: str) -> BoneTree:
        return BoneTree(key)


class Bones(BaseIndexNameDictModel[Bone]):
    """
    ボーンリスト
    """

    def __init__(self) -> None:
        super().__init__()

    def create_bone_links(self) -> BoneTrees:
        """
        ボーンツリー一括生成

        Returns
        -------
        BoneTrees
        """
        # ボーンツリー
        bone_trees = BoneTrees()

        # 計算ボーンリスト
        for end_bone in self:
            # レイヤー込みのINDEXリスト取得を末端ボーンをキーとして保持
            bone_tree = BoneTree(name=end_bone.name)
            for ti, (_, bidx) in enumerate(sorted(self.create_bone_link_indexes(end_bone.index))):
                bone_tree.append(self[bidx].copy())
            bone_trees.append(bone_tree)

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
            bone_link_indexes = [(self[child_idx].layer, self[child_idx].index)]

        for b in reversed(self):
            if b.index == self[child_idx].parent_index:
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
        if bone.is_tail_bone and bone.tail_index >= 0 and bone.tail_index in self:
            # 表示先が指定されているの場合、保持
            to_pos = self[bone.tail_index].position
        elif not bone.is_tail_bone:
            # 表示先が相対パスの場合、保持
            to_pos = from_pos + bone.tail_position
        else:
            # 表示先がない場合、とりあえず親ボーンからの向きにする
            from_pos = self[bone.parent_index].position
            to_pos = self[bone_index].position

        return to_pos

    def get_parent_relative_position(self, bone_index: int) -> MVector3D:
        """親ボーンから見た相対位置"""
        bone = self[bone_index]
        return bone.position - (MVector3D() if bone.index < 0 or bone.parent_index not in self else self[bone.parent_index].position)

    def get_mesh_gl_matrix(self, matrixes: MMatrix4x4List, bone_index: int, matrix: np.ndarray) -> np.ndarray:
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

        if bone.index >= 0 and bone.parent_index in self:
            # 親ボーンがある場合、遡る
            matrix = self.get_mesh_gl_matrix(matrixes, bone.parent_index, matrix)

        # 自身の姿勢をかける
        # 逆BOf行列(初期姿勢行列)
        matrix = matrix @ bone.init_matrix.vector
        # 座標変換行列
        matrix = matrix @ matrixes.vector[0, bone_index]

        return matrix


class Morphs(BaseIndexNameDictModel[Morph]):
    """
    モーフリスト
    """

    def __init__(self) -> None:
        super().__init__()


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


class Joints(BaseIndexNameDictModel[Joint]):
    """
    ジョイントリスト
    """

    def __init__(self) -> None:
        super().__init__()


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
        self.json_data: dict = {}
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

    def init_draw(self, shader: MShader):
        if self.for_draw:
            # 既にフラグが立ってたら描画初期化済み
            return

        # 描画初期化
        self.for_draw = True
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

    def draw(self, mats: np.ndarray):
        if not self.for_draw or not self.meshes:
            return
        self.meshes.draw(mats)

    def setup(self) -> None:
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
            bone.init_matrix = MMatrix4x4()
            bone.init_matrix.translate(bone.parent_relative_position.gl)

            # オフセット行列は自身の位置を原点に戻す行列
            offset_mat = MMatrix4x4()
            offset_mat.translate(bone.position.gl)
            bone.offset_matrix = offset_mat.inverse()

        # ボーンツリー生成
        self.bone_trees = self.bones.create_bone_links()


class Meshes(BaseIndexDictModel[Mesh]):
    """
    メッシュリスト
    """

    def __init__(self, shader: MShader, model: PmxModel) -> None:
        super().__init__()

        self.shader = shader
        self.model = model

        # 頂点情報
        self.vertices = np.array(
            [
                np.array(
                    [
                        -v.position.x,
                        v.position.y,
                        v.position.z,
                        -v.normal.x,
                        v.normal.y,
                        v.normal.z,
                        v.uv.x,
                        1 - v.uv.y,
                        *(v.extended_uvs[0].vector if len(v.extended_uvs) > 0 else [0, 0]),
                        v.edge_factor,
                        *v.deform.normalized_deform(),
                    ],
                    dtype=np.float32,
                )
                for v in model.vertices
            ],
            dtype=np.float32,
        )

        face_dtype: type = np.uint8 if model.vertex_count == 1 else np.uint16 if model.vertex_count == 2 else np.uint32

        # 面情報
        self.faces: np.ndarray = np.array(
            [
                np.array(
                    [f.vertices[2], f.vertices[1], f.vertices[0]],
                    dtype=face_dtype,
                )
                for f in model.faces
            ],
            dtype=face_dtype,
        )

        prev_vertices_count = 0
        for material in model.materials:
            texture: Optional[Texture] = None
            if material.texture_index >= 0:
                texture = model.textures[material.texture_index]
                texture.init_draw(model.path, TextureType.TEXTURE)

            toon_texture: Optional[Texture] = None
            if ToonSharing.SHARING == material.toon_sharing_flg:
                # 共有Toon
                toon_texture = model.toon_textures[material.toon_texture_index]
                toon_texture.init_draw(model.path, TextureType.TOON, is_individual=False)
            elif ToonSharing.INDIVIDUAL == material.toon_sharing_flg and material.toon_texture_index >= 0:
                # 個別Toon
                toon_texture = model.textures[material.toon_texture_index]
                toon_texture.init_draw(model.path, TextureType.TOON)

            sphere_texture: Optional[Texture] = None
            if material.sphere_texture_index >= 0:
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

        self.vao = VAO()
        self.vbo_vertices = VBO(
            self.vertices,
            {
                VsLayout.POSITION_ID.value: {"size": 3, "offset": 0},
                VsLayout.NORMAL_ID.value: {"size": 3, "offset": 3},
                VsLayout.UV_ID.value: {"size": 2, "offset": 6},
                VsLayout.EXTEND_UV_ID.value: {"size": 2, "offset": 8},
                VsLayout.EDGE_ID.value: {"size": 1, "offset": 10},
                VsLayout.BONE_ID.value: {"size": 4, "offset": 11},
                VsLayout.WEIGHT_ID.value: {"size": 4, "offset": 15},
            },
        )

        self.ibo_faces = IBO(self.faces)

    def draw(self, mats: np.ndarray):
        for mesh in self:
            self.vao.bind()
            self.vbo_vertices.set_slot(VsLayout.POSITION_ID)
            self.vbo_vertices.set_slot(VsLayout.NORMAL_ID)
            self.vbo_vertices.set_slot(VsLayout.UV_ID)
            self.vbo_vertices.set_slot(VsLayout.EXTEND_UV_ID)
            self.vbo_vertices.set_slot(VsLayout.EDGE_ID)
            self.vbo_vertices.set_slot(VsLayout.BONE_ID)
            self.vbo_vertices.set_slot(VsLayout.WEIGHT_ID)
            self.ibo_faces.bind()

            # モデル描画
            self.shader.use()
            mesh.draw_model(mats, self.shader, self.ibo_faces)
            self.shader.unuse()

            if DrawFlg.DRAWING_EDGE in mesh.material.draw_flg and mesh.material.diffuse_color.w > 0:
                # エッジ描画
                self.shader.use(edge=True)
                mesh.draw_edge(mats, self.shader, self.ibo_faces)
                self.shader.unuse()

            # ---------------

            self.ibo_faces.unbind()
            self.vbo_vertices.unbind()
            self.vao.unbind()
