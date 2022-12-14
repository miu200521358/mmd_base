import os
from glob import glob
from typing import Any, Optional

import numpy as np

from mlib.base.collection import (
    BaseHashModel,
    BaseIndexDictModel,
    BaseIndexListModel,
    BaseIndexNameListModel,
)
from mlib.base.math import MVector3D
from mlib.pmx.mesh import IBO, VAO, VBO, Mesh
from mlib.pmx.part import (
    Bone,
    DisplaySlot,
    DrawFlg,
    Face,
    Joint,
    Material,
    Morph,
    RigidBody,
    Texture,
    TextureType,
    ToonSharing,
    Vertex,
)
from mlib.pmx.shader import MShader, VsLayout


class Vertices(BaseIndexListModel[Vertex]):
    """
    頂点リスト
    """

    def __init__(self):
        super().__init__()


class Faces(BaseIndexListModel[Face]):
    """
    面リスト
    """

    def __init__(self):
        super().__init__()


class Textures(BaseIndexListModel[Texture]):
    """
    テクスチャリスト
    """

    def __init__(self):
        super().__init__()


class ToonTextures(BaseIndexDictModel[Texture]):
    """
    共有テクスチャ辞書
    """

    def __init__(self):
        super().__init__()


class Materials(BaseIndexNameListModel[Material]):
    """
    材質リスト
    """

    def __init__(self):
        super().__init__()


class BoneTree(BaseIndexDictModel[Bone]):
    """ボーンリンク"""

    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, key: Any) -> Bone:
        if isinstance(key, int) and key < 0:
            # マイナス指定の場合、後ろからの順番に置き換える
            return super().__getitem__(len(self.data) + key)
        return super().__getitem__(key)

    def last_index(self) -> int:
        return len(self.data) - 1

    def last_name(self) -> str:
        return self.data[self.last_index()].name

    def get_relative_position(self, key: Any) -> MVector3D:
        """
        該当ボーンの相対位置を取得

        Parameters
        ----------
        bone_name : int
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


class BoneTrees:
    """
    BoneTreeリスト
    """

    __slots__ = ["data", "__names", "__indices", "__iter_index"]

    def __init__(self):
        super().__init__()
        self.data: dict[int, BoneTree] = {}
        self.__names: dict[str, int] = {}
        self.__indices: list[int] = []
        self.__iter_index = 0

    def __getitem__(self, key: Any) -> BoneTree:
        if isinstance(key, int):
            return self.get_by_index(key)
        else:
            return self.get_by_name(key)

    def __setitem__(self, index: int, bt: BoneTree):
        self.data[index] = bt
        if bt.data[bt.last_index()].name not in self.__names:
            # 名前は先勝ちで保持
            self.__names[bt.data[bt.last_index()].name] = bt.data[bt.last_index()].index

    def names(self) -> dict[str, int]:
        return dict(
            [
                (bt.data[bt.last_index()].name, bt.data[bt.last_index()].index)
                for bt in self.data.values()
            ]
        )

    def gets(self, bone_names: list[str]) -> list[BoneTree]:
        """
        指定したボーン名のみを抽出する

        Parameters
        ----------
        bone_names : list[str]
            抽出対象ボーン名リスト

        Returns
        -------
        抽出ボーンツリーリスト
        """
        new_trees = []
        for bname in bone_names:
            bt = self[bname]
            new_trees.append(bt)
        return new_trees

    def get_by_index(self, index: int) -> BoneTree:
        """
        リストから要素を取得する

        Parameters
        ----------
        index : int
            インデックス番号

        Returns
        -------
        TBaseIndexNameModel
            要素
        """
        if index >= len(self.data):
            raise KeyError(f"Not Found: {index}")
        return self.data[index]

    def get_by_name(self, name: str) -> BoneTree:
        """
        リストから要素を取得する

        Parameters
        ----------
        name : str
            名前

        Returns
        -------
        TBaseIndexNameModel
            要素
        """
        if name not in self.__names:
            raise KeyError(f"Not Found: {name}")
        return self.data[self.names()[name]]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        self.__iter_index = -1
        self.__indices = sorted(list(self.data.keys()))
        return self

    def __next__(self) -> BoneTree:
        self.__iter_index += 1
        if self.__iter_index >= len(self.__indices):
            raise StopIteration
        return self.data[self.__indices[self.__iter_index]]

    def __contains__(self, v) -> bool:
        return v in self.__names or v in self.data.keys()


class Bones(BaseIndexNameListModel[Bone]):
    """
    ボーンリスト
    """

    def __init__(self):
        super().__init__()

    def append(self, v: Bone) -> None:
        super().append(v)
        if v.is_ik() and v.ik:
            # IKボーンの場合
            for link in v.ik.links:
                if link.bone_index in self:
                    # リンクボーンにフラグを立てる
                    self[link.bone_index].ik_link_indices.append(v.index)
            if v.ik.bone_index in self:
                # ターゲットボーンにもフラグを立てる
                self[v.ik.bone_index].ik_target_indices.append(v.index)

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
            bone_tree = BoneTree()
            for ti, (_, bidx) in enumerate(
                sorted(self.create_bone_link_indexes(end_bone.index))
            ):
                bone_tree[ti] = self[bidx].copy()
            bone_tree[len(bone_tree)] = end_bone.copy()
            bone_trees[end_bone.index] = bone_tree

        return bone_trees

    def create_bone_link_indexes(
        self, child_idx: int, bone_link_indexes=None
    ) -> list[tuple[int, int]]:
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
            bone_link_indexes = []

        for b in reversed(self.data):
            if b.index == self[child_idx].parent_index:
                bone_link_indexes.append((b.layer, b.index))
                return self.create_bone_link_indexes(b.index, bone_link_indexes)

        return bone_link_indexes

    def get_tail_position(self, bone_index: int) -> MVector3D:
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
        if bone.is_tail_bone() and bone.tail_position != MVector3D():
            # 表示先が相対パスの場合、保持
            to_pos = from_pos + bone.tail_position
        elif bone.is_tail_bone() and bone.tail_index >= 0 and bone.tail_index in self:
            # 表示先が指定されているの場合、保持
            to_pos = self[bone.tail_index].position
        else:
            # 表示先がない場合、とりあえず親ボーンからの向きにする
            from_pos = self[bone.parent_index].position
            to_pos = self[bone_index].position

        return to_pos

    def get_local_x_axis(self, bone_index: int) -> MVector3D:
        """
        ローカルX軸の取得

        Parameters
        ----------
        bone_index : int
            ボーンINDEX

        Returns
        -------
        ローカルX軸
        """
        if bone_index not in self:
            return MVector3D()

        bone = self[bone_index]

        if bone.is_ik() and ("足ＩＫ" in bone.name or "つま先ＩＫ" in bone.name):
            # 足IK系は固定
            return MVector3D(0, 1, 0)

        if bone.has_fixed_axis() and bone.fixed_axis:
            # 軸制限がある場合、親からの向きを保持
            return bone.fixed_axis.normalized()

        from_pos = self[bone.name].position
        to_pos = self.get_tail_position(bone_index)

        # 軸制限の指定が無い場合、子の方向
        x_axis = (to_pos - from_pos).normalized()

        return x_axis


class Morphs(BaseIndexNameListModel[Morph]):
    """
    モーフリスト
    """

    def __init__(self):
        super().__init__()


class DisplaySlots(BaseIndexNameListModel[DisplaySlot]):
    """
    表示枠リスト
    """

    def __init__(
        self,
    ):
        super().__init__()


class RigidBodies(BaseIndexNameListModel[RigidBody]):
    """
    剛体リスト
    """

    def __init__(self):
        super().__init__()


class Joints(BaseIndexNameListModel[Joint]):
    """
    ジョイントリスト
    """

    def __init__(self):
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
        path: str = None,
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
        self.name: str = ""
        self.english_name: str = ""
        self.comment: str = ""
        self.english_comment: str = ""
        self.json_data: dict = {}
        self.vertices = Vertices()
        self.faces = Faces()
        self.textures = Textures()
        self.toon_textures = ToonTextures()
        self.materials = Materials()
        self.bones = Bones()
        self.bone_trees = BoneTrees()
        self.morphs = Morphs()
        self.display_slots = DisplaySlots()
        self.rigidbodies = RigidBodies()
        self.joints = Joints()
        self.for_draw = False
        self.meshes = None

    def get_name(self) -> str:
        return self.name

    def init_draw(self, shader):
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
            self.toon_textures[tidx] = Texture(os.path.abspath(tpath))

        self.meshes = Meshes(shader, self)

    def update(self):
        if not self.for_draw:
            return
        self.meshes.update()

    def draw(self):
        if not self.for_draw:
            return
        self.meshes.draw()


class Meshes(BaseIndexListModel[Mesh]):
    """
    メッシュリスト
    """

    def __init__(self, shader: MShader, model: PmxModel):
        super().__init__()

        self.shader = shader

        # 頂点情報
        self.vertices = np.array(
            [
                np.array(
                    [
                        *v.position.vector,
                        *v.normal.vector,
                        v.uv.x,
                        1 - v.uv.y,
                        *(
                            v.extended_uvs[0].vector
                            if len(v.extended_uvs) > 0
                            else [0, 0]
                        ),
                        v.edge_factor,
                    ],
                    dtype=np.float32,
                )
                for v in model.vertices
            ],
            dtype=np.float32,
        )

        face_dtype: type = (
            np.uint8
            if model.vertex_count == 1
            else np.uint16
            if model.vertex_count == 2
            else np.uint32
        )

        # 面情報
        self.faces: np.ndarray = np.array(
            [
                np.array(
                    [f.vertices[2], f.vertices[1], f.vertices[0]], dtype=face_dtype
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
                toon_texture.init_draw(
                    model.path, TextureType.TOON, is_individual=False
                )
            elif (
                ToonSharing.INDIVIDUAL == material.toon_sharing_flg
                and material.toon_texture_index >= 0
            ):
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
            },
        )

        self.ibo_faces = IBO(self.faces)

    def update(self):
        pass

    def draw(self):
        for mesh in self.data:
            self.vao.bind()
            self.vbo_vertices.set_slot(VsLayout.POSITION_ID)
            self.vbo_vertices.set_slot(VsLayout.NORMAL_ID)
            self.vbo_vertices.set_slot(VsLayout.UV_ID)
            self.vbo_vertices.set_slot(VsLayout.EXTEND_UV_ID)
            self.vbo_vertices.set_slot(VsLayout.EDGE_ID)
            self.ibo_faces.bind()

            # FIXME MSAA https://blog.techlab-xe.net/opengl%E3%81%A7msaa/

            # モデル描画
            self.shader.use()
            mesh.draw_model(self.shader, self.ibo_faces)
            self.shader.unuse()

            if (
                DrawFlg.DRAWING_EDGE in mesh.material.draw_flg
                and mesh.material.diffuse_color.w > 0
            ):
                # エッジ描画
                self.shader.use(edge=True)
                mesh.draw_edge(self.shader, self.ibo_faces)
                self.shader.unuse()

            self.ibo_faces.unbind()
            self.vbo_vertices.unbind()
            self.vao.unbind()
