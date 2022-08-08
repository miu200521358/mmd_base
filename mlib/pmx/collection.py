from typing import Optional

from mlib.base.collection import (
    BaseHashModel,
    BaseIndexListModel,
    BaseIndexNameListModel,
)
from mlib.pmx.mesh import Mesh
from mlib.pmx.part import (
    Bone,
    DisplaySlot,
    Face,
    Joint,
    Material,
    Morph,
    RigidBody,
    Texture,
    Vertex,
)
from mlib.pmx.shader import MShader


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


class Materials(BaseIndexNameListModel[Material]):
    """
    材質リスト
    """

    def __init__(self):
        super().__init__()


class Bones(BaseIndexNameListModel[Bone]):
    """
    ボーンリスト
    """

    def __init__(self):
        super().__init__()

    def get_max_layer(self) -> int:
        """
        最大変形階層を取得

        Returns
        -------
        int
            最大変形階層
        """
        return max([b.layer for b in self.data])

    def get_bones_by_layer(self, layer: int) -> list[Bone]:
        return [b for b in self.data if b.layer == layer]


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


class Meshs(BaseIndexListModel[Mesh]):
    """
    メッシュリスト
    """

    def __init__(
        self,
        vertices: Vertices,
        faces: Faces,
        materials: Materials,
        textures: Textures,
    ):
        super().__init__()

        self.shader = MShader()
        material_face_count = 0
        vertex_poses: list[float] = []
        vertex_uvs: list[float] = []
        face_indexes: list[int] = []
        for material in materials:
            texture: Optional[Texture] = None
            if material.texture_index >= 0:
                texture = textures[material.texture_index]
            sphere_texture: Optional[Texture] = None
            if material.sphere_texture_index >= 0:
                sphere_texture = textures[material.sphere_texture_index]

            face_count = material.vertices_count // 3
            for face_index in range(face_count):
                face = faces[face_index + material_face_count]
                vertex_poses.extend(
                    vertices[vidx].position.vector for vidx in face.vertices
                )
                vertex_uvs.extend(vertices[vidx].uv.vector for vidx in face.vertices)
                face_indexes.extend(face.vertices)
            material_face_count += face_count

            self.data.append(
                Mesh(
                    self.shader,
                    vertex_poses,
                    vertex_uvs,
                    face_indexes,
                    material,
                    texture,
                    sphere_texture,
                )
            )

    def draw(self):
        for m in self.data:
            m.draw()


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
        self.materials = Materials()
        self.bones = Bones()
        self.morphs = Morphs()
        self.display_slots = DisplaySlots()
        self.rigidbodies = RigidBodies()
        self.joints = Joints()
        self.for_draw = False
        self.meshs = None

    def get_name(self) -> str:
        return self.name

    def init_draw(self):
        if self.for_draw:
            # 既にフラグが立ってたら描画初期化済み
            return

        # 描画初期化
        self.for_draw = True
        self.meshs = Meshs(self.vertices, self.faces, self.materials, self.textures)

    def draw(self):
        if not self.for_draw:
            return
        self.meshs.draw()
