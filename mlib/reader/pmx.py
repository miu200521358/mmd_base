from mlib.base import Encoding
from mlib.exception import MParseException
from mlib.math import MVector3D
from mlib.model.base import Switch
from mlib.model.pmx import (
    Bdef1,
    Bdef2,
    Bdef4,
    Bone,
    BoneFlg,
    BoneMorphOffset,
    DeformType,
    DisplaySlot,
    DisplaySlotReference,
    DisplayType,
    DrawFlg,
    GroupMorphOffset,
    Ik,
    IkLink,
    Joint,
    Material,
    MaterialMorphCalcMode,
    MaterialMorphOffset,
    Morph,
    MorphPanel,
    MorphType,
    PmxModel,
    RigidBody,
    RigidBodyCollisionGroup,
    RigidBodyMode,
    RigidBodyShape,
    Sdef,
    SphereMode,
    Surface,
    Texture,
    ToonSharing,
    UvMorphOffset,
    Vertex,
    VertexMorphOffset,
)
from mlib.reader.base import BaseReader


class PmxReader(BaseReader[PmxModel]):
    def __init__(self) -> None:
        super().__init__()

    def create_model(self, path: str) -> PmxModel:
        return PmxModel(path=path)

    def read_by_buffer_header(self, model: PmxModel):
        # pmx宣言
        model.signature = self.unpack("4s", 4)

        # pmxバージョン
        model.version = self.read_float()

        if model.signature[:3] != b"PMX" or f"{model.version:.1f}" not in [
            "2.0",
            "2.1",
        ]:
            # 整合性チェック
            raise MParseException(
                "PMX2.0/2.1形式外のデータです。signature: %s, version: %s ",
                [model.signature, model.version],
            )

        # 後続するデータ列のバイトサイズ  PMX2.0は 8 で固定
        _ = self.read_ubyte()

        # [0] - エンコード方式  | 0:UTF16 1:UTF8
        encode_type = self.read_ubyte()
        self.define_encoding(Encoding.UTF_8 if encode_type else Encoding.UTF_16_LE)

        # [1] - 追加UV数 	| 0～4 詳細は頂点参照
        model.extended_uv_count = self.read_ubyte()

        # [2] - 頂点Indexサイズ | 1,2,4 のいずれか
        model.vertex_count = self.read_ubyte()

        # [3] - テクスチャIndexサイズ | 1,2,4 のいずれか
        model.texture_count = self.read_ubyte()

        # [4] - 材質Indexサイズ | 1,2,4 のいずれか
        model.material_count = self.read_ubyte()

        # [5] - ボーンIndexサイズ | 1,2,4 のいずれか
        model.bone_count = self.read_ubyte()

        # [6] - モーフIndexサイズ | 1,2,4 のいずれか
        model.morph_count = self.read_ubyte()

        # [7] - 剛体Indexサイズ | 1,2,4 のいずれか
        model.rigidbody_count = self.read_ubyte()

        # モデル名（日本語）
        model.name = self.read_text()

    def read_by_buffer(self, model: PmxModel):

        # モデルの各要素サイズから読み取り処理を設定
        self.read_vertex_index = self.define_read_index(
            model.vertex_count, is_vertex=True
        )
        self.read_texture_index = self.define_read_index(model.texture_count)
        self.read_material_index = self.define_read_index(model.material_count)
        self.read_bone_index = self.define_read_index(model.bone_count)
        self.read_morph_index = self.define_read_index(model.morph_count)
        self.read_rigidbody_index = self.define_read_index(model.rigidbody_count)

        # モデル名（英語）
        model.english_name = self.read_text()

        # コメント
        model.comment = self.read_text()

        # コメント英
        model.english_comment = self.read_text()

        # 頂点
        self.read_vertices(model)

        # 面
        self.read_surfaces(model)

        # テクスチャ
        self.read_textures(model)

        # 材質
        self.read_materials(model)

        # ボーン
        self.read_bones(model)

        # モーフ
        self.read_morphs(model)

        # 表示枠
        self.read_display_slots(model)

        # 剛体
        self.read_rigidbodies(model)

        # ジョイント
        self.read_joints(model)

    def read_vertices(self, model: PmxModel):
        """頂点データ読み込み"""
        for _ in range(self.read_int()):
            vertex = Vertex()
            vertex.position = self.read_MVector3D()
            vertex.normal = self.read_MVector3D()
            vertex.uv = self.read_MVector2D()

            if model.extended_uv_count > 0:
                vertex.extended_uvs.append(self.read_MVector4D())

            vertex.deform_type = DeformType(self.read_ubyte())
            if vertex.deform_type == DeformType.BDEF1:
                vertex.deform = Bdef1(self.read_bone_index())
            elif vertex.deform_type == DeformType.BDEF2:
                vertex.deform = Bdef2(
                    self.read_bone_index(), self.read_bone_index(), self.read_float()
                )
            elif vertex.deform_type == DeformType.BDEF4:
                vertex.deform = Bdef4(
                    self.read_bone_index(),
                    self.read_bone_index(),
                    self.read_bone_index(),
                    self.read_bone_index(),
                    self.read_float(),
                    self.read_float(),
                    self.read_float(),
                    self.read_float(),
                )
            else:
                vertex.deform = Sdef(
                    self.read_bone_index(),
                    self.read_bone_index(),
                    self.read_float(),
                    self.read_MVector3D(),
                    self.read_MVector3D(),
                    self.read_MVector3D(),
                )
            vertex.edge_factor = self.read_float()
            model.vertices.append(vertex)

    def read_surfaces(self, model: PmxModel):
        """面データ読み込み"""
        for _ in range(0, self.read_int(), 3):
            v0 = self.read_vertex_index()
            v1 = self.read_vertex_index()
            v2 = self.read_vertex_index()

            model.surfaces.append(Surface(v0, v1, v2))

    def read_textures(self, model: PmxModel):
        """テクスチャデータ読み込み"""
        for _ in range(self.read_int()):
            model.textures.append(Texture(self.read_text()))

    def read_materials(self, model: PmxModel):
        """材質データ読み込み"""
        for _ in range(self.read_int()):
            material = Material()
            material.name = self.read_text()
            material.english_name = self.read_text()
            material.diffuse_color = self.read_MVector4D()
            material.specular_color = self.read_MVector3D()
            material.specular_factor = self.read_float()
            material.ambient_color = self.read_MVector3D()
            material.draw_flg = DrawFlg(self.read_byte())
            material.edge_color = self.read_MVector4D()
            material.edge_size = self.read_float()
            material.texture_index = self.read_texture_index()
            material.sphere_texture_index = self.read_texture_index()
            material.sphere_mode = SphereMode(self.read_byte())
            material.toon_sharing_flg = ToonSharing(self.read_byte())
            if material.toon_sharing_flg == ToonSharing.INDIVIDUAL:
                # 個別の場合、テクスチャINDEX
                material.toon_texture_index = self.read_texture_index()
            else:
                # 共有の場合、0-9の共有テクスチャINDEX
                material.toon_texture_index = self.read_byte()
            material.comment = self.read_text()
            material.vertices_count = self.read_int()
            model.materials.append(material)

    def read_bones(self, model: PmxModel):
        """ボーンデータ読み込み"""
        for _ in range(self.read_int()):
            bone = Bone()
            bone.name = self.read_text()
            bone.english_name = self.read_text()
            bone.position = self.read_MVector3D()
            bone.parent_index = self.read_bone_index()
            bone.layer = self.read_int()
            bone.bone_flg = BoneFlg(self.read_short())

            if BoneFlg.TAIL_IS_BONE in bone.bone_flg:
                bone.tail_index = self.read_bone_index()
            else:
                bone.tail_position = self.read_MVector3D()

            if (
                BoneFlg.IS_EXTERNAL_TRANSLATION in bone.bone_flg
                or BoneFlg.IS_EXTERNAL_ROTATION in bone.bone_flg
            ):
                bone.effect_index = self.read_bone_index()
                bone.effect_factor = self.read_float()

            if BoneFlg.HAS_FIXED_AXIS in bone.bone_flg:
                bone.fixed_axis = self.read_MVector3D()

            if BoneFlg.HAS_LOCAL_COORDINATE in bone.bone_flg:
                bone.local_x_vector = self.read_MVector3D()
                bone.local_z_vector = self.read_MVector3D()

            if BoneFlg.IS_EXTERNAL_PARENT_DEFORM in bone.bone_flg:
                bone.external_key = self.read_int()

            if BoneFlg.IS_IK in bone.bone_flg:
                ik = Ik()
                ik.bone_index = self.read_bone_index()
                ik.loop_count = self.read_int()
                ik.unit_rotation.radians = MVector3D(self.read_float(), 0, 0)
                for _i in range(self.read_int()):
                    ik_link = IkLink()
                    ik_link.bone_index = self.read_bone_index()
                    ik_link.angle_limit = bool(self.read_byte())
                    if ik_link.angle_limit:
                        ik_link.min_angle_limit.radians = self.read_MVector3D()
                        ik_link.max_angle_limit.radians = self.read_MVector3D()
                    ik.links.append(ik_link)
                bone.ik = ik

            model.bones.append(bone)

    def read_morphs(self, model: PmxModel):
        """モーフデータ読み込み"""
        for _ in range(self.read_int()):
            morph = Morph()
            morph.name = self.read_text()
            morph.english_name = self.read_text()
            morph.panel = MorphPanel(self.read_byte())
            morph.morph_type = MorphType(self.read_byte())

            for _ in range(self.read_int()):
                if morph.morph_type == MorphType.GROUP:
                    morph.offsets.append(
                        GroupMorphOffset(self.read_morph_index(), self.read_float())
                    )
                elif morph.morph_type == MorphType.VERTEX:
                    morph.offsets.append(
                        VertexMorphOffset(
                            self.read_vertex_index(), self.read_MVector3D()
                        )
                    )
                elif morph.morph_type == MorphType.BONE:
                    morph.offsets.append(
                        BoneMorphOffset(
                            self.read_vertex_index(),
                            self.read_MVector3D(),
                            self.read_MQuaternion(),
                        )
                    )
                elif morph.morph_type in [
                    MorphType.UV,
                    MorphType.EXTENDED_UV1,
                    MorphType.EXTENDED_UV2,
                    MorphType.EXTENDED_UV3,
                    MorphType.EXTENDED_UV4,
                ]:
                    morph.offsets.append(
                        UvMorphOffset(self.read_vertex_index(), self.read_MVector4D())
                    )
                elif morph.morph_type == MorphType.MATERIAL:
                    morph.offsets.append(
                        MaterialMorphOffset(
                            self.read_material_index(),
                            MaterialMorphCalcMode(self.read_byte()),
                            self.read_MVector4D(),
                            self.read_MVector3D(),
                            self.read_float(),
                            self.read_MVector3D(),
                            self.read_MVector4D(),
                            self.read_float(),
                            self.read_MVector4D(),
                            self.read_MVector4D(),
                            self.read_MVector4D(),
                        )
                    )

            model.morphs.append(morph)

    def read_display_slots(self, model: PmxModel):
        """表示枠データ読み込み"""
        for _ in range(self.read_int()):
            display_slot = DisplaySlot()
            display_slot.name = self.read_text()
            display_slot.english_name = self.read_text()
            display_slot.special_flg = Switch(self.read_byte())
            for _i in range(self.read_int()):
                reference = DisplaySlotReference()
                reference.display_type = DisplayType(self.read_byte())
                if reference.display_type == DisplayType.BONE:
                    reference.display_index = self.read_bone_index()
                else:
                    reference.display_index = self.read_morph_index()
                display_slot.references.append(reference)

            model.display_slots.append(display_slot)

    def read_rigidbodies(self, model: PmxModel):
        """剛体データ読み込み"""
        for _ in range(self.read_int()):
            rigidbody = RigidBody()
            rigidbody.name = self.read_text()
            rigidbody.english_name = self.read_text()
            rigidbody.bone_index = self.read_bone_index()
            rigidbody.collision_group = self.read_byte()
            rigidbody.no_collision_group = RigidBodyCollisionGroup(self.read_ushort())
            rigidbody.shape_type = RigidBodyShape(self.read_byte())
            rigidbody.shape_size = self.read_MVector3D()
            rigidbody.shape_position = self.read_MVector3D()
            rigidbody.shape_rotation.radians = self.read_MVector3D()
            rigidbody.param.mass = self.read_float()
            rigidbody.param.linear_damping = self.read_float()
            rigidbody.param.angular_damping = self.read_float()
            rigidbody.param.restitution = self.read_float()
            rigidbody.param.friction = self.read_float()
            rigidbody.mode = RigidBodyMode(self.read_byte())
            model.rigidbodies.append(rigidbody)

    def read_joints(self, model: PmxModel):
        """モデルデータ読み込み"""
        for _ in range(self.read_int()):
            joint = Joint()
            joint.name = self.read_text()
            joint.english_name = self.read_text()
            joint.joint_type = self.read_byte()
            joint.rigidbody_index_a = self.read_rigidbody_index()
            joint.rigidbody_index_b = self.read_rigidbody_index()
            joint.position = self.read_MVector3D()
            joint.rotation.radians = self.read_MVector3D()
            joint.param.translation_limit_min = self.read_MVector3D()
            joint.param.translation_limit_max = self.read_MVector3D()
            joint.param.rotation_limit_min.radians = self.read_MVector3D()
            joint.param.rotation_limit_max.radians = self.read_MVector3D()
            joint.param.spring_constant_translation = self.read_MVector3D()
            joint.param.spring_constant_rotation = self.read_MVector3D()
            model.joints.append(joint)

    def define_read_index(self, count: int, is_vertex=False):
        """
        INDEX読み取り定義

        Parameters
        ----------
        count : int
            Indexサイズ数
        is_vertex : bool
            頂点データの場合、1,2 は unsigned なので切り分け

        Returns
        -------
        function
            読み取り定義関数
        """
        if count == 1 and is_vertex:

            def read_index():
                return self.read_ubyte()

            return read_index
        elif count == 2 and is_vertex:

            def read_index():
                return self.read_ushort()

            return read_index
        elif count == 1 and not is_vertex:

            def read_index():
                return self.read_byte()

            return read_index
        elif count == 2 and not is_vertex:

            def read_index():
                return self.read_short()

            return read_index
        else:

            def read_index():
                return self.read_int()

            return read_index
