import struct
from io import BufferedWriter
from math import isinf, isnan

from mlib.base.base import BaseModel
from mlib.base.logger import MLogger
from mlib.pmx.pmx_collection import PmxModel
from mlib.pmx.pmx_part import (
    Bdef1,
    Bdef2,
    Bdef4,
    BoneMorphOffset,
    GroupMorphOffset,
    MaterialMorphOffset,
    Sdef,
    ToonSharing,
    UvMorphOffset,
    VertexMorphOffset,
)

logger = MLogger(__name__)


TYPE_FLOAT = "f"
TYPE_BOOL = "c"
TYPE_BYTE = "<b"
TYPE_UNSIGNED_BYTE = "<B"
TYPE_SHORT = "<h"
TYPE_UNSIGNED_SHORT = "<H"
TYPE_INT = "<i"
TYPE_UNSIGNED_INT = "<I"
TYPE_LONG = "<l"
TYPE_UNSIGNED_LONG = "<L"


class PmxWriter(BaseModel):
    @staticmethod
    def write(model: PmxModel, output_path: str):
        with open(output_path, "wb") as fout:
            # シグニチャ
            fout.write(b"PMX ")
            fout.write(struct.pack(TYPE_FLOAT, float(2)))
            # 後続するデータ列のバイトサイズ  PMX2.0は 8 で固定
            fout.write(struct.pack(TYPE_BYTE, int(8)))
            # エンコード方式  | 0:UTF16
            fout.write(struct.pack(TYPE_BYTE, 0))
            # 追加UV数
            fout.write(struct.pack(TYPE_BYTE, model.extended_uv_count))
            # 頂点Indexサイズ | 1,2,4 のいずれか
            vertex_idx_size, vertex_idx_type = PmxWriter.define_write_index(len(model.vertices), is_vertex=True)
            fout.write(struct.pack(TYPE_BYTE, vertex_idx_size))
            # テクスチャIndexサイズ | 1,2,4 のいずれか
            texture_idx_size, texture_idx_type = PmxWriter.define_write_index(len(model.textures), is_vertex=False)
            fout.write(struct.pack(TYPE_BYTE, texture_idx_size))
            # 材質Indexサイズ | 1,2,4 のいずれか
            material_idx_size, material_idx_type = PmxWriter.define_write_index(len(model.materials), is_vertex=False)
            fout.write(struct.pack(TYPE_BYTE, material_idx_size))
            # ボーンIndexサイズ | 1,2,4 のいずれか
            bone_idx_size, bone_idx_type = PmxWriter.define_write_index(len(model.bones), is_vertex=False)
            fout.write(struct.pack(TYPE_BYTE, bone_idx_size))
            # モーフIndexサイズ | 1,2,4 のいずれか
            morph_idx_size, morph_idx_type = PmxWriter.define_write_index(len(model.morphs), is_vertex=False)
            fout.write(struct.pack(TYPE_BYTE, morph_idx_size))
            # 剛体Indexサイズ | 1,2,4 のいずれか
            rigidbody_idx_size, rigidbody_idx_type = PmxWriter.define_write_index(len(model.rigidbodies), is_vertex=False)
            fout.write(struct.pack(TYPE_BYTE, rigidbody_idx_size))

            # モデル名(日本語)
            PmxWriter.write_text(fout, model.name, "Pmx Model")
            # モデル名(英語)
            PmxWriter.write_text(fout, model.english_name, "Pmx Model")
            # コメント(日本語)
            PmxWriter.write_text(fout, model.comment, "")
            # コメント(英語)
            PmxWriter.write_text(fout, model.english_comment, "")

            # 頂点出力
            model = PmxWriter.write_vertices(model, fout, bone_idx_type)

            # 頂点出力
            model = PmxWriter.write_faces(model, fout, vertex_idx_type)

            # テクスチャ出力
            model = PmxWriter.write_textures(model, fout)

            # 材質の数
            fout.write(struct.pack(TYPE_INT, len(model.materials)))

            # 材質データ
            for midx, material in enumerate(model.materials):
                # 材質名
                PmxWriter.write_text(fout, material.name, f"Material {midx}")
                PmxWriter.write_text(fout, material.english_name, f"Material {midx}")
                # Diffuse
                PmxWriter.write_number(fout, TYPE_FLOAT, float(material.diffuse_color.x), True)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(material.diffuse_color.y), True)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(material.diffuse_color.z), True)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(material.diffuse_color.w), True)
                # Specular
                PmxWriter.write_number(fout, TYPE_FLOAT, float(material.specular_color.x), True)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(material.specular_color.y), True)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(material.specular_color.z), True)
                # Specular係数
                PmxWriter.write_number(fout, TYPE_FLOAT, float(material.specular_factor), True)
                # Ambient
                PmxWriter.write_number(fout, TYPE_FLOAT, float(material.ambient_color.x), True)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(material.ambient_color.y), True)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(material.ambient_color.z), True)
                # 描画フラグ(8bit)
                fout.write(struct.pack(TYPE_BYTE, material.draw_flg.value))
                # エッジ色 (R,G,B,A)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(material.edge_color.x), True)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(material.edge_color.y), True)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(material.edge_color.z), True)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(material.edge_color.w), True)
                # エッジサイズ
                PmxWriter.write_number(fout, TYPE_FLOAT, float(material.edge_size), True)
                # 通常テクスチャ
                fout.write(struct.pack(texture_idx_type, material.texture_index))
                # スフィアテクスチャ
                fout.write(struct.pack(texture_idx_type, material.sphere_texture_index))
                # スフィアモード
                fout.write(struct.pack(TYPE_BYTE, material.sphere_mode))
                # 共有Toonフラグ
                fout.write(struct.pack(TYPE_BYTE, material.toon_sharing_flg))
                if material.toon_sharing_flg == ToonSharing.INDIVIDUAL.value:
                    # 個別Toonテクスチャ
                    fout.write(struct.pack(texture_idx_type, material.toon_texture_index))
                else:
                    # 共有Toonテクスチャ[0～9]
                    fout.write(struct.pack(TYPE_BYTE, material.toon_texture_index))
                # コメント
                PmxWriter.write_text(fout, material.comment, "")
                # 材質に対応する面(頂点)数
                PmxWriter.write_number(fout, TYPE_INT, material.vertices_count)

            logger.debug("-- 材質データ出力終了({count})", count=len(model.materials))

            # ボーンの数
            fout.write(struct.pack(TYPE_INT, len(model.bones)))

            for bidx, bone in enumerate(model.bones):
                # ボーン名
                PmxWriter.write_text(fout, bone.name, f"Bone {bidx}")
                PmxWriter.write_text(fout, bone.english_name, f"Bone {bidx}")
                # position
                PmxWriter.write_number(fout, TYPE_FLOAT, float(bone.position.x))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(bone.position.y))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(bone.position.z))
                # 親ボーンのボーンIndex
                fout.write(struct.pack(bone_idx_type, bone.parent_index))
                # 変形階層
                PmxWriter.write_number(fout, TYPE_INT, bone.layer, True)
                # ボーンフラグ
                fout.write(struct.pack(TYPE_SHORT, bone.bone_flg.value))

                if bone.is_tail_bone():
                    # 接続先ボーンのボーンIndex
                    fout.write(struct.pack(bone_idx_type, bone.tail_index))
                else:
                    # 接続先位置
                    PmxWriter.write_number(fout, TYPE_FLOAT, float(bone.tail_position.x))
                    PmxWriter.write_number(fout, TYPE_FLOAT, float(bone.tail_position.y))
                    PmxWriter.write_number(fout, TYPE_FLOAT, float(bone.tail_position.z))

                if bone.is_external_translation() or bone.is_external_rotation():
                    # 付与親指定ありの場合
                    fout.write(struct.pack(bone_idx_type, bone.effect_index))
                    PmxWriter.write_number(fout, TYPE_FLOAT, bone.effect_factor)

                if bone.has_fixed_axis():
                    # 軸制限先
                    PmxWriter.write_number(fout, TYPE_FLOAT, float(bone.fixed_axis.x))
                    PmxWriter.write_number(fout, TYPE_FLOAT, float(bone.fixed_axis.y))
                    PmxWriter.write_number(fout, TYPE_FLOAT, float(bone.fixed_axis.z))

                if bone.has_local_coordinate():
                    # ローカルX
                    PmxWriter.write_number(fout, TYPE_FLOAT, float(bone.local_x_vector.x))
                    PmxWriter.write_number(fout, TYPE_FLOAT, float(bone.local_x_vector.y))
                    PmxWriter.write_number(fout, TYPE_FLOAT, float(bone.local_x_vector.z))
                    # ローカルZ
                    PmxWriter.write_number(fout, TYPE_FLOAT, float(bone.local_z_vector.x))
                    PmxWriter.write_number(fout, TYPE_FLOAT, float(bone.local_z_vector.y))
                    PmxWriter.write_number(fout, TYPE_FLOAT, float(bone.local_z_vector.z))

                if bone.is_external_parent_deform():
                    PmxWriter.write_number(fout, TYPE_INT, bone.external_key)

                if bone.is_ik():
                    # IKボーン
                    # n  : ボーンIndexサイズ  | IKターゲットボーンのボーンIndex
                    fout.write(struct.pack(bone_idx_type, bone.ik.bone_index))
                    # 4  : int  	| IKループ回数
                    PmxWriter.write_number(fout, TYPE_INT, bone.ik.loop_count)
                    # 4  : float	| IKループ計算時の1回あたりの制限角度 -> ラジアン角
                    PmxWriter.write_number(fout, TYPE_FLOAT, bone.ik.unit_rotation.radians.x)
                    # 4  : int  	| IKリンク数 : 後続の要素数
                    PmxWriter.write_number(fout, TYPE_INT, len(bone.ik.links))

                    for link in bone.ik.links:
                        # n  : ボーンIndexサイズ  | リンクボーンのボーンIndex
                        fout.write(struct.pack(bone_idx_type, link.bone_index))
                        # 1  : byte	| 角度制限 0:OFF 1:ON
                        fout.write(struct.pack(TYPE_BYTE, int(link.angle_limit)))

                        if link.angle_limit == 1:
                            PmxWriter.write_number(fout, TYPE_FLOAT, float(link.min_angle_limit.radians.x))
                            PmxWriter.write_number(fout, TYPE_FLOAT, float(link.min_angle_limit.radians.y))
                            PmxWriter.write_number(fout, TYPE_FLOAT, float(link.min_angle_limit.radians.z))

                            PmxWriter.write_number(fout, TYPE_FLOAT, float(link.max_angle_limit.radians.x))
                            PmxWriter.write_number(fout, TYPE_FLOAT, float(link.max_angle_limit.radians.y))
                            PmxWriter.write_number(fout, TYPE_FLOAT, float(link.max_angle_limit.radians.z))

            logger.debug("-- ボーンデータ出力終了({count})", count=len(model.bones))

            # モーフの数
            PmxWriter.write_number(fout, TYPE_INT, len(model.morphs))

            for midx, morph in enumerate(model.morphs):
                # モーフ名
                PmxWriter.write_text(fout, morph.name, f"Morph {midx}")
                PmxWriter.write_text(fout, morph.english_name, f"Morph {midx}")
                # 操作パネル (PMD:カテゴリ) 1:眉(左下) 2:目(左上) 3:口(右上) 4:その他(右下)  | 0:システム予約
                fout.write(struct.pack(TYPE_BYTE, morph.panel))
                # モーフ種類 - 0:グループ, 1:頂点, 2:ボーン, 3:UV, 4:追加UV1, 5:追加UV2, 6:追加UV3, 7:追加UV4, 8:材質
                fout.write(struct.pack(TYPE_BYTE, morph.morph_type))
                # モーフのオフセット数 : 後続の要素数
                PmxWriter.write_number(fout, TYPE_INT, len(morph.offsets))

                for offset in morph.offsets:
                    if type(offset) is VertexMorphOffset:
                        # 頂点モーフ
                        fout.write(struct.pack(vertex_idx_type, offset.vertex_index))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.position_offset.x))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.position_offset.y))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.position_offset.z))
                    elif type(offset) is UvMorphOffset:
                        # UVモーフ
                        fout.write(struct.pack(vertex_idx_type, offset.vertex_index))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.uv.x))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.uv.y))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.uv.z))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.uv.w))
                    elif type(offset) is BoneMorphOffset:
                        # ボーンモーフ
                        fout.write(struct.pack(bone_idx_type, offset.bone_index))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.position.x))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.position.y))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.position.z))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.rotation.x))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.rotation.y))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.rotation.z))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.rotation.scalar))
                    elif type(offset) is MaterialMorphOffset:
                        # 材質モーフ
                        fout.write(struct.pack(material_idx_type, offset.material_index))
                        fout.write(struct.pack(TYPE_BYTE, int(offset.calc_mode)))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.diffuse.x))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.diffuse.y))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.diffuse.z))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.diffuse.w))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.specular.x))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.specular.y))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.specular.z))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.specular_factor))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.ambient.x))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.ambient.y))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.ambient.z))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.edge_color.x))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.edge_color.y))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.edge_color.z))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.edge_color.w))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.edge_size))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.texture_factor.x))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.texture_factor.y))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.texture_factor.z))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.texture_factor.w))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.sphere_texture_factor.x))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.sphere_texture_factor.y))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.sphere_texture_factor.z))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.sphere_texture_factor.w))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.toon_texture_factor.x))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.toon_texture_factor.y))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.toon_texture_factor.z))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.toon_texture_factor.w))
                    elif type(offset) is GroupMorphOffset:
                        # グループモーフ
                        fout.write(struct.pack(morph_idx_type, offset.morph_index))
                        PmxWriter.write_number(fout, TYPE_FLOAT, float(offset.morph_factor))

            logger.debug("-- モーフデータ出力終了({count})", count=len(model.morphs))

            # 表示枠の数
            PmxWriter.write_number(fout, TYPE_INT, len(model.display_slots))

            for didx, display_slot in enumerate(model.display_slots):
                # 表示枠名
                PmxWriter.write_text(fout, display_slot.name, f"Display {didx}")
                PmxWriter.write_text(fout, display_slot.english_name, f"Display {didx}")
                # 特殊枠フラグ - 0:通常枠 1:特殊枠
                fout.write(struct.pack(TYPE_BYTE, display_slot.special_flg.value))
                # 枠内要素数
                PmxWriter.write_number(fout, TYPE_INT, len(display_slot.references))
                # ボーンの場合
                for reference in display_slot.references:
                    # 要素対象 0:ボーン 1:モーフ
                    fout.write(struct.pack(TYPE_BYTE, reference.display_type))
                    if reference.display_type == 0:
                        # ボーンIndex
                        fout.write(struct.pack(bone_idx_type, reference.display_index))
                    else:
                        # モーフIndex
                        fout.write(struct.pack(morph_idx_type, reference.display_index))

            logger.debug("-- 表示枠データ出力終了({count})", count=len(model.display_slots))

            # 剛体の数
            PmxWriter.write_number(fout, TYPE_INT, len(list(model.rigidbodies)))

            for ridx, rigidbody in enumerate(model.rigidbodies):
                # 剛体名
                PmxWriter.write_text(fout, rigidbody.name, f"Rigidbody {ridx}")
                PmxWriter.write_text(fout, rigidbody.english_name, f"Rigidbody {ridx}")
                # ボーンIndex
                fout.write(struct.pack(bone_idx_type, rigidbody.bone_index))
                # 1  : byte	| グループ
                fout.write(struct.pack(TYPE_BYTE, rigidbody.collision_group))
                # 2  : ushort	| 非衝突グループフラグ
                fout.write(struct.pack(TYPE_UNSIGNED_SHORT, rigidbody.no_collision_group.value))
                # 1  : byte	| 形状 - 0:球 1:箱 2:カプセル
                fout.write(struct.pack(TYPE_BYTE, rigidbody.shape_type))
                # 12 : float3	| サイズ(x,y,z)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(rigidbody.shape_size.x), True)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(rigidbody.shape_size.y), True)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(rigidbody.shape_size.z), True)
                # 12 : float3	| 位置(x,y,z)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(rigidbody.shape_position.x))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(rigidbody.shape_position.y))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(rigidbody.shape_position.z))
                # 12 : float3	| 回転(x,y,z)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(rigidbody.shape_rotation.radians.x))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(rigidbody.shape_rotation.radians.y))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(rigidbody.shape_rotation.radians.z))
                # 4  : float	| 質量
                PmxWriter.write_number(fout, TYPE_FLOAT, float(rigidbody.param.mass), True)
                # 4  : float	| 移動減衰
                PmxWriter.write_number(fout, TYPE_FLOAT, float(rigidbody.param.linear_damping), True)
                # 4  : float	| 回転減衰
                PmxWriter.write_number(fout, TYPE_FLOAT, float(rigidbody.param.angular_damping), True)
                # 4  : float	| 反発力
                PmxWriter.write_number(fout, TYPE_FLOAT, float(rigidbody.param.restitution), True)
                # 4  : float	| 摩擦力
                PmxWriter.write_number(fout, TYPE_FLOAT, float(rigidbody.param.friction), True)
                # 1  : byte	| 剛体の物理演算 - 0:ボーン追従(static) 1:物理演算(dynamic) 2:物理演算 + Bone位置合わせ
                fout.write(struct.pack(TYPE_BYTE, rigidbody.mode))

            logger.debug("-- 剛体データ出力終了({count})", count=len(model.rigidbodies))

            # ジョイントの数
            PmxWriter.write_number(fout, TYPE_INT, len(list(model.joints)))

            for jidx, joint in enumerate(model.joints):
                # ジョイント名
                PmxWriter.write_text(fout, joint.name, f"Joint {jidx}")
                PmxWriter.write_text(fout, joint.english_name, f"Joint {jidx}")
                # 1  : byte	| Joint種類 - 0:スプリング6DOF   | PMX2.0では 0 のみ(拡張用)
                fout.write(struct.pack(TYPE_BYTE, joint.joint_type))
                # n  : 剛体Indexサイズ  | 関連剛体AのIndex - 関連なしの場合は-1
                fout.write(struct.pack(rigidbody_idx_type, joint.rigidbody_index_a))
                # n  : 剛体Indexサイズ  | 関連剛体BのIndex - 関連なしの場合は-1
                fout.write(struct.pack(rigidbody_idx_type, joint.rigidbody_index_b))
                # 12 : float3	| 位置(x,y,z)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.position.x))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.position.y))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.position.z))
                # 12 : float3	| 回転(x,y,z) -> ラジアン角
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.rotation.radians.x))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.rotation.radians.y))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.rotation.radians.z))
                # 12 : float3	| 移動制限-下限(x,y,z)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.param.translation_limit_min.x))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.param.translation_limit_min.y))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.param.translation_limit_min.z))
                # 12 : float3	| 移動制限-上限(x,y,z)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.param.translation_limit_max.x))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.param.translation_limit_max.y))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.param.translation_limit_max.z))
                # 12 : float3	| 回転制限-下限(x,y,z) -> ラジアン角
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.param.rotation_limit_min.radians.x))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.param.rotation_limit_min.radians.y))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.param.rotation_limit_min.radians.z))
                # 12 : float3	| 回転制限-上限(x,y,z) -> ラジアン角
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.param.rotation_limit_max.radians.x))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.param.rotation_limit_max.radians.y))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.param.rotation_limit_max.radians.z))
                # 12 : float3	| バネ定数-移動(x,y,z)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.param.spring_constant_translation.x))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.param.spring_constant_translation.y))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.param.spring_constant_translation.z))
                # 12 : float3	| バネ定数-回転(x,y,z)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.param.spring_constant_rotation.x))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.param.spring_constant_rotation.y))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(joint.param.spring_constant_rotation.z))

            logger.debug("-- ジョイントデータ出力終了({count})", count=len(model.joints))

    @staticmethod
    def write_vertices(model: PmxModel, fout: BufferedWriter, bone_idx_type: str) -> PmxModel:
        fout.write(struct.pack(TYPE_INT, len(model.vertices)))

        # 頂点データ
        for vertex in model.vertices:
            # position
            PmxWriter.write_number(fout, TYPE_FLOAT, float(vertex.position.x))
            PmxWriter.write_number(fout, TYPE_FLOAT, float(vertex.position.y))
            PmxWriter.write_number(fout, TYPE_FLOAT, float(vertex.position.z))
            # normal
            PmxWriter.write_number(fout, TYPE_FLOAT, float(vertex.normal.x))
            PmxWriter.write_number(fout, TYPE_FLOAT, float(vertex.normal.y))
            PmxWriter.write_number(fout, TYPE_FLOAT, float(vertex.normal.z))
            # uv
            PmxWriter.write_number(fout, TYPE_FLOAT, float(vertex.uv.x))
            PmxWriter.write_number(fout, TYPE_FLOAT, float(vertex.uv.y))
            # 追加uv
            for uv in vertex.extended_uvs:
                PmxWriter.write_number(fout, TYPE_FLOAT, float(uv.x))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(uv.y))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(uv.z))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(uv.w))

            # deform
            if type(vertex.deform) is Bdef1:
                fout.write(struct.pack(TYPE_BYTE, 0))
                PmxWriter.write_number(fout, bone_idx_type, vertex.deform.indices[0])
            elif type(vertex.deform) is Bdef2:
                fout.write(struct.pack(TYPE_BYTE, 1))
                PmxWriter.write_number(fout, bone_idx_type, vertex.deform.indices[0])
                PmxWriter.write_number(fout, bone_idx_type, vertex.deform.indices[1])

                PmxWriter.write_number(fout, TYPE_FLOAT, vertex.deform.weights[0], True)
            elif type(vertex.deform) is Bdef4:
                fout.write(struct.pack(TYPE_BYTE, 2))
                PmxWriter.write_number(fout, bone_idx_type, vertex.deform.indices[0])
                PmxWriter.write_number(fout, bone_idx_type, vertex.deform.indices[1])
                PmxWriter.write_number(fout, bone_idx_type, vertex.deform.indices[2])
                PmxWriter.write_number(fout, bone_idx_type, vertex.deform.indices[3])

                PmxWriter.write_number(fout, TYPE_FLOAT, vertex.deform.weights[0], True)
                PmxWriter.write_number(fout, TYPE_FLOAT, vertex.deform.weights[1], True)
                PmxWriter.write_number(fout, TYPE_FLOAT, vertex.deform.weights[2], True)
                PmxWriter.write_number(fout, TYPE_FLOAT, vertex.deform.weights[3], True)
            elif type(vertex.deform) is Sdef:
                fout.write(struct.pack(TYPE_BYTE, 3))
                PmxWriter.write_number(fout, bone_idx_type, vertex.deform.indices[0])
                PmxWriter.write_number(fout, bone_idx_type, vertex.deform.indices[1])
                PmxWriter.write_number(fout, TYPE_FLOAT, vertex.deform.weights[0], True)
                PmxWriter.write_number(fout, TYPE_FLOAT, float(vertex.deform.sdef_c.x))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(vertex.deform.sdef_c.y))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(vertex.deform.sdef_c.z))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(vertex.deform.sdef_r0.x))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(vertex.deform.sdef_r0.y))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(vertex.deform.sdef_r0.z))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(vertex.deform.sdef_r1.x))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(vertex.deform.sdef_r1.y))
                PmxWriter.write_number(fout, TYPE_FLOAT, float(vertex.deform.sdef_r1.z))
            else:
                logger.error("頂点deformなし: {vertex}", vertex=str(vertex))

            PmxWriter.write_number(fout, TYPE_FLOAT, float(vertex.edge_factor), True)

        logger.debug("-- 頂点データ出力終了({count})", count=len(model.vertices))

        return model

    @staticmethod
    def write_faces(model: PmxModel, fout: BufferedWriter, vertex_idx_type: str) -> PmxModel:
        # 面の数
        fout.write(struct.pack(TYPE_INT, len(model.faces) * 3))

        # 面データ
        for face in model.faces:
            for vidx in face.vertices:
                fout.write(struct.pack(vertex_idx_type, vidx))

        logger.debug("-- 面データ出力終了({count})", count=len(model.faces))

        return model

    @staticmethod
    def write_textures(model: PmxModel, fout: BufferedWriter) -> PmxModel:
        # テクスチャの数
        fout.write(struct.pack(TYPE_INT, len(model.textures)))

        # テクスチャデータ
        for texture in model.textures:
            PmxWriter.write_text(fout, texture.texture_path, "")

        logger.debug("-- テクスチャデータ出力終了({count})", count=len(model.textures))

        return model

    @staticmethod
    def define_write_index(size: int, is_vertex: bool) -> tuple[int, str]:
        if is_vertex:
            if 256 > size:
                return 1, TYPE_UNSIGNED_BYTE
            elif 256 <= size <= 65535:
                return 2, TYPE_UNSIGNED_SHORT
        else:
            if 128 > size:
                return 1, TYPE_BYTE
            elif 128 <= size <= 32767:
                return 2, TYPE_SHORT

        return 4, TYPE_INT

    @staticmethod
    def write_text(fout, text: str, default_text: str, type=TYPE_INT):
        try:
            btxt = text.encode("utf-16-le")
        except Exception:
            btxt = default_text.encode("utf-16-le")
        fout.write(struct.pack(type, len(btxt)))
        fout.write(btxt)

    @staticmethod
    def write_number(fout, val_type: str, val: float, is_positive_only=False):
        if isnan(val) or isinf(val):
            # 正常な値を強制設定
            val = 0
        val = max(0, val) if is_positive_only else val

        try:
            # INT型の場合、INT変換
            if val_type in [TYPE_FLOAT]:
                fout.write(struct.pack(val_type, float(val)))
            else:
                fout.write(struct.pack(val_type, int(val)))
        except Exception as e:
            logger.error("val_type in [float]: %s", val_type in [TYPE_FLOAT])
            logger.error(
                "PmxWriter.write_number失敗: type: %s, val: %s, int(val): %s",
                val_type,
                val,
                int(val),
            )
            raise e
