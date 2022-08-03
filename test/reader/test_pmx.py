import pytest
from mlib.model.pmx import (
    BoneFlg,
    DeformType,
    DisplayType,
    DrawFlg,
    MorphPanel,
    MorphType,
    RigidBodyCollisionGroup,
    RigidBodyMode,
    RigidBodyShape,
)


def test_read_by_filepath_error():
    import os

    from mlib.exception import MParseException
    from mlib.reader.pmx import PmxReader

    reader = PmxReader()
    with pytest.raises(MParseException):
        reader.read_by_filepath(os.path.join("test", "resources", "サンプルモーション.vmd"))


def test_read_by_filepath_ok():
    import os

    import numpy as np
    from mlib.model.pmx import PmxModel
    from mlib.reader.pmx import PmxReader

    reader = PmxReader()
    model: PmxModel = reader.read_by_filepath(
        os.path.join("test", "resources", "サンプルモデル.pmx")
    )
    assert b"PMX " == model.signature
    assert 2.0 == model.version
    assert 2 == model.vertex_count
    assert 1 == model.material_count
    assert 2 == model.bone_count
    assert 2 == model.morph_count
    assert 1 == model.rigidbody_count
    assert "v2配布用素体03" == model.name
    assert "sample model" == model.english_name
    # 頂点
    assert np.isclose(
        np.array([0.3187934, 10.7431, -0.6637576]),
        model.vertices[0].position.vector,
    ).all()
    assert np.isclose(
        np.array([-0.4265585, -0.2953186, -0.8548887]),
        model.vertices[0].normal.vector,
    ).all()
    assert np.isclose(
        np.array([0.5220807, 0.552156]),
        model.vertices[0].uv.vector,
    ).all()
    assert DeformType.BDEF2 == model.vertices[0].deform_type
    assert np.isclose(
        np.array([4, 108]),
        model.vertices[0].deform.get_indecies(),
    ).all()
    assert np.isclose(
        np.array([0.7978904, 0.2021096]),
        model.vertices[0].deform.get_weights(),
    ).all()
    assert 1 == model.vertices[0].edge_factor
    # 面
    assert [2, 1, 0] == model.surfaces[0].vertices
    # テクスチャ
    assert "tex\\_09.png" == model.textures[16].texture_path
    assert "tex\\MatcapWarp_01.png" == model.textures[1].texture_path
    # 材質
    assert "00_FaceEyeline" == model.materials[7].name
    assert (
        "N00_000_00_FaceEyeline_00_FACE (Instance)" == model.materials[7].english_name
    )
    assert np.isclose(
        np.array([1, 1, 1, 1]),
        model.materials[7].diffuse_color.vector,
    ).all()
    assert np.isclose(
        np.array([0, 0, 0]),
        model.materials[7].specular_color.vector,
    ).all()
    assert 0 == model.materials[7].specular_factor
    assert np.isclose(
        np.array([0.5, 0.5, 0.5]),
        model.materials[7].ambient_color.vector,
    ).all()
    assert DrawFlg.DOUBLE_SIDED_DRAWING in model.materials[7].draw_flg
    assert DrawFlg.GROUND_SHADOW in model.materials[7].draw_flg
    assert DrawFlg.DRAWING_ON_SELF_SHADOW_MAPS in model.materials[7].draw_flg
    assert DrawFlg.DRAWING_SELF_SHADOWS in model.materials[7].draw_flg
    assert DrawFlg.DRAWING_EDGE not in model.materials[7].draw_flg
    assert np.isclose(
        np.array([0.2745098, 0.09019607, 0.1254902, 1]),
        model.materials[7].edge_color.vector,
    ).all()
    assert 1 == model.materials[7].edge_size
    assert 16 == model.materials[7].texture_index
    assert 1 == model.materials[7].sphere_texture_index
    # 下半身ボーン
    lower_bone = model.bones[4]
    assert "下半身" == lower_bone.name
    assert "J_Bip_C_Spine" == lower_bone.english_name
    assert np.isclose(
        np.array([0, 12.39097, -0.2011687]),
        lower_bone.position.vector,
    ).all()
    assert 3 == lower_bone.parent_index
    assert BoneFlg.TAIL_IS_BONE not in lower_bone.bone_flg
    assert BoneFlg.CAN_ROTATE in lower_bone.bone_flg
    assert BoneFlg.CAN_TRANSLATE not in lower_bone.bone_flg
    assert BoneFlg.IS_VISIBLE in lower_bone.bone_flg
    assert BoneFlg.CAN_MANIPULATE in lower_bone.bone_flg
    assert BoneFlg.IS_IK not in lower_bone.bone_flg
    assert BoneFlg.IS_EXTERNAL_LOCAL not in lower_bone.bone_flg
    assert BoneFlg.IS_EXTERNAL_ROTATION not in lower_bone.bone_flg
    assert BoneFlg.IS_EXTERNAL_TRANSLATION not in lower_bone.bone_flg
    assert BoneFlg.HAS_FIXED_AXIS not in lower_bone.bone_flg
    assert BoneFlg.HAS_LOCAL_COORDINATE not in lower_bone.bone_flg
    assert BoneFlg.IS_AFTER_PHYSICS_DEFORM not in lower_bone.bone_flg
    assert BoneFlg.IS_EXTERNAL_PARENT_DEFORM not in lower_bone.bone_flg
    assert np.isclose(
        np.array([8.003151e-31, -0.6508856, 0.1564678]),
        lower_bone.tail_position.vector,
    ).all()
    # 左足IKボーン
    left_leg_ik_bone = model.bones[98]
    assert "左足ＩＫ" == left_leg_ik_bone.name
    assert "leg_IK_L" == left_leg_ik_bone.english_name
    assert np.isclose(
        np.array([0.9644502, 1.647273, 0.4050385]),
        left_leg_ik_bone.position.vector,
    ).all()
    assert 97 == left_leg_ik_bone.parent_index
    assert BoneFlg.TAIL_IS_BONE not in left_leg_ik_bone.bone_flg
    assert BoneFlg.CAN_ROTATE in left_leg_ik_bone.bone_flg
    assert BoneFlg.CAN_TRANSLATE in left_leg_ik_bone.bone_flg
    assert BoneFlg.IS_VISIBLE in left_leg_ik_bone.bone_flg
    assert BoneFlg.CAN_MANIPULATE in left_leg_ik_bone.bone_flg
    assert BoneFlg.IS_IK in left_leg_ik_bone.bone_flg
    assert BoneFlg.IS_EXTERNAL_LOCAL not in left_leg_ik_bone.bone_flg
    assert BoneFlg.IS_EXTERNAL_ROTATION not in left_leg_ik_bone.bone_flg
    assert BoneFlg.IS_EXTERNAL_TRANSLATION not in left_leg_ik_bone.bone_flg
    assert BoneFlg.HAS_FIXED_AXIS not in left_leg_ik_bone.bone_flg
    assert BoneFlg.HAS_LOCAL_COORDINATE not in left_leg_ik_bone.bone_flg
    assert BoneFlg.IS_AFTER_PHYSICS_DEFORM not in left_leg_ik_bone.bone_flg
    assert BoneFlg.IS_EXTERNAL_PARENT_DEFORM not in left_leg_ik_bone.bone_flg
    assert np.isclose(
        np.array([0, 0, 1]),
        left_leg_ik_bone.tail_position.vector,
    ).all()
    assert 95 == left_leg_ik_bone.ik.bone_index
    assert 40 == left_leg_ik_bone.ik.loop_count
    assert np.isclose(
        np.array([57.29578, 0, 0]),
        left_leg_ik_bone.ik.unit_rotation.degrees.vector,
    ).all()
    assert 94 == left_leg_ik_bone.ik.links[0].bone_index
    assert left_leg_ik_bone.ik.links[0].angle_limit
    assert np.isclose(
        np.array([-180, 0, 0]),
        left_leg_ik_bone.ik.links[0].min_angle_limit.degrees.vector,
    ).all()
    assert np.isclose(
        np.array([-0.5, 0, 0]),
        left_leg_ik_bone.ik.links[0].max_angle_limit.degrees.vector,
    ).all()
    # 頂点モーフ
    surprised_morph = model.morphs[27]
    assert "びっくり" == surprised_morph.name
    assert "Fcl_EYE_Surprised" == surprised_morph.english_name
    assert MorphPanel.EYE_UPPER_LEFT == surprised_morph.panel
    assert MorphType.VERTEX == surprised_morph.morph_type
    assert 8185 == surprised_morph.offsets[29].vertex_index
    assert np.isclose(
        np.array([-0.0001545548, 0.002120972, -4.768372e-07]),
        surprised_morph.offsets[29].position_offset.vector,
    ).all()
    # 表示枠
    right_hand_display_slot = model.display_slots[9]
    assert "右指" == right_hand_display_slot.name
    assert "right hand" == right_hand_display_slot.english_name
    assert DisplayType.BONE == right_hand_display_slot.references[7].display_type
    assert 81 == right_hand_display_slot.references[7].display_index
    # 剛体
    left_knee_rigidbody = model.rigidbodies[20]
    assert "左ひざ" == left_knee_rigidbody.name
    assert "J_Bip_L_LowerLegUpper" == left_knee_rigidbody.english_name
    assert 94 == left_knee_rigidbody.bone_index
    assert RigidBodyMode.STATIC == left_knee_rigidbody.mode
    assert 1 == left_knee_rigidbody.collision_group
    assert RigidBodyCollisionGroup.GROUP01 not in left_knee_rigidbody.no_collision_group
    assert RigidBodyCollisionGroup.GROUP02 not in left_knee_rigidbody.no_collision_group
    assert RigidBodyCollisionGroup.GROUP03 not in left_knee_rigidbody.no_collision_group
    assert RigidBodyCollisionGroup.GROUP04 in left_knee_rigidbody.no_collision_group
    assert RigidBodyCollisionGroup.GROUP05 in left_knee_rigidbody.no_collision_group
    assert RigidBodyCollisionGroup.GROUP06 in left_knee_rigidbody.no_collision_group
    assert RigidBodyCollisionGroup.GROUP07 in left_knee_rigidbody.no_collision_group
    assert RigidBodyCollisionGroup.GROUP08 in left_knee_rigidbody.no_collision_group
    assert RigidBodyCollisionGroup.GROUP09 in left_knee_rigidbody.no_collision_group
    assert RigidBodyCollisionGroup.GROUP10 in left_knee_rigidbody.no_collision_group
    assert RigidBodyCollisionGroup.GROUP11 in left_knee_rigidbody.no_collision_group
    assert RigidBodyCollisionGroup.GROUP12 in left_knee_rigidbody.no_collision_group
    assert RigidBodyCollisionGroup.GROUP13 in left_knee_rigidbody.no_collision_group
    assert RigidBodyCollisionGroup.GROUP14 in left_knee_rigidbody.no_collision_group
    assert RigidBodyCollisionGroup.GROUP15 in left_knee_rigidbody.no_collision_group
    assert RigidBodyCollisionGroup.GROUP16 in left_knee_rigidbody.no_collision_group
    assert RigidBodyShape.CAPSULE == left_knee_rigidbody.shape_type
    assert np.isclose(
        np.array([0.6404971, 2.494654, 0]),
        left_knee_rigidbody.shape_size.vector,
    ).all()
    assert np.isclose(
        np.array([0.9472712, 5.526337, 0.4232892]),
        left_knee_rigidbody.shape_position.vector,
    ).all()
    assert np.isclose(
        np.array([-4.5407, 0.00, 0.00]),
        left_knee_rigidbody.shape_rotation.degrees.vector,
    ).all()
    assert 1 == left_knee_rigidbody.param.mass
    assert 0.5 == left_knee_rigidbody.param.linear_damping
    assert 0.5 == left_knee_rigidbody.param.angular_damping
    assert 0 == left_knee_rigidbody.param.restitution
    assert 0 == left_knee_rigidbody.param.friction
    # ジョイント
    left_bust_joint = model.joints[7]
    assert "Bst|左胸接続|左胸下" == left_bust_joint.name
    assert "Bst|LeftBustConnect|LeftBustLower" == left_bust_joint.english_name
    assert 34 == left_bust_joint.rigidbody_index_a
    assert 33 == left_bust_joint.rigidbody_index_b
    assert np.isclose(
        np.array([0.9984654, 14.25822, -1.298016]),
        left_bust_joint.position.vector,
    ).all()
    assert np.isclose(
        np.array([0, 0, 0]),
        left_bust_joint.rotation.degrees.vector,
    ).all()
    assert np.isclose(
        np.array([0, 0, 0]),
        left_bust_joint.param.translation_limit_min.vector,
    ).all()
    assert np.isclose(
        np.array([0, 0, 0]),
        left_bust_joint.param.translation_limit_max.vector,
    ).all()
    assert np.isclose(
        np.array([-15.1515, 0, 0]),
        left_bust_joint.param.rotation_limit_min.degrees.vector,
    ).all()
    assert np.isclose(
        np.array([15.1515, 0, 0]),
        left_bust_joint.param.rotation_limit_max.degrees.vector,
    ).all()
    assert np.isclose(
        np.array([0, 0, 0]),
        left_bust_joint.param.spring_constant_translation.vector,
    ).all()
    assert np.isclose(
        np.array([66.6667, 33.3333, 0]),
        left_bust_joint.param.spring_constant_rotation.vector,
    ).all()


if __name__ == "__main__":
    pytest.main()
