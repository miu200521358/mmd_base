from multiprocessing import freeze_support

import pytest

from mlib.core.math import MQuaternion, MVector3D


def test_read_by_filepath_error():
    import os

    from mlib.core.exception import MParseException
    from mlib.vmd.vmd_reader import VmdReader

    reader = VmdReader()
    with pytest.raises(MParseException):
        reader.read_by_filepath(os.path.join("tests", "resources", "サンプルモデル.pmx"))


def test_read_by_filepath_ok_calc() -> None:
    import os

    import numpy as np

    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    reader = VmdReader()
    motion: VmdMotion = reader.read_by_filepath(os.path.join("tests", "resources", "サンプルモーション.vmd"))
    # cSpell:disable
    assert "日本 roco式 トレス用" == motion.model_name
    # cSpell:enable

    # キーフレがある
    center_bf = motion.bones["センター"][358]
    assert 358 == center_bf.index
    assert np.isclose(
        np.array([1.094920158, 0, 0.100637913]),
        center_bf.position.vector,
    ).all()
    assert np.isclose(
        np.array([64, 0]),
        center_bf.interpolations.translation_x.start.vector,
    ).all()
    assert np.isclose(
        np.array([87, 87]),
        center_bf.interpolations.translation_x.end.vector,
    ).all()
    assert np.isclose(
        np.array([20, 20]),
        center_bf.interpolations.translation_y.start.vector,
    ).all()
    assert np.isclose(
        np.array([107, 107]),
        center_bf.interpolations.translation_y.end.vector,
    ).all()
    assert np.isclose(
        np.array([64, 0]),
        center_bf.interpolations.translation_z.start.vector,
    ).all()
    assert np.isclose(
        np.array([87, 87]),
        center_bf.interpolations.translation_z.end.vector,
    ).all()
    assert np.isclose(
        np.array([20, 20]),
        center_bf.interpolations.rotation.start.vector,
    ).all()
    assert np.isclose(
        np.array([107, 107]),
        center_bf.interpolations.rotation.end.vector,
    ).all()

    upper_bf = motion.bones["上半身"][689]
    assert 689 == upper_bf.index
    assert np.isclose(
        np.array([0, 0, 0]),
        upper_bf.position.vector,
    ).all()
    assert np.isclose(
        np.array([-6.270921156, -26.96361355, 0.63172903]),
        upper_bf.rotation.to_euler_degrees_mmd().vector,
    ).all()
    assert np.isclose(
        np.array([20, 20]),
        upper_bf.interpolations.rotation.start.vector,
    ).all()
    assert np.isclose(
        np.array([107, 107]),
        upper_bf.interpolations.rotation.end.vector,
    ).all()

    right_leg_ik_bf = motion.bones["右足ＩＫ"][384]
    assert 384 == right_leg_ik_bf.index
    assert np.isclose(
        np.array([0.548680067, 0.134522215, -2.504074097]),
        right_leg_ik_bf.position.vector,
    ).all()
    assert np.isclose(
        np.array([22.20309405, 6.80959631, 2.583712695]),
        right_leg_ik_bf.rotation.to_euler_degrees_mmd().vector,
    ).all()
    assert np.isclose(
        np.array([64, 0]),
        right_leg_ik_bf.interpolations.translation_x.start.vector,
    ).all()
    assert np.isclose(
        np.array([64, 127]),
        right_leg_ik_bf.interpolations.translation_x.end.vector,
    ).all()
    assert np.isclose(
        np.array([64, 0]),
        right_leg_ik_bf.interpolations.translation_y.start.vector,
    ).all()
    assert np.isclose(
        np.array([87, 87]),
        right_leg_ik_bf.interpolations.translation_y.end.vector,
    ).all()
    assert np.isclose(
        np.array([64, 0]),
        right_leg_ik_bf.interpolations.translation_z.start.vector,
    ).all()
    assert np.isclose(
        np.array([64, 127]),
        right_leg_ik_bf.interpolations.translation_z.end.vector,
    ).all()
    assert np.isclose(
        np.array([64, 0]),
        right_leg_ik_bf.interpolations.rotation.start.vector,
    ).all()
    assert np.isclose(
        np.array([87, 87]),
        right_leg_ik_bf.interpolations.rotation.end.vector,
    ).all()

    # キーがないフレーム
    left_leg_ik_bf = motion.bones["左足ＩＫ"][384]
    assert 384 == left_leg_ik_bf.index
    assert np.isclose(
        np.array([-1.63, 0.05, 2.58]),
        left_leg_ik_bf.position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-1.4, 6.7, -5.2]),
        left_leg_ik_bf.rotation.to_euler_degrees_mmd().vector,
        rtol=0.1,
        atol=0.1,
    ).all()

    left_leg_ik_bf = motion.bones["左足ＩＫ"][394]
    assert 394 == left_leg_ik_bf.index
    assert np.isclose(
        np.array([0.76, 1.17, 1.34]),
        left_leg_ik_bf.position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-41.9, -1.6, 1.0]),
        left_leg_ik_bf.rotation.to_euler_degrees_mmd().vector,
        rtol=0.1,
        atol=0.1,
    ).all()

    left_leg_ik_bf = motion.bones["左足ＩＫ"][412]
    assert 412 == left_leg_ik_bf.index
    assert np.isclose(
        np.array([-0.76, -0.61, -1.76]),
        left_leg_ik_bf.position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([43.1, 0.0, 0.0]),
        left_leg_ik_bf.rotation.to_euler_degrees_mmd().vector,
        rtol=0.1,
        atol=0.1,
    ).all()

    left_arm_bf = motion.bones["左腕"][384]
    assert 384 == left_arm_bf.index
    assert np.isclose(
        np.array([13.5, -4.3, 27.0]),
        left_arm_bf.rotation.to_euler_degrees_mmd().vector,
        rtol=0.1,
        atol=0.1,
    ).all()


def test_read_by_filepath_ok_matrix() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(os.path.join("tests", "resources", "サンプルモーション.vmd"))

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(os.path.join("tests", "resources", "サンプルモデル.pmx"))

    # キーフレ
    bone_matrixes = motion.animate_bone([10, 999], model, ["左人指３"])

    assert np.isclose(
        np.array([0, 0, 0]),
        bone_matrixes[10, "全ての親"].position.vector,
    ).all()
    assert np.isclose(
        np.array([0.044920, 8.218059, 0.069347]),
        bone_matrixes[10, "センター"].position.vector,
    ).all()
    assert np.isclose(
        np.array([0.044920, 9.392067, 0.064877]),
        bone_matrixes[10, "グルーブ"].position.vector,
    ).all()

    assert np.isclose(
        np.array([0.044920, 11.740084, 0.055937]),
        bone_matrixes[10, "腰"].position.vector,
    ).all()
    assert np.isclose(
        np.array([0.044920, 12.390969, -0.100531]),
        bone_matrixes[10, "上半身"].position.vector,
    ).all()
    assert np.isclose(
        np.array([0.044920, 13.803633, -0.138654]),
        bone_matrixes[10, "上半身2"].position.vector,
    ).all()
    assert np.isclose(
        np.array([0.044920, 15.149180, 0.044429]),
        bone_matrixes[10, "上半身3"].position.vector,
    ).all()
    assert np.isclose(
        np.array([0.324862, 16.470263, 0.419041]),
        bone_matrixes[10, "左肩P"].position.vector,
    ).all()
    assert np.isclose(
        np.array([0.324862, 16.470263, 0.419041]),
        bone_matrixes[10, "左肩"].position.vector,
    ).all()
    assert np.isclose(
        np.array([1.369838, 16.312170, 0.676838]),
        bone_matrixes[10, "左腕"].position.vector,
    ).all()
    assert np.isclose(
        np.array([1.845001, 15.024807, 0.747681]),
        bone_matrixes[10, "左腕捩"].position.vector,
    ).all()
    assert np.isclose(
        np.array([2.320162, 13.737446, 0.818525]),
        bone_matrixes[10, "左ひじ"].position.vector,
    ).all()
    assert np.isclose(
        np.array([2.516700, 12.502447, 0.336127]),
        bone_matrixes[10, "左手捩"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.732219, 11.267447, -0.146273]),
        bone_matrixes[10, "左手首"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.649188, 10.546797, -0.607412]),
        bone_matrixes[10, "左人指１"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.408238, 10.209290, -0.576288]),
        bone_matrixes[10, "左人指２"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.360455, 10.422402, -0.442668]),
        bone_matrixes[10, "左人指３"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()

    # --------

    # キーフレがない場合
    assert np.isclose(
        np.array([0, 0, 0]),
        bone_matrixes[999, "全ての親"].position.vector,
    ).all()
    assert np.isclose(
        np.array([-0.508560, 8.218059, 0.791827]),
        bone_matrixes[999, "センター"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.508560, 9.182008, 0.787357]),
        bone_matrixes[999, "グルーブ"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()

    assert np.isclose(
        np.array([-0.508560, 11.530025, 0.778416]),
        bone_matrixes[999, "腰"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.508560, 12.180910, 0.621949]),
        bone_matrixes[999, "上半身"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.437343, 13.588836, 0.523215]),
        bone_matrixes[999, "上半身2"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    # 付与親
    assert np.isclose(
        np.array([-0.552491, 14.941880, 0.528703]),
        bone_matrixes[999, "上半身3"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.590927, 16.312325, 0.819156]),
        bone_matrixes[999, "左肩P"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.590927, 16.312325, 0.819156]),
        bone_matrixes[999, "左肩"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.072990, 16.156742, 1.666761]),
        bone_matrixes[999, "左腕"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.043336, 15.182318, 2.635117]),
        bone_matrixes[999, "左腕捩"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.013682, 14.207894, 3.603473]),
        bone_matrixes[999, "左ひじ"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.222444, 13.711100, 3.299384]),
        bone_matrixes[999, "左手捩"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.431205, 13.214306, 2.995294]),
        bone_matrixes[999, "左手首"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([3.283628, 13.209089, 2.884702]),
        bone_matrixes[999, "左人指１"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([3.665809, 13.070156, 2.797680]),
        bone_matrixes[999, "左人指２"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([3.886795, 12.968100, 2.718276]),
        bone_matrixes[999, "左人指３"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_matrix_animate() -> None:
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_reader import VmdReader

    motion = VmdReader().read_by_filepath("D:/MMD/MikuMikuDance_v926x64/UserFile/Motion/ダンス_1人/テレキャスタービーボーイ 粉ふきスティック/TeBeboy.vmd")
    model = PmxReader().read_by_filepath("D:/MMD/MikuMikuDance_v926x64/UserFile/Model/VOCALOID/初音ミク/ISAO式ミク/I_ミクv4/Miku_V4_準標準.pmx")

    # キーフレ
    gl_matrixes, _, _, _, _, _ = motion.animate(999, model)

    assert gl_matrixes is not None


def test_read_by_filepath_ok_matrix_morph() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_part import BoneMorphOffset, Morph, MorphType
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_part import VmdMorphFrame
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(os.path.join("tests", "resources", "サンプルモーション.vmd"))

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(os.path.join("tests", "resources", "サンプルモデル.pmx"))

    # モーフ追加
    morph = Morph(name="底上げ")
    morph.morph_type = MorphType.BONE
    morph.offsets.append(BoneMorphOffset(0, MVector3D(0, 1, 0), MQuaternion()))
    model.morphs.append(morph)

    motion.morphs["底上げ"].append(VmdMorphFrame(0, "底上げ", 1))

    # キーフレ
    bone_matrixes = motion.animate_bone([10, 999], model, append_ik=False)

    # キーフレがない場合
    assert np.isclose(
        np.array([0, 1, 0]),
        bone_matrixes[999, "全ての親"].position.vector,
    ).all()
    assert np.isclose(
        np.array([-0.508560, 9.218059, 0.791827]),
        bone_matrixes[999, "センター"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.508560, 10.182008, 0.787357]),
        bone_matrixes[999, "グルーブ"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()

    assert np.isclose(
        np.array([-0.508560, 12.530025, 0.778416]),
        bone_matrixes[999, "腰"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.508560, 13.180910, 0.621949]),
        bone_matrixes[999, "上半身"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.437343, 14.588836, 0.523215]),
        bone_matrixes[999, "上半身2"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    # 付与親
    assert np.isclose(
        np.array([-0.552491, 15.941880, 0.528703]),
        bone_matrixes[999, "上半身3"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.590927, 17.312325, 0.819156]),
        bone_matrixes[999, "左肩P"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.590927, 17.312325, 0.819156]),
        bone_matrixes[999, "左肩"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.072990, 17.156742, 1.666761]),
        bone_matrixes[999, "左腕"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.043336, 16.182318, 2.635117]),
        bone_matrixes[999, "左腕捩"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.013682, 15.207894, 3.603473]),
        bone_matrixes[999, "左ひじ"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.222444, 14.711100, 3.299384]),
        bone_matrixes[999, "左手捩"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.431205, 14.214306, 2.995294]),
        bone_matrixes[999, "左手首"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([3.283628, 14.209089, 2.884702]),
        bone_matrixes[999, "左人指１"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([3.665809, 14.070156, 2.797680]),
        bone_matrixes[999, "左人指２"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([3.886795, 13.968100, 2.718276]),
        bone_matrixes[999, "左人指３"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_matrix_local_morph() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_part import BoneMorphOffset, Morph, MorphType
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_part import VmdMorphFrame
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(os.path.join("tests", "resources", "サンプルモーション.vmd"))

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(os.path.join("tests", "resources", "サンプルモデル.pmx"))

    # モーフ追加
    morph = Morph(name="底上げ")
    morph.morph_type = MorphType.BONE
    morph.offsets.append(BoneMorphOffset(model.bones["センター"].index, MVector3D(), MQuaternion(), local_position=MVector3D(1, 0, 0)))
    model.morphs.append(morph)

    motion.morphs["底上げ"].append(VmdMorphFrame(0, "底上げ", 1))

    # キーフレ
    bone_matrixes = motion.animate_bone([10, 999], model, ["左人指３"])

    assert np.isclose(
        np.array([0, 0, 0]),
        bone_matrixes[10, "全ての親"].position.vector,
    ).all()
    assert np.isclose(
        np.array([0.044920, 9.218059, 0.069347]),
        bone_matrixes[10, "センター"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.044920, 9.392067, 0.064877]),
        bone_matrixes[10, "グルーブ"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.044920, 11.740084, 0.055937]),
        bone_matrixes[10, "腰"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.044920, 12.390969, -0.100531]),
        bone_matrixes[10, "上半身"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.044920, 13.803633, -0.138654]),
        bone_matrixes[10, "上半身2"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.044920, 15.149180, 0.044429]),
        bone_matrixes[10, "上半身3"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.324862, 16.470263, 0.419041]),
        bone_matrixes[10, "左肩P"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.324862, 16.470263, 0.419041]),
        bone_matrixes[10, "左肩"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.369838, 16.312170, 0.676838]),
        bone_matrixes[10, "左腕"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.845001, 15.024807, 0.747681]),
        bone_matrixes[10, "左腕捩"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.320162, 13.737446, 0.818525]),
        bone_matrixes[10, "左ひじ"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.516700, 12.502447, 0.336127]),
        bone_matrixes[10, "左手捩"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.732219, 11.267447, -0.146273]),
        bone_matrixes[10, "左手首"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.649188, 10.546797, -0.607412]),
        bone_matrixes[10, "左人指１"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.408238, 10.209290, -0.576288]),
        bone_matrixes[10, "左人指２"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.360455, 10.422402, -0.442668]),
        bone_matrixes[10, "左人指３"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_leg_ik1() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(os.path.join("tests", "resources", "サンプルモーション.vmd"))

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(os.path.join("tests", "resources", "サンプルモデル.pmx"))

    # キーフレ
    bone_matrixes = motion.animate_bone([29], model, ["左つま先"])

    # --------
    # キーフレがある場合

    assert np.isclose(
        np.array([-0.781335, 11.717622, 1.557067]),
        bone_matrixes[29, "下半身"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.368843, 10.614175, 2.532657]),
        bone_matrixes[29, "左足"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.983212, 6.945313, 0.487476]),
        bone_matrixes[29, "左ひざ"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.345842, 2.211842, 2.182894]),
        bone_matrixes[29, "左足首"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.109262, -0.025810, 1.147780]),
        bone_matrixes[29, "左つま先"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_leg_ik2() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(os.path.join("tests", "resources", "サンプルモーション.vmd"))

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(os.path.join("tests", "resources", "サンプルモデル.pmx"))

    # キーフレ
    bone_matrixes = motion.animate_bone([3152], model)

    # --------
    # キーフレがない場合

    assert np.isclose(
        np.array([7.928583, 11.713336, 1.998830]),
        bone_matrixes[3152, "下半身"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([7.370017, 10.665785, 2.963280]),
        bone_matrixes[3152, "左足"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([9.282883, 6.689319, 2.96825]),
        bone_matrixes[3152, "左ひざ"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([4.115521, 7.276527, 2.980609]),
        bone_matrixes[3152, "左足首"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.931355, 6.108739, 2.994883]),
        bone_matrixes[3152, "左つま先"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_leg_ik3() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(os.path.join("tests", "resources", "サンプルモーション2.vmd"))

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(os.path.join("tests", "resources", "サンプルモデル.pmx"))

    # キーフレ
    bone_matrixes = motion.animate_bone([60], model)

    # --------
    # キーフレがない場合

    assert np.isclose(
        np.array([1.931959, 11.695199, -1.411883]),
        bone_matrixes[60, "下半身"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.927524, 10.550287, -1.218106]),
        bone_matrixes[60, "左足"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.263363, 7.061642, -3.837192]),
        bone_matrixes[60, "左ひざ"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.747242, 2.529942, -1.331971]),
        bone_matrixes[60, "左足首"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.809291, 0.242514, -1.182168]),
        bone_matrixes[60, "左つま先"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.263363, 7.061642, -3.837192]),
        bone_matrixes[60, "左ひざD"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.916109, 1.177077, -1.452845]),
        bone_matrixes[60, "左足先EX"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_leg_ik4() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    # 好き雪 2794F
    motion: VmdMotion = vmd_reader.read_by_filepath(os.path.join("tests", "resources", "サンプルモーション3.vmd"))

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(os.path.join("tests", "resources", "サンプルモデル.pmx"))

    # キーフレ
    bone_matrixes = motion.animate_bone([0], model)

    # --------
    # キーフレがある場合

    assert np.isclose(
        np.array([1.316121, 11.687257, 2.263307]),
        bone_matrixes[0, "下半身"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.175478, 10.780540, 2.728409]),
        bone_matrixes[0, "右足"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.950410, 11.256771, -1.589462]),
        bone_matrixes[0, "右ひざ"].position.vector,
        rtol=0.2,
        atol=0.2,
    ).all()
    assert np.isclose(
        np.array([-1.025194, 7.871110, 1.828258]),
        bone_matrixes[0, "右足首"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-1.701147, 6.066556, 3.384271]),
        bone_matrixes[0, "右つま先"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_leg_ik5() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(os.path.join("tests", "resources", "サンプルモーション2.vmd"))

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(os.path.join("tests", "resources", "サンプルモデル.pmx"))

    # キーフレ
    bone_matrixes = motion.animate_bone([7409], model)

    # --------
    # 最後を越したキーフレ

    assert np.isclose(
        np.array([-7.652257, 11.990970, -4.511993]),
        bone_matrixes[7409, "下半身"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-8.637265, 10.835548, -4.326830]),
        bone_matrixes[7409, "右足"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-8.693436, 7.595280, -7.321638]),
        bone_matrixes[7409, "右ひざ"].position.vector,
        rtol=0.3,
        atol=0.3,
    ).all()
    assert np.isclose(
        np.array([-7.521027, 2.827226, -9.035607]),
        bone_matrixes[7409, "右足首"].position.vector,
        rtol=0.3,
        atol=0.3,
    ).all()
    assert np.isclose(
        np.array([-7.453236, 0.356456, -8.876783]),
        bone_matrixes[7409, "右つま先"].position.vector,
        rtol=0.3,
        atol=0.3,
    ).all()


def test_read_by_filepath_ok_arm_ik() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(os.path.join("tests", "resources", "サンプルモーション.vmd"))

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(os.path.join("tests", "resources", "ボーンツリーテストモデル.pmx"))

    # キーフレ
    bone_matrixes = motion.animate_bone([10, 999], model)

    assert np.isclose(
        np.array([0, 0, 0]),
        bone_matrixes[10, "全ての親"].position.vector,
    ).all()
    assert np.isclose(
        np.array([0.044920, 9.000000, -0.199362]),
        bone_matrixes[10, "センター"].position.vector,
    ).all()
    assert np.isclose(
        np.array([0.044920, 9.410000, -0.199362]),
        bone_matrixes[10, "グルーブ"].position.vector,
    ).all()

    assert np.isclose(
        np.array([0.044920, 12.458570, 0.368584]),
        bone_matrixes[10, "腰"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.044920, 13.397310, -0.855492]),
        bone_matrixes[10, "上半身"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.044920, 14.613530, -0.791352]),
        bone_matrixes[10, "上半身2"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.329451, 16.681561, -0.348142]),
        bone_matrixes[10, "左肩P"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.329451, 16.681561, -0.348142]),
        bone_matrixes[10, "左肩"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.290706, 16.678047, -0.133773]),
        bone_matrixes[10, "左腕YZ"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.611640, 15.785284, -0.086812]),
        bone_matrixes[10, "左腕捩YZ"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.611641, 15.785284, -0.086812]),
        bone_matrixes[10, "左腕捩X"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.321083, 13.811781, 0.016998]),
        bone_matrixes[10, "左ひじYZ"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.414732, 13.218668, -0.214754]),
        bone_matrixes[10, "左手捩YZ"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.414731, 13.218668, -0.214755]),
        bone_matrixes[10, "左手捩X"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.620196, 11.955698, -0.693675]),
        bone_matrixes[10, "左手捩6"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.691600, 11.503933, -0.870235]),
        bone_matrixes[10, "左手首R"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.633156, 11.364628, -0.882837]),
        bone_matrixes[10, "左手首2"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.473304, 10.728573, -1.304400]),
        bone_matrixes[10, "左人指１"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.261877, 10.458740, -1.299257]),
        bone_matrixes[10, "左人指２"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.228296, 10.653198, -1.178544]),
        bone_matrixes[10, "左人指３"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.417647, 10.880006, -1.173676]),
        bone_matrixes[10, "左人指先"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()


def test_vmd_save_01():
    import os

    from mlib.vmd.vmd_reader import VmdReader
    from mlib.vmd.vmd_writer import VmdWriter

    input_path = os.path.join("tests", "resources", "サンプルモーション.vmd")
    motion = VmdReader().read_by_filepath(input_path)
    output_path = os.path.join("tests", "resources", "result.vmd")
    VmdWriter(motion, output_path, motion.name).save()

    with open(output_path, "rb") as f:
        output_motion = f.read()

    assert output_motion


# def test_vmd_save_02():
#     from mlib.pmx.pmx_part import STANDARD_BONE_NAMES
#     from mlib.pmx.pmx_collection import PmxModel
#     from mlib.pmx.pmx_reader import PmxReader
#     from mlib.vmd.vmd_part import VmdBoneFrame
#     from mlib.vmd.vmd_collection import VmdMotion
#     from mlib.vmd.vmd_reader import VmdReader
#     from mlib.vmd.vmd_writer import VmdWriter

#     vmd_reader = VmdReader()
#     motion: VmdMotion = vmd_reader.read_by_filepath(
#         "D:/MMD/MikuMikuDance_v926x64/UserFile/Motion/ダンス_1人/カミサマネジマキ 粉ふきスティック/sora_guiter1_2016.vmd"
#     )

#     pmx_reader = PmxReader()
#     model: PmxModel = pmx_reader.read_by_filepath(
#         "D:/MMD/MikuMikuDance_v926x64/UserFile/Model/VOCALOID/弱音ハク/Tda式改変弱音ハクccvセット1.05 coa/
# Tda式改変弱音ハクccv_nc 1.05/ギター用/Tda式改変弱音ハクccv_nc ver1.05_ギター.pmx"
#     )

#     fno = 3100
#     new_motion = VmdMotion()
#     poses, qqs, scales, local_poses, local_qqs, local_scales = motion.bones.get_bone_matrixes(
#         [fno], model, list(set(STANDARD_BONE_NAMES.keys()) & set(model.bones.names))
#     )
#     for bidx, bone_name in enumerate(STANDARD_BONE_NAMES.keys()):
#         if bone_name not in model.bones:
#             continue
#         new_bf = VmdBoneFrame(index=fno, name=bone_name)
#         new_bf.position = MMatrix4x4(poses[0, bidx]).to_position()
#         new_bf.rotation = MMatrix4x4(qqs[0, bidx]).to_quaternion()
#         new_motion.bones[bone_name].append(new_bf)
#         print(f"{bone_name}: {fno}")

#     VmdWriter(
#         new_motion, "D:/MMD/MikuMikuDance_v926x64/UserFile/Motion/ダンス_1人/カミサマネジマキ 粉ふきスティック/sora_guiter1_2016_焼き込み.vmd", "ハク腕焼き込み"
#     ).save()

#     assert True


if __name__ == "__main__":
    freeze_support()
    pytest.main()
