from multiprocessing import freeze_support

import pytest

from mlib.core.math import MQuaternion, MVector2D, MVector3D
from mlib.vmd.vmd_part import VmdBoneFrame


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
    motion: VmdMotion = reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモーション.vmd")
    )
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
        upper_bf.rotation.to_euler_degrees().mmd.vector,
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
        right_leg_ik_bf.rotation.to_euler_degrees().mmd.vector,
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
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-1.4, 6.7, -5.2]),
        left_leg_ik_bf.rotation.to_euler_degrees().mmd.vector,
        atol=0.05,
    ).all()

    left_leg_ik_bf = motion.bones["左足ＩＫ"][394]
    assert 394 == left_leg_ik_bf.index
    assert np.isclose(
        np.array([0.76, 1.17, 1.34]),
        left_leg_ik_bf.position.vector,
        atol=0.05,
    ).all()
    assert np.isclose(
        np.array([-41.9, -1.6, 1.0]),
        left_leg_ik_bf.rotation.to_euler_degrees().mmd.vector,
        atol=0.05,
    ).all()

    left_leg_ik_bf = motion.bones["左足ＩＫ"][412]
    assert 412 == left_leg_ik_bf.index
    assert np.isclose(
        np.array([-0.76, -0.61, -1.76]),
        left_leg_ik_bf.position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([43.1, 0.0, 0.0]),
        left_leg_ik_bf.rotation.to_euler_degrees().mmd.vector,
        atol=0.05,
    ).all()

    left_arm_bf = motion.bones["左腕"][384]
    assert 384 == left_arm_bf.index
    assert np.isclose(
        np.array([13.5, -4.3, 27.0]),
        left_arm_bf.rotation.to_euler_degrees().mmd.vector,
        atol=0.05,
    ).all()


def test_read_by_filepath_ok_matrix() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモーション.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモデル.pmx")
    )

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
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.732219, 11.267447, -0.146273]),
        bone_matrixes[10, "左手首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.649188, 10.546797, -0.607412]),
        bone_matrixes[10, "左人指１"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.408238, 10.209290, -0.576288]),
        bone_matrixes[10, "左人指２"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.360455, 10.422402, -0.442668]),
        bone_matrixes[10, "左人指３"].position.vector,
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
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.508560, 9.182008, 0.787357]),
        bone_matrixes[999, "グルーブ"].position.vector,
        atol=0.01,
    ).all()

    assert np.isclose(
        np.array([-0.508560, 11.530025, 0.778416]),
        bone_matrixes[999, "腰"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.508560, 12.180910, 0.621949]),
        bone_matrixes[999, "上半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.437343, 13.588836, 0.523215]),
        bone_matrixes[999, "上半身2"].position.vector,
        atol=0.01,
    ).all()
    # 付与親
    assert np.isclose(
        np.array([-0.552491, 14.941880, 0.528703]),
        bone_matrixes[999, "上半身3"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.590927, 16.312325, 0.819156]),
        bone_matrixes[999, "左肩P"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.590927, 16.312325, 0.819156]),
        bone_matrixes[999, "左肩"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.072990, 16.156742, 1.666761]),
        bone_matrixes[999, "左腕"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.043336, 15.182318, 2.635117]),
        bone_matrixes[999, "左腕捩"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.013682, 14.207894, 3.603473]),
        bone_matrixes[999, "左ひじ"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.222444, 13.711100, 3.299384]),
        bone_matrixes[999, "左手捩"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.431205, 13.214306, 2.995294]),
        bone_matrixes[999, "左手首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([3.283628, 13.209089, 2.884702]),
        bone_matrixes[999, "左人指１"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([3.665809, 13.070156, 2.797680]),
        bone_matrixes[999, "左人指２"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([3.886795, 12.968100, 2.718276]),
        bone_matrixes[999, "左人指３"].position.vector,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_matrix_animate() -> None:
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_reader import VmdReader

    motion = VmdReader().read_by_filepath(
        "D:/MMD/MikuMikuDance_v926x64/UserFile/Motion/ダンス_1人/テレキャスタービーボーイ 粉ふきスティック/TeBeboy.vmd"
    )
    model = PmxReader().read_by_filepath(
        "D:/MMD/MikuMikuDance_v926x64/UserFile/Model/VOCALOID/初音ミク/ISAO式ミク/I_ミクv4/Miku_V4_準標準.pmx"
    )

    # キーフレ
    _, gl_matrixes, _, _, _, _, _, _ = motion.animate(999, model)

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
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモーション.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモデル.pmx")
    )

    # モーフ追加
    morph = Morph(name="底上げ")
    morph.morph_type = MorphType.BONE
    morph.offsets.append(BoneMorphOffset(0, MVector3D(0, 1, 0), MQuaternion()))
    model.morphs.append(morph)

    motion.morphs["底上げ"].append(VmdMorphFrame(0, "底上げ", 1))

    # キーフレ
    bone_matrixes = motion.animate_bone([10, 999], model, is_calc_ik=False)

    # キーフレがない場合
    assert np.isclose(
        np.array([0, 1, 0]),
        bone_matrixes[999, "全ての親"].position.vector,
    ).all()
    assert np.isclose(
        np.array([-0.508560, 9.218059, 0.791827]),
        bone_matrixes[999, "センター"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.508560, 10.182008, 0.787357]),
        bone_matrixes[999, "グルーブ"].position.vector,
        atol=0.01,
    ).all()

    assert np.isclose(
        np.array([-0.508560, 12.530025, 0.778416]),
        bone_matrixes[999, "腰"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.508560, 13.180910, 0.621949]),
        bone_matrixes[999, "上半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.437343, 14.588836, 0.523215]),
        bone_matrixes[999, "上半身2"].position.vector,
        atol=0.01,
    ).all()
    # 付与親
    assert np.isclose(
        np.array([-0.552491, 15.941880, 0.528703]),
        bone_matrixes[999, "上半身3"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.590927, 17.312325, 0.819156]),
        bone_matrixes[999, "左肩P"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.590927, 17.312325, 0.819156]),
        bone_matrixes[999, "左肩"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.072990, 17.156742, 1.666761]),
        bone_matrixes[999, "左腕"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.043336, 16.182318, 2.635117]),
        bone_matrixes[999, "左腕捩"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.013682, 15.207894, 3.603473]),
        bone_matrixes[999, "左ひじ"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.222444, 14.711100, 3.299384]),
        bone_matrixes[999, "左手捩"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.431205, 14.214306, 2.995294]),
        bone_matrixes[999, "左手首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([3.283628, 14.209089, 2.884702]),
        bone_matrixes[999, "左人指１"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([3.665809, 14.070156, 2.797680]),
        bone_matrixes[999, "左人指２"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([3.886795, 13.968100, 2.718276]),
        bone_matrixes[999, "左人指３"].position.vector,
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
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモーション.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモデル.pmx")
    )

    # モーフ追加
    morph = Morph(name="底上げ")
    morph.morph_type = MorphType.BONE
    morph.offsets.append(
        BoneMorphOffset(
            model.bones["センター"].index,
            MVector3D(),
            MQuaternion(),
            local_position=MVector3D(1, 0, 0),
        )
    )
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
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.044920, 9.392067, 0.064877]),
        bone_matrixes[10, "グルーブ"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.044920, 11.740084, 0.055937]),
        bone_matrixes[10, "腰"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.044920, 12.390969, -0.100531]),
        bone_matrixes[10, "上半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.044920, 13.803633, -0.138654]),
        bone_matrixes[10, "上半身2"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.044920, 15.149180, 0.044429]),
        bone_matrixes[10, "上半身3"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.324862, 16.470263, 0.419041]),
        bone_matrixes[10, "左肩P"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.324862, 16.470263, 0.419041]),
        bone_matrixes[10, "左肩"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.369838, 16.312170, 0.676838]),
        bone_matrixes[10, "左腕"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.845001, 15.024807, 0.747681]),
        bone_matrixes[10, "左腕捩"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.320162, 13.737446, 0.818525]),
        bone_matrixes[10, "左ひじ"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.516700, 12.502447, 0.336127]),
        bone_matrixes[10, "左手捩"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.732219, 11.267447, -0.146273]),
        bone_matrixes[10, "左手首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.649188, 10.546797, -0.607412]),
        bone_matrixes[10, "左人指１"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.408238, 10.209290, -0.576288]),
        bone_matrixes[10, "左人指２"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.360455, 10.422402, -0.442668]),
        bone_matrixes[10, "左人指３"].position.vector,
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
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモーション.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモデル.pmx")
    )

    # キーフレ
    bone_matrixes = motion.animate_bone([29], model, ["左つま先"])

    # --------
    # キーフレがある場合

    assert np.isclose(
        np.array([-0.781335, 11.717622, 1.557067]),
        bone_matrixes[29, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.368843, 10.614175, 2.532657]),
        bone_matrixes[29, "左足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.983212, 6.945313, 0.487476]),
        bone_matrixes[29, "左ひざ"].position.vector,
        atol=0.02,
    ).all()
    assert np.isclose(
        np.array([-0.345842, 2.211842, 2.182894]),
        bone_matrixes[29, "左足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.109262, -0.025810, 1.147780]),
        bone_matrixes[29, "左つま先"].position.vector,
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
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモーション.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモデル.pmx")
    )

    # キーフレ
    bone_matrixes = motion.animate_bone([3152], model)

    # --------
    # キーフレがない場合

    assert np.isclose(
        np.array([7.928583, 11.713336, 1.998830]),
        bone_matrixes[3152, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([7.370017, 10.665785, 2.963280]),
        bone_matrixes[3152, "左足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([9.282883, 6.689319, 2.96825]),
        bone_matrixes[3152, "左ひざ"].position.vector,
        atol=0.02,
    ).all()
    assert np.isclose(
        np.array([4.115521, 7.276527, 2.980609]),
        bone_matrixes[3152, "左足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.931355, 6.108739, 2.994883]),
        bone_matrixes[3152, "左つま先"].position.vector,
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
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモーション2.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモデル.pmx")
    )

    # キーフレ
    bone_matrixes = motion.animate_bone([60], model)

    # --------
    # キーフレがない場合

    assert np.isclose(
        np.array([1.931959, 11.695199, -1.411883]),
        bone_matrixes[60, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.927524, 10.550287, -1.218106]),
        bone_matrixes[60, "左足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.263363, 7.061642, -3.837192]),
        bone_matrixes[60, "左ひざ"].position.vector,
        atol=0.02,
    ).all()
    assert np.isclose(
        np.array([2.747242, 2.529942, -1.331971]),
        bone_matrixes[60, "左足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.809291, 0.242514, -1.182168]),
        bone_matrixes[60, "左つま先"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.263363, 7.061642, -3.837192]),
        bone_matrixes[60, "左ひざD"].position.vector,
        atol=0.02,
    ).all()
    assert np.isclose(
        np.array([1.916109, 1.177077, -1.452845]),
        bone_matrixes[60, "左足先EX"].position.vector,
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
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモーション3.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモデル.pmx")
    )

    # キーフレ
    bone_matrixes = motion.animate_bone([0], model, ["右つま先"])

    # --------
    # キーフレがある場合

    assert np.isclose(
        np.array([1.316121, 11.687257, 2.263307]),
        bone_matrixes[0, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.175478, 10.780540, 2.728409]),
        bone_matrixes[0, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.950410, 11.256771, -1.589462]),
        bone_matrixes[0, "右ひざ"].position.vector,
        atol=0.4,
    ).all()
    assert np.isclose(
        np.array([-1.025194, 7.871110, 1.828258]),
        bone_matrixes[0, "右足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-1.701147, 6.066556, 3.384271]),
        bone_matrixes[0, "右つま先"].position.vector,
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
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモーション2.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモデル.pmx")
    )

    # キーフレ
    bone_matrixes = motion.animate_bone([7409], model)

    # --------
    # 最後を越したキーフレ

    assert np.isclose(
        np.array([-7.652257, 11.990970, -4.511993]),
        bone_matrixes[7409, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-8.637265, 10.835548, -4.326830]),
        bone_matrixes[7409, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-8.693436, 7.595280, -7.321638]),
        bone_matrixes[7409, "右ひざ"].position.vector,
        atol=0.02,
    ).all()
    assert np.isclose(
        np.array([-7.521027, 2.827226, -9.035607]),
        bone_matrixes[7409, "右足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-7.453236, 0.356456, -8.876783]),
        bone_matrixes[7409, "右つま先"].position.vector,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_leg_ik6_snow() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    # 好き雪
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモーション2.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモデル.pmx")
    )

    # --------
    # IK ON

    # キーフレ
    bone_on_matrixes = motion.animate_bone([0], model, clear_ik=True)

    assert np.isclose(
        np.array([2.143878, 6.558880, 1.121747]),
        bone_on_matrixes[0, "左ひざ"].position.vector,
        atol=0.02,
    ).all()

    assert np.isclose(
        np.array([2.214143, 1.689811, 2.947619]),
        bone_on_matrixes[0, "左足首"].position.vector,
        atol=0.01,
    ).all()

    # --------
    # IK OFF

    # キーフレ
    bone_off_matrixes = motion.animate_bone([0], model, is_calc_ik=False, clear_ik=True)

    assert np.isclose(
        np.array([1.622245, 6.632885, 0.713205]),
        bone_off_matrixes[0, "左ひざ"].position.vector,
        atol=0.02,
    ).all()

    assert np.isclose(
        np.array([1.003185, 1.474691, 0.475763]),
        bone_off_matrixes[0, "左足首"].position.vector,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_leg_ik7_syou() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "唱(ダンスのみ)_0278F.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモデル.pmx")
    )

    # キーフレ
    bone_matrixes = motion.animate_bone([0], model, ["右つま先"])

    # --------
    # 残存回転判定用

    assert np.isclose(
        np.array([0.721499, 11.767294, 1.638818]),
        bone_matrixes[0, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.133304, 10.693992, 2.314730]),
        bone_matrixes[0, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-2.833401, 8.174604, -0.100545]),
        bone_matrixes[0, "右ひざ"].position.vector,
        atol=2.5,
    ).all()
    assert np.isclose(
        np.array([-0.409387, 5.341005, 3.524572]),
        bone_matrixes[0, "右足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.578271, 2.874233, 3.669599]),
        bone_matrixes[0, "右つま先"].position.vector,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_leg_ik7_animate() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "シャイニングミラクル_50F.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモデル.pmx")
    )

    # キーフレ
    (_, _, bone_matrixes, _, _, _, _, _) = motion.animate(0, model, is_gl=False)

    # --------
    # 残存回転判定用

    assert np.isclose(
        np.array([0.0, 9.379668, -1.051170]),
        bone_matrixes[0, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.919751, 8.397145, -0.324375]),
        bone_matrixes[0, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.422861, 6.169319, -4.100779]),
        bone_matrixes[0, "右ひざ"].position.vector,
        atol=0.1,
    ).all()
    assert np.isclose(
        np.array([-1.821804, 2.095607, -1.186269]),
        bone_matrixes[0, "右足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-1.390510, -0.316872, -1.544655]),
        bone_matrixes[0, "右つま先"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.919751, 8.397145, -0.324375]),
        bone_matrixes[0, "右足D"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.422861, 6.169319, -4.100779]),
        bone_matrixes[0, "右ひざD"].position.vector,
        atol=0.1,
    ).all()
    assert np.isclose(
        np.array([-1.821804, 2.095607, -1.186269]),
        bone_matrixes[0, "右足首D"].position.vector,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_leg_ik8_syou() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "唱(ダンスのみ)_0-300F.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモデル.pmx")
    )

    # キーフレ
    bone_matrixes = motion.animate_bone([278], model, ["右つま先"])

    # --------

    assert np.isclose(
        np.array([0.721499, 11.767294, 1.638818]),
        bone_matrixes[278, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.133304, 10.693992, 2.314730]),
        bone_matrixes[278, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-2.833401, 8.174604, -0.100545]),
        bone_matrixes[278, "右ひざ"].position.vector,
        atol=0.6,
    ).all()
    assert np.isclose(
        np.array([-0.409387, 5.341005, 3.524572]),
        bone_matrixes[278, "右足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.578271, 2.874233, 3.669599]),
        bone_matrixes[278, "右つま先"].position.vector,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_leg_ik9_syou() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "唱(ダンスのみ)_0-300F.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモデル.pmx")
    )

    # キーフレ
    bone_matrixes = motion.animate_bone(
        [
            100,
            107,
            272,
            273,
            274,
            275,
            278,
        ],
        model,
        ["右つま先"],
    )

    # --------

    assert np.isclose(
        np.array([0.365000, 11.411437, 1.963828]),
        bone_matrixes[100, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.513678, 10.280550, 2.500991]),
        bone_matrixes[100, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-2.891708, 8.162312, -0.553409]),
        bone_matrixes[100, "右ひざ"].position.vector,
        atol=0.1,
    ).all()
    assert np.isclose(
        np.array([-0.826174, 4.330670, 2.292396]),
        bone_matrixes[100, "右足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-1.063101, 1.865613, 2.335564]),
        bone_matrixes[100, "右つま先"].position.vector,
        atol=0.01,
    ).all()

    # --------

    assert np.isclose(
        np.array([0.365000, 12.042871, 2.034023]),
        bone_matrixes[107, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.488466, 10.920292, 2.626419]),
        bone_matrixes[107, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-1.607765, 6.0764317, 1.653586]),
        bone_matrixes[107, "右ひざ"].position.vector,
        atol=0.7,
    ).all()
    assert np.isclose(
        np.array([-1.110289, 1.718307, 2.809817]),
        bone_matrixes[107, "右足首"].position.vector,
        atol=0.3,
    ).all()
    assert np.isclose(
        np.array([-1.753089, -0.026766, 1.173958]),
        bone_matrixes[107, "右つま先"].position.vector,
        atol=0.3,
    ).all()

    # --------

    assert np.isclose(
        np.array([-0.330117, 10.811301, 1.914508]),
        bone_matrixes[272, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-1.325985, 9.797281, 2.479780]),
        bone_matrixes[272, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-1.394679, 6.299243, -0.209150]),
        bone_matrixes[272, "右ひざ"].position.vector,
        atol=0.02,
    ).all()
    assert np.isclose(
        np.array([-0.865021, 1.642431, 2.044760]),
        bone_matrixes[272, "右足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-1.191817, -0.000789, 0.220605]),
        bone_matrixes[272, "右つま先"].position.vector,
        atol=0.01,
    ).all()

    # --------

    assert np.isclose(
        np.array([-0.154848, 10.862784, 1.868560]),
        bone_matrixes[273, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-1.153633, 9.846655, 2.436846]),
        bone_matrixes[273, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-1.498977, 6.380789, -0.272370]),
        bone_matrixes[273, "右ひざ"].position.vector,
        atol=0.02,
    ).all()
    assert np.isclose(
        np.array([-0.845777, 1.802650, 2.106815]),
        bone_matrixes[273, "右足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-1.239674, 0.026274, 0.426385]),
        bone_matrixes[273, "右つま先"].position.vector,
        atol=0.01,
    ).all()

    # --------

    assert np.isclose(
        np.array([0.049523, 10.960778, 1.822612]),
        bone_matrixes[274, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.930675, 9.938401, 2.400088]),
        bone_matrixes[274, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-1.710987, 6.669293, -0.459177]),
        bone_matrixes[274, "右ひざ"].position.vector,
        atol=0.02,
    ).all()
    assert np.isclose(
        np.array([-0.773748, 2.387820, 2.340310]),
        bone_matrixes[274, "右足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-1.256876, 0.365575, 0.994345]),
        bone_matrixes[274, "右つま先"].position.vector,
        atol=0.01,
    ).all()

    # --------

    assert np.isclose(
        np.array([0.721499, 11.767294, 1.638818]),
        bone_matrixes[278, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.133304, 10.693992, 2.314730]),
        bone_matrixes[278, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-2.833401, 8.174604, -0.100545]),
        bone_matrixes[278, "右ひざ"].position.vector,
        atol=0.6,
    ).all()
    assert np.isclose(
        np.array([-0.409387, 5.341005, 3.524572]),
        bone_matrixes[278, "右足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.578271, 2.874233, 3.669599]),
        bone_matrixes[278, "右つま先"].position.vector,
        atol=0.01,
    ).all()

    # --------

    assert np.isclose(
        np.array([0.271027, 11.113775, 1.776663]),
        bone_matrixes[275, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.689199, 10.081417, 2.369725]),
        bone_matrixes[275, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-1.955139, 7.141531, -0.667679]),
        bone_matrixes[275, "右ひざ"].position.vector,
        atol=0.2,
    ).all()
    assert np.isclose(
        np.array([-0.639503, 3.472883, 2.775674]),
        bone_matrixes[275, "右足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-1.136614, 1.219771, 1.875187]),
        bone_matrixes[275, "右つま先"].position.vector,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_leg_ik10_shining() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "シャイニングミラクル_50F.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        "D:/MMD/MikuMikuDance_v926x64/UserFile/Model/VOCALOID/初音ミク/Tda式初音ミク_盗賊つばき流Ｍトレースモデル配布 v1.07/Tda式初音ミク_盗賊つばき流Mトレースモデルv1.07.pmx"
    )

    # キーフレ
    bone_matrixes = motion.animate_bone([0], model, ["足首_R_", "右足ＩＫ"])

    # --------

    assert np.isclose(
        np.array([-1.869911, 2.074591, -0.911531]),
        bone_matrixes[0, "右足ＩＫ"].position.vector,
        atol=0.01,
    ).all()

    assert np.isclose(
        np.array([0.0, 10.142656, -1.362172]),
        bone_matrixes[0, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.843381, 8.895412, -0.666409]),
        bone_matrixes[0, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.274925, 5.679991, -4.384042]),
        bone_matrixes[0, "右ひざ"].position.vector,
        atol=0.02,
    ).all()
    assert np.isclose(
        np.array([-1.870632, 2.072767, -0.910016]),
        bone_matrixes[0, "右足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-1.485913, -0.300011, -1.310446]),
        bone_matrixes[0, "足首_R_"].position.vector,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_leg_ik11_down() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "しゃがむ.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        "D:/MMD/MikuMikuDance_v926x64/UserFile/Model/VOCALOID/初音ミク/Tda式初音ミク_盗賊つばき流Ｍトレースモデル配布 v1.07/Tda式初音ミク_盗賊つばき流Mトレースモデルv1.07.pmx"
    )

    # キーフレ
    bone_matrixes = motion.animate_bone([0], model, ["足首_R_", "右足ＩＫ"])

    # --------

    assert np.isclose(
        np.array([-1.012964, 1.623157, 0.680305]),
        bone_matrixes[0, "右足ＩＫ"].position.vector,
        atol=0.01,
    ).all()

    assert np.isclose(
        np.array([0.0, 5.953951, -0.512170]),
        bone_matrixes[0, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.896440, 4.569404, -0.337760]),
        bone_matrixes[0, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.691207, 1.986888, -4.553376]),
        bone_matrixes[0, "右ひざ"].position.vector,
        atol=0.15,
    ).all()
    assert np.isclose(
        np.array([-1.012964, 1.623157, 0.680305]),
        bone_matrixes[0, "右足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-1.013000, 0.002578, -1.146909]),
        bone_matrixes[0, "足首_R_"].position.vector,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_leg_ik12_lamb() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "Lamb_2689F.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        "D:/MMD/MikuMikuDance_v926x64/UserFile/Model/ゲーム/戦国BASARA/幸村 たぬき式 ver.1.24/真田幸村没第二衣装1.24軽量版.pmx"
    )

    # キーフレ
    bone_matrixes = motion.animate_bone([0], model, ["右つま先", "右足ＩＫ"])

    # --------

    assert np.isclose(
        np.array([-1.216134, 1.887670, -10.788675]),
        bone_matrixes[0, "右足ＩＫ"].position.vector,
        atol=0.01,
    ).all()

    assert np.isclose(
        np.array([0.803149, 6.056844, -10.232766]),
        bone_matrixes[0, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.728442, 4.560226, -11.571869]),
        bone_matrixes[0, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([4.173470, 0.361388, -11.217197]),
        bone_matrixes[0, "右ひざ"].position.vector,
        atol=0.1,
    ).all()
    assert np.isclose(
        np.array([-1.217569, 1.885731, -10.788104]),
        bone_matrixes[0, "右足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.922247, -1.163554, -10.794323]),
        bone_matrixes[0, "右つま先"].position.vector,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_leg_ik13_lamb() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "Lamb_2689F.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        "D:/MMD/MikuMikuDance_v926x64/UserFile/Model/ゲーム/戦国BASARA/幸村 たぬき式 ver.1.24/真田幸村没第二衣装1.24軽量版.pmx"
    )

    # キーフレ
    bone_matrixes = motion.animate_bone([0], model, ["左つま先", "左足ＩＫ"])

    # --------

    assert np.isclose(
        np.array([2.322227, 1.150214, -9.644499]),
        bone_matrixes[0, "左足ＩＫ"].position.vector,
        atol=0.01,
    ).all()

    assert np.isclose(
        np.array([0.803149, 6.056844, -10.232766]),
        bone_matrixes[0, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.720821, 4.639688, -8.810255]),
        bone_matrixes[0, "左足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([6.126388, 5.074682, -8.346903]),
        bone_matrixes[0, "左ひざ"].position.vector,
        atol=0.3,
    ).all()
    assert np.isclose(
        np.array([2.323599, 1.147291, -9.645196]),
        bone_matrixes[0, "左足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([5.163002, -0.000894, -9.714369]),
        bone_matrixes[0, "左つま先"].position.vector,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_leg_ik14_ballet() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "ミク用バレリーコ_1069.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        "D:/MMD/MikuMikuDance_v926x64/UserFile/Model/_あにまさ式/初音ミク_準標準.pmx"
    )

    # キーフレ
    bone_matrixes = motion.animate_bone([0], model, ["右つま先", "右足ＩＫ"])

    # --------

    assert np.isclose(
        np.array([11.324574, 10.920002, -7.150005]),
        bone_matrixes[0, "右足ＩＫ"].position.vector,
        atol=0.01,
    ).all()

    assert np.isclose(
        np.array([2.433170, 13.740387, 0.992719]),
        bone_matrixes[0, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.982654, 11.188538, 0.602013]),
        bone_matrixes[0, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([5.661557, 11.008962, -2.259013]),
        bone_matrixes[0, "右ひざ"].position.vector,
        atol=0.3,
    ).all()
    assert np.isclose(
        np.array([9.224476, 10.979847, -5.407887]),
        bone_matrixes[0, "右足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([11.345482, 10.263426, -7.003638]),
        bone_matrixes[0, "右つま先"].position.vector,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_leg_ik15_bottom() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "●ボトム.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        "D:/MMD/MikuMikuDance_v926x64/UserFile/Model/VOCALOID/初音ミク/Tda式初音ミク_盗賊つばき流Ｍトレースモデル配布 v1.07/Tda式初音ミク_盗賊つばき流Mトレースモデルv1.07.pmx"
    )

    # キーフレ
    bone_matrixes = motion.animate_bone([0, 207, 212, 218], model, ["足首_R_", "右足ＩＫ"])

    # --------

    assert np.isclose(
        np.array([-1.358434, 1.913062, 0.611182]),
        bone_matrixes[218, "右足ＩＫ"].position.vector,
        atol=0.01,
    ).all()

    assert np.isclose(
        np.array([0.150000, 4.253955, 0.237829]),
        bone_matrixes[218, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.906292, 2.996784, 0.471846]),
        bone_matrixes[218, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-2.533418, 3.889916, -4.114837]),
        bone_matrixes[218, "右ひざ"].position.vector,
        atol=0.7,
    ).all()
    assert np.isclose(
        np.array([-1.358807, 1.912181, 0.611265]),
        bone_matrixes[218, "右足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-2.040872, -0.188916, -0.430442]),
        bone_matrixes[218, "足首_R_"].position.vector,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_leg_ik16_lamb() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "lamb足ボーン長い人用.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        "D:/MMD/MikuMikuDance_v926x64/UserFile/Model/ゲーム/戦国BASARA/幸村 たぬき式 ver.1.24/真田幸村没第二衣装1.24軽量版.pmx"
    )

    # キーフレ
    bone_matrixes = motion.animate_bone([2706, 2715], model, ["左つま先", "左足ＩＫ"])

    # --------

    assert np.isclose(
        np.array([2.322227, 1.150214, -9.644499]),
        bone_matrixes[2715, "左足ＩＫ"].position.vector,
        atol=0.01,
    ).all()

    assert np.isclose(
        np.array([0.6116510, 5.235756, -10.261883]),
        bone_matrixes[2715, "下半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.669714, 3.940243, -9.147968]),
        bone_matrixes[2715, "左足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([7.072974, 4.147403, -9.769277]),
        bone_matrixes[2715, "左ひざ"].position.vector,
        atol=0.7,
    ).all()
    assert np.isclose(
        np.array([2.322686, 1.149005, -9.644754]),
        bone_matrixes[2715, "左足首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([5.161618, -0.000332, -9.714334]),
        bone_matrixes[2715, "左つま先"].position.vector,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_leg_ik7_fk() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "足FK.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモデル_ひざ制限なし.pmx")
    )

    # キーフレ
    bone_matrixes = motion.animate_bone([0], model, ["右つま先"], is_calc_ik=False)

    # --------

    assert np.isclose(
        np.array([-0.133305, 10.693993, 2.314730]),
        bone_matrixes[0, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.708069, 9.216356, -0.720822]),
        bone_matrixes[0, "右ひざ"].position.vector,
        atol=0.02,
    ).all()


def test_read_by_filepath_ok_leg_ik7_bake() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "足FK焼き込み.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモデル_ひざ制限なし.pmx")
    )

    # キーフレ
    bone_matrixes = motion.animate_bone([0], model, ["右つま先"], is_calc_ik=False)

    # --------

    assert np.isclose(
        np.array([-0.133306, 10.693994, 2.314731]),
        bone_matrixes[0, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-3.753989, 8.506582, 1.058842]),
        bone_matrixes[0, "右ひざ"].position.vector,
        atol=0.02,
    ).all()


def test_read_by_filepath_ok_leg_ik7_no_limit() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "足FK.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモデル_ひざ制限なし.pmx")
    )

    # キーフレ
    bone_matrixes = motion.animate_bone([0], model, ["右つま先"])

    # --------

    assert np.isclose(
        np.array([-0.133305, 10.693993, 2.314730]),
        bone_matrixes[0, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.081436, 7.884178, -0.268146]),
        bone_matrixes[0, "右ひざ"].position.vector,
        atol=0.3,
    ).all()


def test_read_by_filepath_ok_leg_ik7_no_limit_loop3() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "足FK.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモデル_ひざ制限なし_loop3.pmx")
    )

    # キーフレ
    bone_matrixes = motion.animate_bone([0], model, ["右つま先"])

    # --------

    assert np.isclose(
        np.array([-0.133305, 10.693993, 2.314730]),
        bone_matrixes[0, "右足"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.081436, 7.884178, -0.268146]),
        bone_matrixes[0, "右ひざ"].position.vector,
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
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモーション.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "ボーンツリーテストモデル.pmx")
    )

    # キーフレ
    bone_matrixes = motion.animate_bone([10], model)

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
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.044920, 13.397310, -0.855492]),
        bone_matrixes[10, "上半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.044920, 14.613530, -0.791352]),
        bone_matrixes[10, "上半身2"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.329451, 16.681561, -0.348142]),
        bone_matrixes[10, "左肩P"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.329451, 16.681561, -0.348142]),
        bone_matrixes[10, "左肩"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.290706, 16.678047, -0.133773]),
        bone_matrixes[10, "左腕YZ"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.611640, 15.785284, -0.086812]),
        bone_matrixes[10, "左腕捩YZ"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.611641, 15.785284, -0.086812]),
        bone_matrixes[10, "左腕捩X"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.321083, 13.811781, 0.016998]),
        bone_matrixes[10, "左ひじYZ"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.414732, 13.218668, -0.214754]),
        bone_matrixes[10, "左手捩YZ"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.414731, 13.218668, -0.214755]),
        bone_matrixes[10, "左手捩X"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.620196, 11.955698, -0.693675]),
        bone_matrixes[10, "左手捩6"].position.vector,
        atol=0.1,
    ).all()
    assert np.isclose(
        np.array([2.691600, 11.503933, -0.870235]),
        bone_matrixes[10, "左手首R"].position.vector,
        atol=0.1,
    ).all()
    assert np.isclose(
        np.array([2.633156, 11.364628, -0.882837]),
        bone_matrixes[10, "左手首2"].position.vector,
        atol=0.1,
    ).all()
    assert np.isclose(
        np.array([2.473304, 10.728573, -1.304400]),
        bone_matrixes[10, "左人指１"].position.vector,
        atol=0.2,
    ).all()
    assert np.isclose(
        np.array([2.261877, 10.458740, -1.299257]),
        bone_matrixes[10, "左人指２"].position.vector,
        atol=0.5,
    ).all()
    assert np.isclose(
        np.array([2.228296, 10.653198, -1.178544]),
        bone_matrixes[10, "左人指３"].position.vector,
        atol=0.5,
    ).all()
    assert np.isclose(
        np.array([2.417647, 10.880006, -1.173676]),
        bone_matrixes[10, "左人指先"].position.vector,
        atol=0.5,
    ).all()


def test_read_by_filepath_ok_arm_ik_2() -> None:
    import os

    import numpy as np

    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモーション.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "ボーンツリーテストモデル.pmx")
    )

    # キーフレ
    bone_matrixes = motion.animate_bone([3182], model)

    assert np.isclose(
        np.array([0, 0, 0]),
        bone_matrixes[3182, "全ての親"].position.vector,
    ).all()
    assert np.isclose(
        np.array([12.400011, 9.000000, 1.885650]),
        bone_matrixes[3182, "センター"].position.vector,
    ).all()
    assert np.isclose(
        np.array([12.400011, 8.580067, 1.885650]),
        bone_matrixes[3182, "グルーブ"].position.vector,
    ).all()

    assert np.isclose(
        np.array([12.400011, 11.628636, 2.453597]),
        bone_matrixes[3182, "腰"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([12.400011, 12.567377, 1.229520]),
        bone_matrixes[3182, "上半身"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([12.344202, 13.782951, 1.178849]),
        bone_matrixes[3182, "上半身2"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([12.425960, 15.893852, 1.481421]),
        bone_matrixes[3182, "左肩P"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([12.425960, 15.893852, 1.481421]),
        bone_matrixes[3182, "左肩"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([13.348320, 15.767927, 1.802947]),
        bone_matrixes[3182, "左腕"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([13.564770, 14.998386, 1.289923]),
        bone_matrixes[3182, "左腕捩"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([14.043257, 13.297290, 0.155864]),
        bone_matrixes[3182, "左ひじ"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([13.811955, 13.552182, -0.388005]),
        bone_matrixes[3182, "左手捩"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([13.144803, 14.287374, -1.956703]),
        bone_matrixes[3182, "左手首"].position.vector,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([12.813587, 14.873419, -2.570278]),
        bone_matrixes[3182, "左人指１"].position.vector,
        atol=0.2,
    ).all()
    assert np.isclose(
        np.array([12.541822, 15.029200, -2.709604]),
        bone_matrixes[3182, "左人指２"].position.vector,
        atol=0.2,
    ).all()
    assert np.isclose(
        np.array([12.476499, 14.950351, -2.502167]),
        bone_matrixes[3182, "左人指３"].position.vector,
        atol=0.2,
    ).all()
    assert np.isclose(
        np.array([12.620306, 14.795185, -2.295859]),
        bone_matrixes[3182, "左人指先"].position.vector,
        atol=0.2,
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


def test_insert_vmd01() -> None:
    from mlib.vmd.vmd_collection import VmdMotion

    motion = VmdMotion()

    bf1 = VmdBoneFrame(9, "センター")
    bf1.position = MVector3D(1.5, 25, 0)
    motion.append_bone_frame(bf1)

    bf2 = VmdBoneFrame(17, "センター")
    bf2.position = MVector3D(0, 35, 0)
    motion.append_bone_frame(bf2)

    test_bf1 = VmdBoneFrame(12, "センター")
    test_bf1.position = MVector3D(0.5, 30, 0)
    motion.insert_bone_frame(test_bf1)

    interpolations = motion.bones["センター"].data[12].interpolations

    assert interpolations.translation_x.start.x == 20
    assert interpolations.translation_x.start.y == 20
    assert interpolations.translation_x.end.x == 107
    assert interpolations.translation_x.end.y == 107

    assert interpolations.translation_y.start.x == 20
    assert interpolations.translation_y.start.y == 20
    assert interpolations.translation_y.end.x == 107
    assert interpolations.translation_y.end.y == 107

    assert interpolations.translation_z.start.x == 20
    assert interpolations.translation_z.start.y == 20
    assert interpolations.translation_z.end.x == 107
    assert interpolations.translation_z.end.y == 107

    assert interpolations.rotation.start.x == 20
    assert interpolations.rotation.start.y == 20
    assert interpolations.rotation.end.x == 107
    assert interpolations.rotation.end.y == 107


def test_insert_vmd02() -> None:
    from mlib.vmd.vmd_collection import VmdMotion

    motion = VmdMotion()

    bf1 = VmdBoneFrame(9, "センター")
    bf1.position = MVector3D(1.5, 25, 0)
    motion.append_bone_frame(bf1)

    bf2 = VmdBoneFrame(17, "センター")
    bf2.position = MVector3D(0, 35, 0)
    bf2.interpolations.translation_x.start = MVector2D(75, 14)
    bf2.interpolations.translation_x.end = MVector2D(52, 114)
    motion.append_bone_frame(bf2)

    test_bf1 = VmdBoneFrame(12, "センター")
    test_bf1.position = MVector3D(0.5, 30, 0)
    motion.insert_bone_frame(test_bf1)

    split_interpolations = motion.bones["センター"].data[12].interpolations

    assert split_interpolations.translation_x.start.x == 63
    assert split_interpolations.translation_x.start.y == 17
    assert split_interpolations.translation_x.end.x == 100
    assert split_interpolations.translation_x.end.y == 66

    assert split_interpolations.translation_y.start.x == 20
    assert split_interpolations.translation_y.start.y == 20
    assert split_interpolations.translation_y.end.x == 107
    assert split_interpolations.translation_y.end.y == 107

    assert split_interpolations.translation_z.start.x == 20
    assert split_interpolations.translation_z.start.y == 20
    assert split_interpolations.translation_z.end.x == 107
    assert split_interpolations.translation_z.end.y == 107

    assert split_interpolations.rotation.start.x == 20
    assert split_interpolations.rotation.start.y == 20
    assert split_interpolations.rotation.end.x == 107
    assert split_interpolations.rotation.end.y == 107

    next_interpolations = motion.bones["センター"].data[17].interpolations

    assert next_interpolations.translation_x.start.x == 36
    assert next_interpolations.translation_x.start.y == 47
    assert next_interpolations.translation_x.end.x == 45
    assert next_interpolations.translation_x.end.y == 115

    assert next_interpolations.translation_y.start.x == 20
    assert next_interpolations.translation_y.start.y == 20
    assert next_interpolations.translation_y.end.x == 107
    assert next_interpolations.translation_y.end.y == 107

    assert next_interpolations.translation_z.start.x == 20
    assert next_interpolations.translation_z.start.y == 20
    assert next_interpolations.translation_z.end.x == 107
    assert next_interpolations.translation_z.end.y == 107

    assert next_interpolations.rotation.start.x == 20
    assert next_interpolations.rotation.start.y == 20
    assert next_interpolations.rotation.end.x == 107
    assert next_interpolations.rotation.end.y == 107


if __name__ == "__main__":
    freeze_support()
    pytest.main()
