import pytest


def test_read_by_filepath_error():
    import os

    from mlib.base.exception import MParseException
    from mlib.vmd.reader import VmdReader

    reader = VmdReader()
    with pytest.raises(MParseException):
        reader.read_by_filepath(os.path.join("tests", "resources", "サンプルモデル.pmx"))


def test_read_by_filepath_ok_calc():
    import os

    import numpy as np

    from mlib.vmd.collection import VmdMotion
    from mlib.vmd.reader import VmdReader

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


def test_read_by_filepath_ok_matrix():
    import os

    import numpy as np

    from mlib.pmx.collection import PmxModel
    from mlib.pmx.reader import PmxReader
    from mlib.vmd.collection import VmdMotion
    from mlib.vmd.reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモーション.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモデル.pmx")
    )

    # キーフレ
    bone_trees = model.bone_trees.gets(["グルーブ", "左人指先"])
    bone_matrixes = motion.bones.get_matrix_by_indexes([10, 999], bone_trees, model)

    assert np.isclose(
        np.array([0, 0, 0]),
        bone_matrixes["グルーブ"][10]["全ての親"].position.vector,
    ).all()
    assert np.isclose(
        np.array([0.044920, 8.218059, 0.069347]),
        bone_matrixes["グルーブ"][10]["センター"].position.vector,
    ).all()
    assert np.isclose(
        np.array([0.044920, 9.392067, 0.064877]),
        bone_matrixes["グルーブ"][10]["グルーブ"].position.vector,
    ).all()

    assert np.isclose(
        np.array([0.044920, 11.740084, 0.055937]),
        bone_matrixes["左人指先"][10]["腰"].position.vector,
    ).all()
    assert np.isclose(
        np.array([0.044920, 12.390969, -0.100531]),
        bone_matrixes["左人指先"][10]["上半身"].position.vector,
    ).all()
    assert np.isclose(
        np.array([0.044920, 13.803633, -0.138654]),
        bone_matrixes["左人指先"][10]["上半身2"].position.vector,
    ).all()
    assert np.isclose(
        np.array([0.044920, 15.149180, 0.044429]),
        bone_matrixes["左人指先"][10]["上半身3"].position.vector,
    ).all()
    assert np.isclose(
        np.array([0.324862, 16.470263, 0.419041]),
        bone_matrixes["左人指先"][10]["左肩P"].position.vector,
    ).all()
    assert np.isclose(
        np.array([0.324862, 16.470263, 0.419041]),
        bone_matrixes["左人指先"][10]["左肩"].position.vector,
    ).all()
    assert np.isclose(
        np.array([1.369838, 16.312170, 0.676838]),
        bone_matrixes["左人指先"][10]["左腕"].position.vector,
    ).all()
    assert np.isclose(
        np.array([1.845001, 15.024807, 0.747681]),
        bone_matrixes["左人指先"][10]["左腕捩"].position.vector,
    ).all()
    assert np.isclose(
        np.array([2.320162, 13.737446, 0.818525]),
        bone_matrixes["左人指先"][10]["左ひじ"].position.vector,
    ).all()
    assert np.isclose(
        np.array([2.516700, 12.502447, 0.336127]),
        bone_matrixes["左人指先"][10]["左手捩"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.732219, 11.267447, -0.146273]),
        bone_matrixes["左人指先"][10]["左手首"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.649188, 10.546797, -0.607412]),
        bone_matrixes["左人指先"][10]["左人指１"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.408238, 10.209290, -0.576288]),
        bone_matrixes["左人指先"][10]["左人指２"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.360455, 10.422402, -0.442668]),
        bone_matrixes["左人指先"][10]["左人指３"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()

    # --------

    # キーフレがない場合
    assert np.isclose(
        np.array([0, 0, 0]),
        bone_matrixes["グルーブ"][999]["全ての親"].position.vector,
    ).all()
    assert np.isclose(
        np.array([-0.508560, 8.218059, 0.791827]),
        bone_matrixes["グルーブ"][999]["センター"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.508560, 9.182008, 0.787357]),
        bone_matrixes["グルーブ"][999]["グルーブ"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()

    assert np.isclose(
        np.array([-0.508560, 11.530025, 0.778416]),
        bone_matrixes["左人指先"][999]["腰"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.508560, 12.180910, 0.621949]),
        bone_matrixes["左人指先"][999]["上半身"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.437343, 13.588836, 0.523215]),
        bone_matrixes["左人指先"][999]["上半身2"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    # 付与親
    assert np.isclose(
        np.array([-0.552491, 14.941880, 0.528703]),
        bone_matrixes["左人指先"][999]["上半身3"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.590927, 16.312325, 0.819156]),
        bone_matrixes["左人指先"][999]["左肩P"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([-0.590927, 16.312325, 0.819156]),
        bone_matrixes["左人指先"][999]["左肩"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.072990, 16.156742, 1.666761]),
        bone_matrixes["左人指先"][999]["左腕"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.043336, 15.182318, 2.635117]),
        bone_matrixes["左人指先"][999]["左腕捩"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.013682, 14.207894, 3.603473]),
        bone_matrixes["左人指先"][999]["左ひじ"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.222444, 13.711100, 3.299384]),
        bone_matrixes["左人指先"][999]["左手捩"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.431205, 13.214306, 2.995294]),
        bone_matrixes["左人指先"][999]["左手首"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([3.283628, 13.209089, 2.884702]),
        bone_matrixes["左人指先"][999]["左人指１"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([3.665809, 13.070156, 2.797680]),
        bone_matrixes["左人指先"][999]["左人指２"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([3.886795, 12.968100, 2.718276]),
        bone_matrixes["左人指先"][999]["左人指３"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()


def test_read_by_filepath_ok_ik():
    import os

    import numpy as np

    from mlib.pmx.collection import PmxModel
    from mlib.pmx.reader import PmxReader
    from mlib.vmd.collection import VmdMotion
    from mlib.vmd.reader import VmdReader

    vmd_reader = VmdReader()
    motion: VmdMotion = vmd_reader.read_by_filepath(
        os.path.join("tests", "resources", "サンプルモーション.vmd")
    )

    pmx_reader = PmxReader()
    model: PmxModel = pmx_reader.read_by_filepath(
        os.path.join("tests", "resources", "ボーンツリーテストモデル.pmx")
    )

    # キーフレ
    bone_trees = model.bone_trees.gets(["グルーブ", "左人指先"])
    bone_matrixes = motion.bones.get_matrix_by_indexes([10, 999], bone_trees, model)

    assert np.isclose(
        np.array([0, 0, 0]),
        bone_matrixes["グルーブ"][10]["全ての親"].position.vector,
    ).all()
    assert np.isclose(
        np.array([0.044920, 9.000000, -0.199362]),
        bone_matrixes["グルーブ"][10]["センター"].position.vector,
    ).all()
    assert np.isclose(
        np.array([0.044920, 9.410000, -0.199362]),
        bone_matrixes["グルーブ"][10]["グルーブ"].position.vector,
    ).all()

    assert np.isclose(
        np.array([0.044920, 12.458570, 0.368584]),
        bone_matrixes["左人指先"][10]["腰"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.044920, 13.397310, -0.855492]),
        bone_matrixes["左人指先"][10]["上半身"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.044920, 14.613530, -0.791352]),
        bone_matrixes["左人指先"][10]["上半身2"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.329451, 16.681561, -0.348142]),
        bone_matrixes["左人指先"][10]["左肩P"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([0.329451, 16.681561, -0.348142]),
        bone_matrixes["左人指先"][10]["左肩"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.290706, 16.678047, -0.133773]),
        bone_matrixes["左人指先"][10]["左腕YZ"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.611640, 15.785284, -0.086812]),
        bone_matrixes["左人指先"][10]["左腕捩YZ"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([1.611641, 15.785284, -0.086812]),
        bone_matrixes["左人指先"][10]["左腕捩X"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.321083, 13.811781, 0.016998]),
        bone_matrixes["左人指先"][10]["左ひじYZ"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.414732, 13.218668, -0.214754]),
        bone_matrixes["左人指先"][10]["左手捩YZ"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.414731, 13.218668, -0.214755]),
        bone_matrixes["左人指先"][10]["左手捩X"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.620196, 11.955698, -0.693675]),
        bone_matrixes["左人指先"][10]["左手捩6"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.691600, 11.503933, -0.870235]),
        bone_matrixes["左人指先"][10]["左手首R"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.633156, 11.364628, -0.882837]),
        bone_matrixes["左人指先"][10]["左手首2"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.473304, 10.728573, -1.304400]),
        bone_matrixes["左人指先"][10]["左人指１"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.261877, 10.458740, -1.299257]),
        bone_matrixes["左人指先"][10]["左人指２"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.228296, 10.653198, -1.178544]),
        bone_matrixes["左人指先"][10]["左人指３"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()
    assert np.isclose(
        np.array([2.417647, 10.880006, -1.173676]),
        bone_matrixes["左人指先"][10]["左人指先"].position.vector,
        rtol=0.01,
        atol=0.01,
    ).all()


if __name__ == "__main__":
    pytest.main()
