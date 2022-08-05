import pytest


def test_read_by_filepath_error():
    import os

    from mlib.exception import MParseException
    from mlib.reader.vmd import VmdReader

    reader = VmdReader()
    with pytest.raises(MParseException):
        reader.read_by_filepath(os.path.join("test", "resources", "サンプルモデル.pmx"))


def test_read_by_filepath_ok():
    import os

    import numpy as np
    from mlib.model.vmd.collection import VmdMotion
    from mlib.reader.vmd import VmdReader

    reader = VmdReader()
    motion: VmdMotion = reader.read_by_filepath(
        os.path.join("test", "resources", "サンプルモーション.vmd")
    )
    assert "日本 roco式 トレス用" == motion.model_name

    center_bf = motion.bones["センター", 358]
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

    upper_bf = motion.bones["上半身", 689]
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

    left_leg_ik_bf = motion.bones["右足ＩＫ", 384]
    assert 384 == left_leg_ik_bf.index
    assert np.isclose(
        np.array([0.548680067, 0.134522215, -2.504074097]),
        left_leg_ik_bf.position.vector,
    ).all()
    assert np.isclose(
        np.array([22.20309405, 6.80959631, 2.583712695]),
        left_leg_ik_bf.rotation.to_euler_degrees_mmd().vector,
    ).all()
    assert np.isclose(
        np.array([64, 0]),
        left_leg_ik_bf.interpolations.translation_x.start.vector,
    ).all()
    assert np.isclose(
        np.array([64, 127]),
        left_leg_ik_bf.interpolations.translation_x.end.vector,
    ).all()
    assert np.isclose(
        np.array([64, 0]),
        left_leg_ik_bf.interpolations.translation_y.start.vector,
    ).all()
    assert np.isclose(
        np.array([87, 87]),
        left_leg_ik_bf.interpolations.translation_y.end.vector,
    ).all()
    assert np.isclose(
        np.array([64, 0]),
        left_leg_ik_bf.interpolations.translation_z.start.vector,
    ).all()
    assert np.isclose(
        np.array([64, 127]),
        left_leg_ik_bf.interpolations.translation_z.end.vector,
    ).all()
    assert np.isclose(
        np.array([64, 0]),
        left_leg_ik_bf.interpolations.rotation.start.vector,
    ).all()
    assert np.isclose(
        np.array([87, 87]),
        left_leg_ik_bf.interpolations.rotation.end.vector,
    ).all()


if __name__ == "__main__":
    pytest.main()
