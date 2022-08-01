import pytest


def test_Bdef2_get_indecies():
    import numpy as np
    from mlib.model.pmx import Bdef2

    assert np.isclose(
        np.array([1, 2]),
        Bdef2(1, 2, 0.3).get_indecies(),
    ).all()

    assert np.isclose(
        np.array([2]),
        Bdef2(1, 2, 0.3).get_indecies(0.5),
    ).all()


def test_Bdef4_get_indecies():
    import numpy as np
    from mlib.model.pmx import Bdef4

    assert np.isclose(
        np.array([1, 2, 3, 4]),
        Bdef4(1, 2, 3, 4, 0.3, 0.2, 0.4, 0.1).get_indecies(),
    ).all()

    assert np.isclose(
        np.array([1, 3]),
        Bdef4(1, 2, 3, 4, 0.3, 0.2, 0.4, 0.1).get_indecies(0.3),
    ).all()


def test_Bdef4_normalized():
    import numpy as np
    from mlib.model.pmx import Bdef4

    d = Bdef4(1, 2, 3, 4, 5, 6, 7, 8)
    d.normalize()
    assert np.isclose(
        np.array([0.19230769, 0.23076923, 0.26923077, 0.30769231]),
        d.weights,
    ).all()


def test_Material_draw_flg():
    from mlib.model.pmx import DrawFlg, Material

    m = Material()
    m.draw_flg |= DrawFlg.DOUBLE_SIDED_DRAWING
    assert DrawFlg.DOUBLE_SIDED_DRAWING in m.draw_flg
    assert DrawFlg.DRAWING_EDGE not in m.draw_flg


def test_Bone_copy():
    from mlib.model.pmx import Bone

    b = Bone()
    assert b != b.copy()


def test_DisplaySlots_init():
    from mlib.model.pmx import DisplaySlot, DisplaySlots, Switch

    dd = DisplaySlots()

    d: DisplaySlot = dd.get(0)
    assert 0 == d.index
    assert "Root" == d.name

    d: DisplaySlot = dd.get(1)
    assert 1 == d.index
    assert "表情" == d.name

    d: DisplaySlot = dd.get_by_name("表情")
    assert 1 == d.index
    assert "表情" == d.name
    assert Switch.ON == d.special_flag

    d: DisplaySlot = dd.get(2)
    assert not d

    with pytest.raises(KeyError) as e:
        dd.get(2, required=True)
        assert "Not Found 2" == e.value

    d: DisplaySlot = dd.get_by_name("センター")
    assert not d

    with pytest.raises(KeyError) as e:
        dd.get_by_name("センター", required=True)
        assert "Not Found センター" == e.value


if __name__ == "__main__":
    pytest.main()
