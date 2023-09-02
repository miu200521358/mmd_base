import pytest


def test_MVector3D_std_mean():
    import numpy as np

    from mlib.core.math import MVector3D

    values = [
        MVector3D(1, 2, 3),
        MVector3D(1.5, 1.2, 20.3),
        MVector3D(1.8, 0.3, 1.3),
        MVector3D(15, 0.2, 1.3),
        MVector3D(1.3, 2.2, 2.3),
    ]

    assert np.isclose(
        MVector3D(1.36666667, 1.5, 2.2).one().vector,
        MVector3D.std_mean(values).vector,
    ).all()


def test_MVector3D_one():
    import numpy as np

    from mlib.core.math import MVector3D

    assert np.isclose(
        MVector3D(1, 2, 3.2).one().vector,
        MVector3D(1, 2, 3.2).vector,
    ).all()

    assert np.isclose(
        MVector3D(0, 2, 3.2).one().vector,
        MVector3D(1, 2, 3.2).vector,
    ).all()

    assert np.isclose(
        MVector3D(1, 0, 3.2).one().vector,
        MVector3D(1, 1, 3.2).vector,
    ).all()

    assert np.isclose(
        MVector3D(2, 0, 0).one().vector,
        MVector3D(2, 1, 1).vector,
    ).all()


def test_MVector3D_length():
    from mlib.core.math import MVector3D

    assert 3.7416573867739413 == MVector3D(1, 2, 3).length()
    assert 9.291393867445294 == MVector3D(2.3, 0.2, 9).length()


def test_MVector3D_length_squared():
    from mlib.core.math import MVector3D

    assert 14.0 == MVector3D(1, 2, 3).length_squared()
    assert 86.33000000000001 == MVector3D(2.3, 0.2, 9).length_squared()


def test_MVector3D_effective():
    from math import inf, nan

    import numpy as np

    from mlib.core.math import MVector3D

    v1 = MVector3D(1, 2, 3.2)
    ev1 = v1.effective()
    assert np.isclose(
        MVector3D(1, 2, 3.2).vector,
        ev1.vector,
    ).all()

    v2 = MVector3D(1.2, nan, 3.2)
    ev2 = v2.effective()
    assert np.isclose(
        MVector3D(1.2, 0, 3.2).vector,
        ev2.vector,
    ).all()

    v3 = MVector3D(1.2, 0.45, inf)
    ev3 = v3.effective()
    assert np.isclose(
        MVector3D(1.2, 0.45, 0).vector,
        ev3.vector,
    ).all()

    v4 = MVector3D(1.2, 0.45, -inf)
    ev4 = v4.effective()
    assert np.isclose(
        MVector3D(1.2, 0.45, 0).vector,
        ev4.vector,
    ).all()


def test_MVector3D_normalized():
    import numpy as np

    from mlib.core.math import MVector3D

    assert np.isclose(
        MVector3D(0.2672612419124244, 0.5345224838248488, 0.8017837257372732).vector,
        MVector3D(1, 2, 3).normalized().vector,
    ).all()
    assert np.isclose(
        MVector3D(0.24754089997827142, 0.021525295650284472, 0.9686383042628013).vector,
        MVector3D(2.3, 0.2, 9).normalized().vector,
    ).all()


def test_MVector3D_normalize():
    import numpy as np

    from mlib.core.math import MVector3D

    v = MVector3D(1, 2, 3)
    v.normalize()
    assert np.isclose(
        MVector3D(0.2672612419124244, 0.5345224838248488, 0.8017837257372732).vector,
        v.vector,
    ).all()

    v = MVector3D(2.3, 0.2, 9)
    v.normalize()
    assert np.isclose(
        MVector3D(0.24754089997827142, 0.021525295650284472, 0.9686383042628013).vector,
        v.vector,
    ).all()


def test_MVector3D_distance():
    from mlib.core.math import MVector2D, MVector3D

    assert 6.397655820689325 == MVector3D(1, 2, 3).distance(MVector3D(2.3, 0.2, 9))
    assert 6.484682804030502 == MVector3D(-1, -0.3, 3).distance(MVector3D(-2.3, 0.2, 9.33333333))

    with pytest.raises(ValueError):
        MVector3D(-1, -0.3, 3).distance(MVector2D(-2.3, 0.2))


def test_MVector3D_str():
    from mlib.core.math import MVector3D

    assert "[x=1.0, y=2.0, z=3.0]" == str(MVector3D(1, 2, 3))
    assert "[x=1.23, y=2.56, z=3.56789]" == str(MVector3D(1.23, 2.56, 3.56789))


def test_MVector3D_element():
    import numpy as np

    from mlib.core.math import MVector3D

    v = MVector3D(1, 2, 3)
    v.x += 3
    assert np.isclose(
        MVector3D(4, 2, 3).vector,
        v.vector,
    ).all()

    v.y -= 3
    assert np.isclose(
        MVector3D(4, -1, 3).vector,
        v.vector,
    ).all()

    v.z = 8
    assert np.isclose(
        MVector3D(4, -1, 8).vector,
        v.vector,
    ).all()


def test_MVector3D_calc():
    import numpy as np

    from mlib.core.math import MVector3D

    v = MVector3D(1, 2, 3)
    v += MVector3D(1, 2, 3)
    assert np.isclose(
        MVector3D(2, 4, 6).vector,
        v.vector,
    ).all()
    v -= 2
    assert np.isclose(
        MVector3D(0, 2, 4).vector,
        v.vector,
    ).all()
    w = v * 2
    assert np.isclose(
        MVector3D(0, 4, 8).vector,
        w.vector,
    ).all()
    w = v + MVector3D(3, 4, 5)
    assert np.isclose(
        MVector3D(
            3,
            6,
            9,
        ).vector,
        w.vector,
    ).all()
    w = v - MVector3D(2, 1, -1)
    assert np.isclose(
        MVector3D(
            -2,
            1,
            5,
        ).vector,
        w.vector,
    ).all()


def test_MVector3D_get_local_matrix():
    import numpy as np

    from mlib.core.math import MVector3D

    local_matrix = MVector3D(0.8, 0.6, 1).to_local_matrix4x4()

    assert np.isclose(
        local_matrix.vector,
        np.array(
            [
                [0.56568542, 0.6, 0.56568542, 0.0],
                [0.42426407, -0.8, 0.42426407, 0.0],
                [0.70710678, 0.0, -0.70710678, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ).all()

    local_vector = local_matrix * MVector3D(1, 0, 0)
    assert np.isclose(
        local_vector.vector,
        np.array([0.56568542, 0.42426407, 0.70710678]),
    ).all()

    local_vector = local_matrix * MVector3D(1, 0, 1)
    assert np.isclose(
        local_vector.vector,
        np.array([1.13137085e00, 8.48528137e-01, -1.11022302e-16]),
    ).all()

    local_matrix = MVector3D(0, 0, -0.5).to_local_matrix4x4()

    assert np.isclose(
        local_matrix.vector,
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    ).all()


def test_calc_local_positions():
    import numpy as np

    from mlib.core.math import MVector3D, calc_local_positions

    vertex_local_positions = calc_local_positions(
        np.array([[1, 0, 0], [0.5, 3, 2], [-1, -2, 3]]), MVector3D(0.5, 0.5, 1), MVector3D(0.7, 2, 1.5)
    )

    assert np.isclose(
        vertex_local_positions,
        np.array(
            [
                [1.38777878e-16, 5.61695810e-01, 8.14756728e-01],
                [0.00000000e00, -3.30409300e-01, -1.72073304e-01],
                [2.22044605e-16, -1.15643255e00, -2.73866065e00],
            ]
        ),
    ).all()


def test_operate_vector():
    import operator

    import numpy as np

    from mlib.core.math import MVector3D, operate_vector

    assert np.isclose(
        operate_vector(MVector3D(0.247, 0.021, 3), 2, operator.add).vector,
        MVector3D(2.247, 2.021, 5).vector,
    ).all()

    assert np.isclose(
        operate_vector(MVector3D(0.247, 0.021, -3), 2, operator.sub).vector,
        MVector3D(-1.753, -1.979, -5).vector,
    ).all()

    assert np.isclose(
        operate_vector(MVector3D(0.247, 0.021, -3), MVector3D(1, 2, 3), operator.mul).vector,
        MVector3D(0.247, 0.042, -9).vector,
    ).all()


def test_MQuaternion_normalized():
    import numpy as np

    from mlib.core.math import MQuaternion

    assert np.isclose(
        np.array([0.18257419, 0.36514837, 0.54772256, 0.73029674]),
        MQuaternion(1, 2, 3, 4).normalized().vector.components,
    ).all()

    assert np.isclose(
        np.array([1, 0, 0, 0]),
        MQuaternion(1, 0, 0, 0).normalized().vector.components,
    ).all()

    assert np.isclose(
        np.array([1, 0, 0, 0]),
        MQuaternion().normalized().vector.components,
    ).all()


def test_MQuaternion_from_euler_degrees():
    import numpy as np

    from mlib.core.math import MQuaternion, MVector3D

    assert np.isclose(
        np.array([1, 0, 0, 0]),
        MQuaternion.from_euler_degrees(MVector3D()).vector.components,
    ).all()

    assert np.isclose(
        np.array([0.9961946980917455, 0.08715574274765817, 0.0, 0.0]),
        MQuaternion.from_euler_degrees(MVector3D(10, 0, 0)).vector.components,
    ).all()

    assert np.isclose(
        np.array(
            [
                0.9515485246437885,
                0.12767944069578063,
                0.14487812541736916,
                0.2392983377447303,
            ]
        ),
        MQuaternion.from_euler_degrees(MVector3D(10, 20, 30)).vector.components,
    ).all()

    assert np.isclose(
        np.array(
            [
                0.7091446481376844,
                0.4738680537545347,
                0.20131048764138487,
                -0.48170221425083437,
            ]
        ),
        MQuaternion.from_euler_degrees(60, -20, -80).vector.components,
    ).all()


def test_MQuaternion_to_euler_degrees():
    import numpy as np

    from mlib.core.math import MQuaternion

    assert np.isclose(
        np.array([0, 0, 0]),
        MQuaternion(1, 0, 0, 0).to_euler_degrees().vector,
    ).all()

    assert np.isclose(
        np.array([10, 0, 0]),
        MQuaternion(0.9961946980917455, 0.08715574274765817, 0.0, 0.0).to_euler_degrees().vector,
    ).all()

    assert np.isclose(
        np.array([10, 20, 30]),
        MQuaternion(
            0.9515485246437885,
            0.12767944069578063,
            0.14487812541736916,
            0.2392983377447303,
        )
        .to_euler_degrees()
        .vector,
    ).all()

    assert np.isclose(
        np.array([60, -20, -80]),
        MQuaternion(
            0.7091446481376844,
            0.4738680537545347,
            0.20131048764138487,
            -0.48170221425083437,
        )
        .to_euler_degrees()
        .vector,
    ).all()


def test_MQuaternion_multiply():
    import numpy as np

    from mlib.core.math import MQuaternion

    assert np.isclose(
        # x: 79.22368, y: -56.99869 z: -87.05808
        np.array(
            [
                0.7003873887093154,
                0.6594130183457979,
                0.11939693791117263,
                -0.24571599091322077,
            ]
        ),
        (
            # 60, -20, -80
            MQuaternion(
                0.7091446481376844,
                0.4738680537545347,
                0.20131048764138487,
                -0.48170221425083437,
            )
            # 10, 20, 30
            * MQuaternion(
                0.9515485246437885,
                0.12767944069578063,
                0.14487812541736916,
                0.2392983377447303,
            )
        ).vector.components,
    ).all()

    assert np.isclose(
        # x: 64.74257, y: 61.89256 z: -9.05046
        np.array(
            [
                0.7003873887093154,
                0.4234902605993554,
                0.46919555165368526,
                -0.3316158006229952,
            ]
        ),
        (
            # 10, 20, 30
            MQuaternion(
                0.9515485246437885,
                0.12767944069578063,
                0.14487812541736916,
                0.2392983377447303,
            )
            # 60, -20, -80
            * MQuaternion(
                0.7091446481376844,
                0.4738680537545347,
                0.20131048764138487,
                -0.48170221425083437,
            )
        ).vector.components,
    ).all()


def test_MQuaternion_to_degrees():
    import numpy as np

    from mlib.core.math import MQuaternion

    assert np.isclose(
        # np.array([0, 0, 0]),
        0,
        MQuaternion(1, 0, 0, 0).to_degrees(),
    ).all()

    assert np.isclose(
        # np.array([10, 0, 0]),
        10,
        MQuaternion(0.9961946980917455, 0.08715574274765817, 0.0, 0.0).to_degrees(),
    ).all()

    assert np.isclose(
        # np.array([10, 20, 30]),
        35.81710117358426,
        MQuaternion(
            0.9515485246437885,
            0.12767944069578063,
            0.14487812541736916,
            0.2392983377447303,
        ).to_degrees(),
    ).all()

    assert np.isclose(
        # np.array([60, -20, -80]),
        89.66927179998277,
        MQuaternion(
            0.7091446481376844,
            0.4738680537545347,
            0.20131048764138487,
            -0.48170221425083437,
        ).to_degrees(),
    ).all()


def test_MQuaternion_to_signed_degrees():
    import numpy as np

    from mlib.core.math import MQuaternion, MVector3D

    assert np.isclose(
        # np.array([0, 0, 0]),
        0,
        MQuaternion(1, 0, 0, 0).to_signed_degrees(MVector3D(1, 2, -3)),
    ).all()

    assert np.isclose(
        # np.array([10, 0, 0]),
        10,
        MQuaternion(0.9961946980917455, 0.08715574274765817, 0.0, 0.0).to_signed_degrees(MVector3D(1, 2, -3)),
    ).all()

    assert np.isclose(
        # np.array([10, 20, 30]),
        -35.81710117358426,
        MQuaternion(
            0.9515485246437885,
            0.12767944069578063,
            0.14487812541736916,
            0.2392983377447303,
        ).to_signed_degrees(MVector3D(1, 2, -3)),
    ).all()

    assert np.isclose(
        # np.array([60, -20, -80]),
        89.66927179998277,
        MQuaternion(
            0.7091446481376844,
            0.4738680537545347,
            0.20131048764138487,
            -0.48170221425083437,
        ).to_signed_degrees(MVector3D(1, 2, -3)),
    ).all()


def test_MQuaternion_dot():
    import numpy as np

    from mlib.core.math import MQuaternion

    assert np.isclose(
        0.6491836986795888,
        MQuaternion(
            # np.array([60, -20, -80]),
            0.7091446481376844,
            0.4738680537545347,
            0.20131048764138487,
            -0.48170221425083437,
        ).dot(
            # np.array([10, 20, 30]),
            MQuaternion(
                0.9515485246437885,
                0.12767944069578063,
                0.14487812541736916,
                0.2392983377447303,
            )
        ),
    ).all()

    assert np.isclose(
        0.9992933154462645,
        MQuaternion(
            # np.array([10, 23, 45]),
            0.908536845412201,
            0.1549093965157679,
            0.15080756177478563,
            0.3575205710320892,
        ).dot(
            # np.array([12, 20, 42]),
            MQuaternion(
                0.9208654879256133,
                0.15799222008931638,
                0.1243359045760714,
                0.33404459937562386,
            )
        ),
    ).all()


def test_MQuaternion_nlerp():
    import numpy as np

    from mlib.core.math import MQuaternion

    assert np.isclose(
        np.array(
            [
                0.8467301086780454,
                0.40070795732908604,
                0.1996771758076988,
                -0.2874200435819989,
            ]
        ),
        MQuaternion.nlerp(
            MQuaternion(
                # np.array([60, -20, -80]),
                0.7091446481376844,
                0.4738680537545347,
                0.20131048764138487,
                -0.48170221425083437,
            ),
            (
                # np.array([10, 20, 30]),
                MQuaternion(
                    0.9515485246437885,
                    0.12767944069578063,
                    0.14487812541736916,
                    0.2392983377447303,
                )
            ),
            0.3,
        ).vector.components,
    ).all()


def test_MQuaternion_slerp():
    import numpy as np

    from mlib.core.math import MQuaternion

    assert np.isclose(
        np.array(
            [
                0.851006131620254,
                0.3973722198386427,
                0.19936467087655246,
                -0.27953105525419597,
            ]
        ),
        MQuaternion.slerp(
            MQuaternion(
                # np.array([60, -20, -80]),
                0.7091446481376844,
                0.4738680537545347,
                0.20131048764138487,
                -0.48170221425083437,
            ),
            (
                # np.array([10, 20, 30]),
                MQuaternion(
                    0.9515485246437885,
                    0.12767944069578063,
                    0.14487812541736916,
                    0.2392983377447303,
                )
            ),
            0.3,
        ).vector.components,
    ).all()


def test_MQuaternion_from_axis_angles():
    import numpy as np

    from mlib.core.math import MQuaternion, MVector3D

    assert np.isclose(
        np.array(
            [
                0.9659258262890683,
                0.06917229942468747,
                0.13834459884937494,
                0.20751689827406242,
            ]
        ),
        MQuaternion.from_axis_angles(
            MVector3D(1, 2, 3),
            30,
        ).vector.components,
    ).all()

    assert np.isclose(
        np.array(
            [
                0.4771587602596084,
                -0.1168586510166092,
                0.7790576734440612,
                -0.3895288367220306,
            ]
        ),
        MQuaternion.from_axis_angles(
            MVector3D(-3, 20, -10),
            123,
        ).vector.components,
    ).all()


def test_MQuaternion_from_direction():
    import numpy as np

    from mlib.core.math import MQuaternion, MVector3D

    assert np.isclose(
        np.array(
            [
                0.7791421414666787,
                -0.3115472173245163,
                -0.045237910083403,
                -0.5420603160713341,
            ]
        ),
        MQuaternion.from_direction(
            MVector3D(1, 2, 3),
            MVector3D(4, 5, 6),
        ).vector.components,
    ).all()

    assert np.isclose(
        np.array(
            [
                -0.42497433477564167,
                0.543212292317204,
                0.6953153333136457,
                0.20212324833235548,
            ]
        ),
        MQuaternion.from_direction(
            MVector3D(-10, 20, -15),
            MVector3D(40, -5, 6),
        ).vector.components,
    ).all()


def test_MQuaternion_rotate():
    import numpy as np

    from mlib.core.math import MQuaternion, MVector3D

    assert np.isclose(
        np.array(
            [
                0.9936377222602503,
                -0.04597839511020707,
                0.0919567902204141,
                -0.04597839511020706,
            ]
        ),
        MQuaternion.rotate(
            MVector3D(1, 2, 3),
            MVector3D(4, 5, 6),
        ).vector.components,
    ).all()

    assert np.isclose(
        np.array(
            [
                0.48080755245182594,
                0.042643949239185255,
                -0.511727390870223,
                -0.7107324873197542,
            ]
        ),
        MQuaternion.rotate(
            MVector3D(-10, 20, -15),
            MVector3D(40, -5, 6),
        ).vector.components,
    ).all()


def test_MQuaternion_to_matrix4x4():
    import numpy as np

    from mlib.core.math import MQuaternion

    assert np.isclose(
        np.array(
            [
                [0.45487413, 0.87398231, -0.17101007, 0.0],
                [-0.49240388, 0.08682409, -0.8660254, 0.0],
                [-0.74204309, 0.47813857, 0.46984631, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        MQuaternion(
            # np.array([60, -20, -80]),
            0.7091446481376844,
            0.4738680537545347,
            0.20131048764138487,
            -0.48170221425083437,
        )
        .to_matrix4x4()
        .vector,
    ).all()

    assert np.isclose(
        np.array(
            [
                [-0.28213944, 0.48809647, 0.82592928, 0.0],
                [0.69636424, 0.69636424, -0.17364818, 0.0],
                [-0.65990468, 0.52615461, -0.53636474, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        MQuaternion(
            # np.array([10, 123, 45]),
            0.4684709324967611,
            0.3734504874442106,
            0.7929168339527322,
            0.11114231087966482,
        )
        .to_matrix4x4()
        .vector,
    ).all()


def test_MQuaternion_separate_local_axis_x_x():
    import numpy as np

    from mlib.core.math import MQuaternion, MVector3D

    x_qq, y_qq, z_qq, xz_qq = MQuaternion.from_euler_degrees(10, 0, 0).separate_by_axis(MVector3D(1, 0, 0))

    assert np.isclose(
        np.array(
            [
                10.0,
                0.0,
                0.0,
            ]
        ),
        x_qq.to_euler_degrees().vector,
    ).all()

    assert np.isclose(
        np.array(
            [
                0.0,
                0.0,
                0.0,
            ]
        ),
        y_qq.to_euler_degrees().vector,
    ).all()

    assert np.isclose(
        np.array(
            [
                0.0,
                0.0,
                0.0,
            ]
        ),
        z_qq.to_euler_degrees().vector,
    ).all()


def test_MQuaternion_separate_local_axis_y_x():
    import numpy as np

    from mlib.core.math import MQuaternion, MVector3D

    x_qq, y_qq, z_qq, xz_qq = MQuaternion.from_euler_degrees(0, 10, 0).separate_by_axis(MVector3D(1, 0, 0))

    assert np.isclose(
        np.array(
            [
                0.0,
                0.0,
                0.0,
            ]
        ),
        x_qq.to_euler_degrees().vector,
    ).all()

    assert np.isclose(
        np.array(
            [
                0.0,
                10.0,
                0.0,
            ]
        ),
        y_qq.to_euler_degrees().vector,
    ).all()

    assert np.isclose(
        np.array(
            [
                0.0,
                0.0,
                0.0,
            ]
        ),
        z_qq.to_euler_degrees().vector,
    ).all()


def test_MQuaternion_separate_local_axis_z_x():
    import numpy as np

    from mlib.core.math import MQuaternion, MVector3D

    x_qq, y_qq, z_qq, xz_qq = MQuaternion.from_euler_degrees(0, 0, 10).separate_by_axis(MVector3D(1, 0, 0))

    assert np.isclose(
        np.array(
            [
                0.0,
                0.0,
                0.0,
            ]
        ),
        x_qq.to_euler_degrees().vector,
    ).all()

    assert np.isclose(
        np.array(
            [
                0.0,
                0.0,
                0.0,
            ]
        ),
        y_qq.to_euler_degrees().vector,
    ).all()

    assert np.isclose(
        np.array(
            [
                0.0,
                0.0,
                10.0,
            ]
        ),
        z_qq.to_euler_degrees().vector,
    ).all()


def test_MQuaternion_separate_local_axis_x_y():
    import numpy as np

    from mlib.core.math import MQuaternion, MVector3D

    x_qq, y_qq, z_qq, xz_qq = MQuaternion.from_euler_degrees(10, 0, 0).separate_by_axis(MVector3D(0, 1, 0))

    assert np.isclose(
        np.array(
            [
                0.0,
                0.0,
                0.0,
            ]
        ),
        x_qq.to_euler_degrees().vector,
    ).all()

    assert np.isclose(
        np.array(
            [
                10.0,
                0.0,
                0.0,
            ]
        ),
        y_qq.to_euler_degrees().vector,
    ).all()

    assert np.isclose(
        np.array(
            [
                0.0,
                0.0,
                0.0,
            ]
        ),
        z_qq.to_euler_degrees().vector,
    ).all()


def test_MQuaternion_separate_local_axis_x_z():
    import numpy as np

    from mlib.core.math import MQuaternion, MVector3D

    x_qq, y_qq, z_qq, xz_qq = MQuaternion.from_euler_degrees(10, 0, 0).separate_by_axis(MVector3D(0, 0, 1))

    assert np.isclose(
        np.array(
            [
                0.0,
                0.0,
                0.0,
            ]
        ),
        x_qq.to_euler_degrees().vector,
    ).all()

    assert np.isclose(
        np.array(
            [
                0.0,
                0.0,
                0.0,
            ]
        ),
        y_qq.to_euler_degrees().vector,
    ).all()

    assert np.isclose(
        np.array(
            [
                10.0,
                0.0,
                0.0,
            ]
        ),
        z_qq.to_euler_degrees().vector,
    ).all()


def test_MQuaternion_separate_local_axis_x_xy():
    import numpy as np

    from mlib.core.math import MQuaternion, MVector3D

    x_qq, y_qq, z_qq, xz_qq = MQuaternion.from_euler_degrees(10, 0, 0).separate_by_axis(MVector3D(1, 1, 0))

    assert np.isclose(
        np.array([4.96246607, 4.98060393, 1.08639405]),
        x_qq.to_euler_degrees().vector,
    ).all()

    assert np.isclose(
        np.array(
            [
                4.96246607,
                -4.98060393,
                -1.08639405,
            ]
        ),
        y_qq.to_euler_degrees().vector,
    ).all()

    assert np.isclose(
        np.array(
            [
                0.0,
                0.0,
                0.0,
            ]
        ),
        z_qq.to_euler_degrees().vector,
    ).all()


def test_MQuaternion_mul():
    import numpy as np

    from mlib.core.math import MQuaternion, MVector3D

    assert np.isclose(
        np.array(
            [
                16.89808539,
                -29.1683191,
                16.23772986,
            ]
        ),
        (
            MQuaternion(
                # np.array([60, -20, -80]),
                0.7091446481376844,
                0.4738680537545347,
                0.20131048764138487,
                -0.48170221425083437,
            )
            * MVector3D(10, 20, 30)
        ).vector,
    ).all()


def test_MMatrix4x4_bool():
    import numpy as np

    from mlib.core.math import MMatrix4x4

    if MMatrix4x4():
        assert False
    else:
        assert True

    if MMatrix4x4(
        np.array(
            [
                [-0.28213944, 0.48809647, 0.82592928, 0.0],
                [0.69636424, 0.69636424, -0.17364818, 0.0],
                [-0.65990468, 0.52615461, -0.53636474, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
    ):
        assert True
    else:
        assert False


def test_MMatrix4x4_inverse():
    import numpy as np

    from mlib.core.math import MMatrix4x4

    assert np.isclose(
        np.array(
            [
                [-0.28213944, 0.69636424, -0.65990468, 0.0],
                [0.48809647, 0.69636424, 0.52615461, 0.0],
                [0.82592928, -0.17364818, -0.53636474, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        MMatrix4x4(
            np.array(
                [
                    [-0.28213944, 0.48809647, 0.82592928, 0.0],
                    [0.69636424, 0.69636424, -0.17364818, 0.0],
                    [-0.65990468, 0.52615461, -0.53636474, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            )
        )
        .inverse()
        .vector,
    ).all()

    assert np.isclose(
        np.array(
            [
                [0.45487413, -0.49240388, -0.74204309, -0.0],
                [0.87398231, 0.08682409, 0.47813857, 0.0],
                [-0.17101007, -0.8660254, 0.46984631, -0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        MMatrix4x4(
            np.array(
                [
                    [0.45487413, 0.87398231, -0.17101007, 0.0],
                    [-0.49240388, 0.08682409, -0.8660254, 0.0],
                    [-0.74204309, 0.47813857, 0.46984631, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            )
        )
        .inverse()
        .vector,
    ).all()


def test_MMatrix4x4_rotate():
    import numpy as np

    from mlib.core.math import MMatrix4x4, MQuaternion

    m = MMatrix4x4(
        np.array(
            [
                [-0.28213944, 0.48809647, 0.82592928, 0.0],
                [0.69636424, 0.69636424, -0.17364818, 0.0],
                [-0.65990468, 0.52615461, -0.53636474, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
    )

    m.rotate(MQuaternion.from_euler_degrees(10, 20, 30))
    assert np.isclose(
        np.array(
            [
                [-0.17489495, 0.79229066, 0.58454023, 0.0],
                [0.96753767, 0.24830537, -0.04706704, 0.0],
                [-0.18243525, 0.5573329, -0.8099984, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        m.vector,
    ).all()

    m.rotate(MQuaternion.from_euler_degrees(-40, 20, -32))
    assert np.isclose(
        np.array(
            [
                [-0.46381786, 0.0548533, 0.8842308, 0.0],
                [0.78154296, 0.49535822, 0.379224, 0.0],
                [-0.41720931, 0.86695521, -0.2726262, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        m.vector,
    ).all()


def test_MMatrix4x4_translate():
    import numpy as np

    from mlib.core.math import MMatrix4x4, MVector3D

    m = MMatrix4x4(
        np.array(
            [
                [-0.28213944, 0.48809647, 0.82592928, 0.0],
                [0.69636424, 0.69636424, -0.17364818, 0.0],
                [-0.65990468, 0.52615461, -0.53636474, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
    )

    m.translate(MVector3D(10, 20, 30))
    assert np.isclose(
        np.array(
            [
                [-0.28213944, 0.48809647, 0.82592928, 31.7184134],
                [0.69636424, 0.69636424, -0.17364818, 15.6814818],
                [-0.65990468, 0.52615461, -0.53636474, -12.1668968],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        m.vector,
    ).all()

    m.translate(MVector3D(-8, -12, 3))
    assert np.isclose(
        np.array(
            [
                [-0.28213944, 0.48809647, 0.82592928, 30.59615912],
                [0.69636424, 0.69636424, -0.17364818, 1.23325246],
                [-0.65990468, 0.52615461, -0.53636474, -14.8106089],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        m.vector,
    ).all()


def test_MMatrix4x4_scale():
    import numpy as np

    from mlib.core.math import MMatrix4x4, MQuaternion, MVector3D

    m = MMatrix4x4()

    m.rotate(MQuaternion.from_euler_degrees(10, 20, 30))
    assert np.isclose(
        np.array(
            [
                [0.8434932827949524, -0.4184120297431946, 0.33682408928871155, 0.0],
                [0.49240386486053467, 0.8528685569763184, -0.1736481636762619, 0.0],
                [-0.21461018919944763, 0.31232452392578125, 0.9254165887832642, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        m.vector,
    ).all()

    m.translate(MVector3D(1, 2, 3))
    assert np.isclose(
        np.array(
            [
                [0.8434932827949524, -0.4184120297431946, 0.33682408928871155, 1.017141580581665],
                [0.49240386486053467, 0.8528685569763184, -0.1736481636762619, 1.6771966218948364],
                [-0.21461018919944763, 0.31232452392578125, 0.9254165887832642, 3.186288833618164],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        m.vector,
    ).all()

    m.scale(MVector3D(4, 5, 6))
    assert np.isclose(
        np.array(
            [
                [3.3739731311798096, -2.092060089111328, 2.020944595336914, 1.017141580581665],
                [1.9696154594421387, 4.264342784881592, -1.041888952255249, 1.6771966218948364],
                [-0.8584407567977905, 1.5616226196289062, 5.552499771118164, 3.186288833618164],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        m.vector,
    ).all()

    m.scale(MVector3D(-0.8, -0.12, 0.3))
    assert np.isclose(
        np.array(
            [
                [-2.699178457260132, 0.25104719400405884, 0.60628342628479, 1.017141580581665],
                [-1.5756924152374268, -0.511721134185791, -0.31256669759750366, 1.6771966218948364],
                [0.6867526173591614, -0.18739470839500427, 1.6657500267028809, 3.186288833618164],
                [-0.0, -0.0, 0.0, 1.0],
            ]
        ),
        m.vector,
    ).all()


def test_MMatrix4x4_to_quaternion():
    import numpy as np

    from mlib.core.math import MMatrix4x4

    assert np.isclose(
        np.array(
            [
                0.46847093448041827,
                0.37345048680206483,
                0.7929168335960588,
                0.11114230870887896,
            ]
        ),
        MMatrix4x4(
            np.array(
                [
                    [-0.28213944, 0.48809647, 0.82592928, 0.0],
                    [0.69636424, 0.69636424, -0.17364818, 0.0],
                    [-0.65990468, 0.52615461, -0.53636474, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            )
        )
        .to_quaternion()
        .vector.components,
    ).all()

    assert np.isclose(
        np.array(
            [
                0.9515485246437885,
                0.12767944069578063,
                0.14487812541736916,
                0.2392983377447303,
            ]
        ),
        MMatrix4x4(
            np.array(
                [
                    [0.84349327, -0.41841204, 0.33682409, 0.0],
                    [0.49240388, 0.85286853, -0.17364818, 0.0],
                    [-0.21461018, 0.31232456, 0.92541658, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            )
        )
        .to_quaternion()
        .vector.components,
    ).all()


def test_MMatrix4x4_mul():
    import numpy as np

    from mlib.core.math import MMatrix4x4, MVector3D

    assert np.isclose(
        np.array([31.7184134, 15.6814818, -12.166896800000002]),
        (
            MMatrix4x4(
                np.array(
                    [
                        [-0.28213944, 0.48809647, 0.82592928, 0.0],
                        [0.69636424, 0.69636424, -0.17364818, 0.0],
                        [-0.65990468, 0.52615461, -0.53636474, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                )
            )
            * MVector3D(10, 20, 30)
        ).vector,
    ).all()


def test_MVector3D_class():
    from mlib.core.math import MVector3D

    v1 = MVector3D(0, 1, 2)
    v2 = MVector3D(3, 4, 5)
    v3 = v1 + v2
    assert MVector3D == v3.__class__


def test_MVectorDict_distances():
    import numpy as np

    from mlib.core.math import MVector3D, MVectorDict

    vd = MVectorDict()
    vd.append(1, MVector3D(1, 2, 3))
    vd.append(2, MVector3D(3, 4, 5))
    vd.append(3, MVector3D(4, -5, 6))

    assert np.isclose(
        np.array([4.242640687119285, 6.782329983125268, 5.385164807134504]),
        vd.distances(MVector3D(2, -2, 2)),
    ).all()

    assert 4.242640687119285 == vd.nearest_distance(MVector3D(2, -2, 2))
    assert MVector3D(1, 2, 3) == vd.nearest_value(MVector3D(2, -2, 2))
    assert 1 == vd.nearest_key(MVector3D(2, -2, 2))


def test_MVectorDict_nearest_all_keys():
    from mlib.core.math import MVector3D, MVectorDict

    vd = MVectorDict()
    vd.append(1, MVector3D(1, 2, 3))
    vd.append(2, MVector3D(3, 4, 5))
    vd.append(3, MVector3D(1, 2, 3))
    vd.append(4, MVector3D(4, -5, 6))

    nearest_keys = vd.nearest_all_keys(MVector3D(1, 2, 3.1))

    assert 2 == len(nearest_keys)
    assert 1 in nearest_keys
    assert 3 in nearest_keys


def test_MMatrix4x4List_translate():
    import numpy as np

    from mlib.core.math import MMatrix4x4List, MVector3D

    ms = MMatrix4x4List(4, 3)
    vs = [
        [
            MVector3D(1, 2, 3).vector,
            MVector3D(4, 5, 6).vector,
            MVector3D(7, 8, 9).vector,
        ],
        [
            MVector3D(10, 11, 12).vector,
            MVector3D(13, 14, 15).vector,
            MVector3D(16, 17, 18).vector,
        ],
        [
            MVector3D(19, 20, 21).vector,
            MVector3D(22, 23, 24).vector,
            MVector3D(25, 26, 27).vector,
        ],
        [
            MVector3D(28, 29, 30).vector,
            MVector3D(31, 32, 33).vector,
            MVector3D(34, 35, 36).vector,
        ],
    ]
    ms.translate(vs)

    assert np.isclose(
        np.array(
            [
                [
                    [
                        [1, 0, 0, 1],
                        [0, 1, 0, 2],
                        [0, 0, 1, 3],
                        [0, 0, 0, 1],
                    ],
                    [
                        [1, 0, 0, 4],
                        [0, 1, 0, 5],
                        [0, 0, 1, 6],
                        [0, 0, 0, 1],
                    ],
                    [
                        [1, 0, 0, 7],
                        [0, 1, 0, 8],
                        [0, 0, 1, 9],
                        [0, 0, 0, 1],
                    ],
                ],
                [
                    [
                        [1, 0, 0, 10],
                        [0, 1, 0, 11],
                        [0, 0, 1, 12],
                        [0, 0, 0, 1],
                    ],
                    [
                        [1, 0, 0, 13],
                        [0, 1, 0, 14],
                        [0, 0, 1, 15],
                        [0, 0, 0, 1],
                    ],
                    [
                        [1, 0, 0, 16],
                        [0, 1, 0, 17],
                        [0, 0, 1, 18],
                        [0, 0, 0, 1],
                    ],
                ],
                [
                    [
                        [1, 0, 0, 19],
                        [0, 1, 0, 20],
                        [0, 0, 1, 21],
                        [0, 0, 0, 1],
                    ],
                    [
                        [1, 0, 0, 22],
                        [0, 1, 0, 23],
                        [0, 0, 1, 24],
                        [0, 0, 0, 1],
                    ],
                    [
                        [1, 0, 0, 25],
                        [0, 1, 0, 26],
                        [0, 0, 1, 27],
                        [0, 0, 0, 1],
                    ],
                ],
                [
                    [
                        [1, 0, 0, 28],
                        [0, 1, 0, 29],
                        [0, 0, 1, 30],
                        [0, 0, 0, 1],
                    ],
                    [
                        [1, 0, 0, 31],
                        [0, 1, 0, 32],
                        [0, 0, 1, 33],
                        [0, 0, 0, 1],
                    ],
                    [
                        [1, 0, 0, 34],
                        [0, 1, 0, 35],
                        [0, 0, 1, 36],
                        [0, 0, 0, 1],
                    ],
                ],
            ],
            dtype=np.float64,
        ),
        ms.vector,
    ).all()

    ms.translate(vs)

    assert np.isclose(
        np.array(
            [
                [
                    [
                        [1, 0, 0, 2],
                        [0, 1, 0, 4],
                        [0, 0, 1, 6],
                        [0, 0, 0, 1],
                    ],
                    [
                        [1, 0, 0, 8],
                        [0, 1, 0, 10],
                        [0, 0, 1, 12],
                        [0, 0, 0, 1],
                    ],
                    [
                        [1, 0, 0, 14],
                        [0, 1, 0, 16],
                        [0, 0, 1, 18],
                        [0, 0, 0, 1],
                    ],
                ],
                [
                    [
                        [1, 0, 0, 20],
                        [0, 1, 0, 22],
                        [0, 0, 1, 24],
                        [0, 0, 0, 1],
                    ],
                    [
                        [1, 0, 0, 26],
                        [0, 1, 0, 28],
                        [0, 0, 1, 30],
                        [0, 0, 0, 1],
                    ],
                    [
                        [1, 0, 0, 32],
                        [0, 1, 0, 34],
                        [0, 0, 1, 36],
                        [0, 0, 0, 1],
                    ],
                ],
                [
                    [
                        [1, 0, 0, 38],
                        [0, 1, 0, 40],
                        [0, 0, 1, 42],
                        [0, 0, 0, 1],
                    ],
                    [
                        [1, 0, 0, 44],
                        [0, 1, 0, 46],
                        [0, 0, 1, 48],
                        [0, 0, 0, 1],
                    ],
                    [
                        [1, 0, 0, 50],
                        [0, 1, 0, 52],
                        [0, 0, 1, 54],
                        [0, 0, 0, 1],
                    ],
                ],
                [
                    [
                        [1, 0, 0, 56],
                        [0, 1, 0, 58],
                        [0, 0, 1, 60],
                        [0, 0, 0, 1],
                    ],
                    [
                        [1, 0, 0, 62],
                        [0, 1, 0, 64],
                        [0, 0, 1, 66],
                        [0, 0, 0, 1],
                    ],
                    [
                        [1, 0, 0, 68],
                        [0, 1, 0, 70],
                        [0, 0, 1, 72],
                        [0, 0, 0, 1],
                    ],
                ],
            ],
            dtype=np.float64,
        ),
        ms.vector,
    ).all()


def test_MMatrix4x4List_rotate():
    import numpy as np

    from mlib.core.math import MMatrix4x4List, MQuaternion, MVector3D

    ms = MMatrix4x4List(4, 3)
    qs = [
        [
            MQuaternion.from_euler_degrees(MVector3D(1, 2, 3)).to_matrix4x4().vector,
            MQuaternion.from_euler_degrees(MVector3D(4, 5, 6)).to_matrix4x4().vector,
            MQuaternion.from_euler_degrees(MVector3D(7, 8, 9)).to_matrix4x4().vector,
        ],
        [
            MQuaternion.from_euler_degrees(MVector3D(10, 11, 12)).to_matrix4x4().vector,
            MQuaternion.from_euler_degrees(MVector3D(13, 14, 15)).to_matrix4x4().vector,
            MQuaternion.from_euler_degrees(MVector3D(16, 17, 18)).to_matrix4x4().vector,
        ],
        [
            MQuaternion.from_euler_degrees(MVector3D(19, 20, 21)).to_matrix4x4().vector,
            MQuaternion.from_euler_degrees(MVector3D(22, 23, 24)).to_matrix4x4().vector,
            MQuaternion.from_euler_degrees(MVector3D(25, 26, 27)).to_matrix4x4().vector,
        ],
        [
            MQuaternion.from_euler_degrees(MVector3D(28, 29, 30)).to_matrix4x4().vector,
            MQuaternion.from_euler_degrees(MVector3D(31, 32, 33)).to_matrix4x4().vector,
            MQuaternion.from_euler_degrees(MVector3D(34, 35, 36)).to_matrix4x4().vector,
        ],
    ]
    ms.rotate(qs)

    assert np.isclose(
        np.array(
            [
                [
                    [
                        [
                            0.9980530734188116,
                            -0.051695829114809175,
                            0.03489418134011366,
                            0.0,
                        ],
                        [
                            0.052327985223313125,
                            0.9984774386394599,
                            -0.01745240643728351,
                            0.0,
                        ],
                        [
                            -0.03393883618707353,
                            0.019244370088830134,
                            0.9992386149554824,
                            0.0,
                        ],
                        [0, 0, 0, 1],
                    ],
                    [
                        [
                            0.9913729386253349,
                            -0.09808432873455783,
                            0.08694343573875721,
                            0.0,
                        ],
                        [
                            0.10427383718471567,
                            0.992099290015652,
                            -0.0697564737441253,
                            0.0,
                        ],
                        [
                            -0.07941450396586014,
                            0.07822060602635747,
                            0.9937680178757646,
                            0.0,
                        ],
                        [0, 0, 0, 1],
                    ],
                    [
                        [
                            0.9807295002644256,
                            -0.1381599383942841,
                            0.13813572576990216,
                            0.0,
                        ],
                        [
                            0.15526842625975007,
                            0.9803262614787074,
                            -0.12186934340514752,
                            0.0,
                        ],
                        [
                            -0.11858061864364906,
                            0.1409689770058251,
                            0.9828867607227298,
                            0.0,
                        ],
                        [0, 0, 0, 1],
                    ],
                ],
                [
                    [
                        [
                            0.9670651442408191,
                            -0.17168218254019432,
                            0.1879101779912919,
                            0.0,
                        ],
                        [
                            0.20475304505920652,
                            0.9632873407929416,
                            -0.17364817766693036,
                            0.0,
                        ],
                        [
                            -0.15119919752917388,
                            0.20640428112395992,
                            0.9667140608267966,
                            0.0,
                        ],
                        [0, 0, 0, 1],
                    ],
                    [
                        [
                            0.9513187851167855,
                            -0.19856476434103137,
                            0.23572145308841513,
                            0.0,
                        ],
                        [
                            0.25218552974419584,
                            0.9411692099390112,
                            -0.22495105434386503,
                            0.0,
                        ],
                        [
                            -0.17718642067484328,
                            0.27344570324831924,
                            0.9454271096723794,
                            0.0,
                        ],
                        [0, 0, 0, 1],
                    ],
                    [
                        [
                            0.9344031054290521,
                            -0.21887014283830303,
                            0.2810457207261553,
                            0.0,
                        ],
                        [
                            0.2970462000866238,
                            0.9142141997870686,
                            -0.2756373558169991,
                            0.0,
                        ],
                        [
                            -0.19660720123800332,
                            0.3410399646399675,
                            0.9192591315509074,
                            0.0,
                        ],
                        [0, 0, 0, 1],
                    ],
                ],
                [
                    [
                        [
                            0.9171832195226041,
                            -0.2328007279495099,
                            0.32338639874356045,
                            0.0,
                        ],
                        [
                            0.3388435531945202,
                            0.8827176350690368,
                            -0.3255681544571567,
                            0.0,
                        ],
                        [
                            -0.20966637375760366,
                            0.40818304448409537,
                            0.8884968283066811,
                            0.0,
                        ],
                        [0, 0, 0, 1],
                    ],
                    [
                        [
                            0.9004572559289628,
                            -0.2406869879165799,
                            0.3622795938119156,
                            0.0,
                        ],
                        [
                            0.37711964852057606,
                            0.8470245987390466,
                            -0.37460659341591207,
                            0.0,
                        ],
                        [
                            -0.2166967949569158,
                            0.47393997824471545,
                            0.8534772381714695,
                            0.0,
                        ],
                        [0, 0, 0, 1],
                    ],
                    [
                        [
                            0.8849392971334606,
                            -0.24297283556785476,
                            0.3972991839471272,
                            0.0,
                        ],
                        [
                            0.41145512515461147,
                            0.807526151172377,
                            -0.42261826174069944,
                            0.0,
                        ],
                        [
                            -0.21814472345885433,
                            0.5374622929553696,
                            0.8145840431031144,
                            0.0,
                        ],
                        [0, 0, 0, 1],
                    ],
                ],
                [
                    [
                        [
                            0.8712450500685638,
                            -0.24019872171714876,
                            0.4280614871913539,
                            0.0,
                        ],
                        [
                            0.4414737964294635,
                            0.7646550456261503,
                            -0.4694715627858909,
                            0.0,
                        ],
                        [
                            -0.2145529067553793,
                            0.5980027050807696,
                            0.772243365085709,
                            0.0,
                        ],
                        [0, 0, 0, 1],
                    ],
                    [
                        [
                            0.859880546532465,
                            -0.23298291468829632,
                            0.45422946531282565,
                            0.0,
                        ],
                        [
                            0.4668467715008341,
                            0.7188809869040866,
                            -0.5150380749100543,
                            0.0,
                        ],
                        [
                            -0.20654185443700618,
                            0.6549267807405288,
                            0.7269190974479691,
                            0.0,
                        ],
                        [0, 0, 0, 1],
                    ],
                    [
                        [
                            0.8512340918640373,
                            -0.22200148314495471,
                            0.4755164164672426,
                            0.0,
                        ],
                        [
                            0.4872960587442045,
                            0.6707054851723824,
                            -0.559192903470747,
                            0.0,
                        ],
                        [
                            -0.19478981487945518,
                            0.7077213389753907,
                            0.6791078223508458,
                            0.0,
                        ],
                        [0, 0, 0, 1],
                    ],
                ],
            ],
            dtype=np.float64,
        ),
        ms.vector,
    ).all()


def test_MMatrix4x4List_scale():
    import numpy as np

    from mlib.core.math import MMatrix4x4List, MQuaternion, MVector3D

    ms = MMatrix4x4List(2, 3)

    qs = [
        [
            MQuaternion.from_euler_degrees(MVector3D(1, 2, 3)).to_matrix4x4().vector,
            MQuaternion.from_euler_degrees(MVector3D(4, 5, 6)).to_matrix4x4().vector,
            MQuaternion.from_euler_degrees(MVector3D(7, 8, 9)).to_matrix4x4().vector,
        ],
        [
            MQuaternion.from_euler_degrees(MVector3D(10, 11, 12)).to_matrix4x4().vector,
            MQuaternion.from_euler_degrees(MVector3D(13, 14, 15)).to_matrix4x4().vector,
            MQuaternion.from_euler_degrees(MVector3D(16, 17, 18)).to_matrix4x4().vector,
        ],
    ]
    ms.rotate(qs)

    vs = [
        [
            MVector3D(1, 2, 3).vector,
            MVector3D(4, 5, 6).vector,
            MVector3D(7, 8, 9).vector,
        ],
        [
            MVector3D(10, 11, 12).vector,
            MVector3D(13, 14, 15).vector,
            MVector3D(16, 17, 18).vector,
        ],
    ]
    ms.translate(vs)

    ss = [
        [
            MVector3D(1, 2, 3).vector,
            MVector3D(4, 5, 6).vector,
            MVector3D(7, 8, 9).vector,
        ],
        [
            MVector3D(10, 11, 12).vector,
            MVector3D(13, 14, 15).vector,
            MVector3D(16, 17, 18).vector,
        ],
    ]
    ms.scale(ss)

    assert np.isclose(
        np.array(
            [
                [
                    [
                        [0.9980530738830566, -0.10339166224002838, 0.10468252748250961, 0.9993439316749573],
                        [0.05232799053192139, 1.9969549179077148, -0.05235721915960312, 1.9969258308410645],
                        [-0.03393883258104324, 0.03848874196410179, 2.997715950012207, 3.0022659301757812],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    [
                        [3.965491771697998, -0.49042168259620667, 0.5216606259346008, 3.9967305660247803],
                        [0.4170953929424286, 4.960496425628662, -0.4185388684272766, 4.959052562713623],
                        [-0.31765803694725037, 0.39110302925109863, 5.962608337402344, 6.03605318069458],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    [
                        [6.865106582641602, -1.105279564857483, 1.243221402168274, 7.003048419952393],
                        [1.0868791341781616, 7.842609882354736, -1.0968241691589355, 7.832664966583252],
                        [-0.8300642371177673, 1.1277518272399902, 8.84598159790039, 9.143669128417969],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                ],
                [
                    [
                        [9.67065143585205, -1.888504147529602, 2.254922389984131, 10.037069320678711],
                        [2.0475306510925293, 10.596160888671875, -2.083778142929077, 10.559913635253906],
                        [-1.5119922161102295, 2.270447254180908, 11.600568771362305, 12.359024047851562],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    [
                        [12.367144584655762, -2.779906988143921, 3.5358221530914307, 13.123059272766113],
                        [3.278412103652954, 13.176368713378906, -3.3742661476135254, 13.080514907836914],
                        [-2.3034238815307617, 3.828240156173706, 14.18140697479248, 15.706223487854004],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    [
                        [14.95044994354248, -3.7207934856414795, 5.058823108673096, 16.28847885131836],
                        [4.752740383148193, 15.541641235351562, -4.961472988128662, 15.332908630371094],
                        [-3.1457152366638184, 5.797680377960205, 16.54666519165039, 19.198631286621094],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                ],
            ],
            dtype=np.float64,
        ),
        ms.vector,
    ).all()


def test_MMatrix4x4List_inverse():
    import numpy as np

    from mlib.core.math import MMatrix4x4List

    ms = MMatrix4x4List(4, 3)
    ms.vector = np.array(
        [
            [
                [
                    [
                        0.9980530734188116,
                        -0.051695829114809175,
                        0.03489418134011366,
                        0.0,
                    ],
                    [
                        0.052327985223313125,
                        0.9984774386394599,
                        -0.01745240643728351,
                        0.0,
                    ],
                    [
                        -0.03393883618707353,
                        0.019244370088830134,
                        0.9992386149554824,
                        0.0,
                    ],
                    [0, 0, 0, 1],
                ],
                [
                    [
                        0.9913729386253349,
                        -0.09808432873455783,
                        0.08694343573875721,
                        0.0,
                    ],
                    [
                        0.10427383718471567,
                        0.992099290015652,
                        -0.0697564737441253,
                        0.0,
                    ],
                    [
                        -0.07941450396586014,
                        0.07822060602635747,
                        0.9937680178757646,
                        0.0,
                    ],
                    [0, 0, 0, 1],
                ],
                [
                    [
                        0.9807295002644256,
                        -0.1381599383942841,
                        0.13813572576990216,
                        0.0,
                    ],
                    [
                        0.15526842625975007,
                        0.9803262614787074,
                        -0.12186934340514752,
                        0.0,
                    ],
                    [
                        -0.11858061864364906,
                        0.1409689770058251,
                        0.9828867607227298,
                        0.0,
                    ],
                    [0, 0, 0, 1],
                ],
            ],
            [
                [
                    [
                        0.9670651442408191,
                        -0.17168218254019432,
                        0.1879101779912919,
                        0.0,
                    ],
                    [
                        0.20475304505920652,
                        0.9632873407929416,
                        -0.17364817766693036,
                        0.0,
                    ],
                    [
                        -0.15119919752917388,
                        0.20640428112395992,
                        0.9667140608267966,
                        0.0,
                    ],
                    [0, 0, 0, 1],
                ],
                [
                    [
                        0.9513187851167855,
                        -0.19856476434103137,
                        0.23572145308841513,
                        0.0,
                    ],
                    [
                        0.25218552974419584,
                        0.9411692099390112,
                        -0.22495105434386503,
                        0.0,
                    ],
                    [
                        -0.17718642067484328,
                        0.27344570324831924,
                        0.9454271096723794,
                        0.0,
                    ],
                    [0, 0, 0, 1],
                ],
                [
                    [
                        0.9344031054290521,
                        -0.21887014283830303,
                        0.2810457207261553,
                        0.0,
                    ],
                    [
                        0.2970462000866238,
                        0.9142141997870686,
                        -0.2756373558169991,
                        0.0,
                    ],
                    [
                        -0.19660720123800332,
                        0.3410399646399675,
                        0.9192591315509074,
                        0.0,
                    ],
                    [0, 0, 0, 1],
                ],
            ],
            [
                [
                    [
                        0.9171832195226041,
                        -0.2328007279495099,
                        0.32338639874356045,
                        0.0,
                    ],
                    [
                        0.3388435531945202,
                        0.8827176350690368,
                        -0.3255681544571567,
                        0.0,
                    ],
                    [
                        -0.20966637375760366,
                        0.40818304448409537,
                        0.8884968283066811,
                        0.0,
                    ],
                    [0, 0, 0, 1],
                ],
                [
                    [
                        0.9004572559289628,
                        -0.2406869879165799,
                        0.3622795938119156,
                        0.0,
                    ],
                    [
                        0.37711964852057606,
                        0.8470245987390466,
                        -0.37460659341591207,
                        0.0,
                    ],
                    [
                        -0.2166967949569158,
                        0.47393997824471545,
                        0.8534772381714695,
                        0.0,
                    ],
                    [0, 0, 0, 1],
                ],
                [
                    [
                        0.8849392971334606,
                        -0.24297283556785476,
                        0.3972991839471272,
                        0.0,
                    ],
                    [
                        0.41145512515461147,
                        0.807526151172377,
                        -0.42261826174069944,
                        0.0,
                    ],
                    [
                        -0.21814472345885433,
                        0.5374622929553696,
                        0.8145840431031144,
                        0.0,
                    ],
                    [0, 0, 0, 1],
                ],
            ],
            [
                [
                    [
                        0.8712450500685638,
                        -0.24019872171714876,
                        0.4280614871913539,
                        0.0,
                    ],
                    [
                        0.4414737964294635,
                        0.7646550456261503,
                        -0.4694715627858909,
                        0.0,
                    ],
                    [
                        -0.2145529067553793,
                        0.5980027050807696,
                        0.772243365085709,
                        0.0,
                    ],
                    [0, 0, 0, 1],
                ],
                [
                    [
                        0.859880546532465,
                        -0.23298291468829632,
                        0.45422946531282565,
                        0.0,
                    ],
                    [
                        0.4668467715008341,
                        0.7188809869040866,
                        -0.5150380749100543,
                        0.0,
                    ],
                    [
                        -0.20654185443700618,
                        0.6549267807405288,
                        0.7269190974479691,
                        0.0,
                    ],
                    [0, 0, 0, 1],
                ],
                [
                    [
                        0.8512340918640373,
                        -0.22200148314495471,
                        0.4755164164672426,
                        0.0,
                    ],
                    [
                        0.4872960587442045,
                        0.6707054851723824,
                        -0.559192903470747,
                        0.0,
                    ],
                    [
                        -0.19478981487945518,
                        0.7077213389753907,
                        0.6791078223508458,
                        0.0,
                    ],
                    [0, 0, 0, 1],
                ],
            ],
        ],
        dtype=np.float64,
    )
    inv_ms = ms.inverse()

    assert np.isclose(
        np.array(
            [
                [
                    [
                        [
                            [0.99805307, 0.05232799, -0.03393884, 0.0],
                            [-0.05169583, 0.99847744, 0.01924437, 0.0],
                            [0.03489418, -0.01745241, 0.99923861, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        [
                            [0.99137294, 0.10427384, -0.0794145, 0.0],
                            [-0.09808433, 0.99209929, 0.07822061, 0.0],
                            [0.08694344, -0.06975647, 0.99376802, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        [
                            [0.9807295, 0.15526843, -0.11858062, 0.0],
                            [-0.13815994, 0.98032626, 0.14096898, 0.0],
                            [0.13813573, -0.12186934, 0.98288676, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    ],
                    [
                        [
                            [0.96706514, 0.20475305, -0.1511992, 0.0],
                            [-0.17168218, 0.96328734, 0.20640428, 0.0],
                            [0.18791018, -0.17364818, 0.96671406, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        [
                            [0.95131879, 0.25218553, -0.17718642, 0.0],
                            [-0.19856476, 0.94116921, 0.2734457, 0.0],
                            [0.23572145, -0.22495105, 0.94542711, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        [
                            [0.93440311, 0.2970462, -0.1966072, 0.0],
                            [-0.21887014, 0.9142142, 0.34103996, 0.0],
                            [0.28104572, -0.27563736, 0.91925913, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    ],
                    [
                        [
                            [0.91718322, 0.33884355, -0.20966637, 0.0],
                            [-0.23280073, 0.88271764, 0.40818304, 0.0],
                            [0.3233864, -0.32556815, 0.88849683, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        [
                            [0.90045726, 0.37711965, -0.21669679, 0.0],
                            [-0.24068699, 0.8470246, 0.47393998, 0.0],
                            [0.36227959, -0.37460659, 0.85347724, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        [
                            [0.8849393, 0.41145513, -0.21814472, 0.0],
                            [-0.24297284, 0.80752615, 0.53746229, 0.0],
                            [0.39729918, -0.42261826, 0.81458404, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    ],
                    [
                        [
                            [0.87124505, 0.4414738, -0.21455291, 0.0],
                            [-0.24019872, 0.76465505, 0.59800271, 0.0],
                            [0.42806149, -0.46947156, 0.77224337, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        [
                            [0.85988055, 0.46684677, -0.20654185, 0.0],
                            [-0.23298291, 0.71888099, 0.65492678, 0.0],
                            [0.45422947, -0.51503807, 0.7269191, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        [
                            [0.85123409, 0.48729606, -0.19478981, 0.0],
                            [-0.22200148, 0.67070549, 0.70772134, 0.0],
                            [0.47551642, -0.5591929, 0.67910782, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    ],
                ]
            ],
            dtype=np.float64,
        ),
        inv_ms.vector,
    ).all()


def test_MMatrix4x4List_matmul_cols():
    import numpy as np

    from mlib.core.math import MMatrix4x4List, MQuaternion, MVector3D

    ms = MMatrix4x4List(2, 3)
    vs1 = [
        [
            MVector3D(1, 2, 3).vector,
            MVector3D(4, 5, 6).vector,
            MVector3D(7, 8, 9).vector,
        ],
        [
            MVector3D(10, 11, 12).vector,
            MVector3D(13, 14, 15).vector,
            MVector3D(16, 17, 18).vector,
        ],
    ]
    ms.translate(vs1)

    qs1 = [
        [
            MQuaternion.from_euler_degrees(MVector3D(1, 2, 3)).to_matrix4x4().vector,
            MQuaternion.from_euler_degrees(MVector3D(4, 5, 6)).to_matrix4x4().vector,
            MQuaternion.from_euler_degrees(MVector3D(7, 8, 9)).to_matrix4x4().vector,
        ],
        [
            MQuaternion.from_euler_degrees(MVector3D(10, 11, 12)).to_matrix4x4().vector,
            MQuaternion.from_euler_degrees(MVector3D(13, 14, 15)).to_matrix4x4().vector,
            MQuaternion.from_euler_degrees(MVector3D(16, 17, 18)).to_matrix4x4().vector,
        ],
    ]
    ms.rotate(qs1)

    # colごとに行列積
    rms = ms.matmul_cols()

    assert np.isclose(
        np.array(
            [
                [
                    [
                        [0.99805307, -0.05169583, 0.03489418, 1.0],
                        [0.05232799, 0.99847744, -0.01745241, 2.0],
                        [-0.03393884, 0.01924437, 0.99923861, 3.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    [
                        [0.98128118, -0.14645132, 0.125057, 4.94309824],
                        [0.1573776, 0.98409106, -0.08244433, 7.0969847],
                        [-0.1109934, 0.10058224, 0.9887182, 8.9558982],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    [
                        [0.9248028, -0.26151466, 0.27631479, 11.765969],
                        [0.31691942, 0.93136494, -0.17922451, 15.32935739],
                        [-0.21048007, 0.25331685, 0.94420798, 17.88206615],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                ],
                [
                    [
                        [0.96706514, -0.17168218, 0.18791018, 10.0],
                        [0.20475305, 0.96328734, -0.17364818, 11.0],
                        [-0.1511992, 0.20640428, 0.96671406, 12.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    [
                        [0.84339634, -0.30222382, 0.44423347, 22.98694899],
                        [0.46848065, 0.8184763, -0.33259951, 24.54308969],
                        [-0.26307507, 0.48862799, 0.83188593, 27.42478128],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    [
                        [0.61095823, -0.30939022, 0.72870278, 39.33968801],
                        [0.7462665, 0.5322965, -0.3996833, 39.96608585],
                        [-0.26422784, 0.78799627, 0.55609848, 46.49620273],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                ],
            ],
            dtype=np.float64,
        ),
        rms.vector,
    ).all()

    # 位置
    assert np.isclose(
        np.array(
            [
                [
                    [1.0, 2.0, 3.0],
                    [
                        4.94309824,
                        7.0969847,
                        8.9558982,
                    ],
                    [
                        11.765969,
                        15.32935739,
                        17.88206615,
                    ],
                ],
                [
                    [
                        10.0,
                        11.0,
                        12.0,
                    ],
                    [
                        22.98694899,
                        24.54308969,
                        27.42478128,
                    ],
                    [
                        39.33968801,
                        39.96608585,
                        46.49620273,
                    ],
                ],
            ],
            dtype=np.float64,
        ),
        rms.to_positions(),
    ).all()


def test_intersect_line_plane():
    import numpy as np

    from mlib.core.math import MVector3D, intersect_line_plane

    line_point = MVector3D(1, 2, 3.2)
    line_direction = MVector3D(0, -1, 0)
    plane_point = MVector3D(0, 0, 0)
    plane_normal = MVector3D(0, 1, 0)
    assert np.isclose(
        MVector3D(1.0, 0.0, 3.2).vector,
        intersect_line_plane(line_point, line_direction, plane_point, plane_normal).vector,
    ).all()

    line_point = MVector3D(1, 2, 3.2)
    line_direction = MVector3D(0, -2.8, -0.2)
    plane_point = MVector3D(0, 0, 0)
    plane_normal = MVector3D(0, 1, 0)
    assert np.isclose(
        MVector3D(1.0, 0.0, 3.05714286).vector,
        intersect_line_plane(line_point, line_direction, plane_point, plane_normal).vector,
    ).all()


def test_align_triangle():
    import numpy as np

    from mlib.core.math import MVector3D, align_triangle

    assert np.isclose(
        MVector3D(-0.0005278992773858041, 15.126109523919748, 0.006334573849846925).vector,
        align_triangle(
            MVector3D(0, 15.75281, 0.3646003),
            MVector3D(0, 11.93415, -0.2263783),
            MVector3D(0, 13.20861, -0.3309),
            MVector3D(-0.00166671, 18.97112, 0.2007481),
            MVector3D(5.339733e-07, 13.34194, 0.5546426),
        ).vector,
    ).all()


def test_intersect_line_point():
    import numpy as np

    from mlib.core.math import MVector3D, intersect_line_point

    assert np.isclose(
        MVector3D(-5.89536869, 10.05249932, -0.01969229).vector,
        intersect_line_point(
            MVector3D(-3.219347, 12.35182, -0.01938487),
            MVector3D(-4.016106, 11.66722, -0.0194764),
            MVector3D(-6.391519, 10.62996, -0.2093448),
        ).vector,
    ).all()

    assert np.isclose(
        MVector3D(-4.26564252, 11.4528105, -0.01950507).vector,
        intersect_line_point(
            MVector3D(-3.219347, 12.35182, -0.01938487),
            MVector3D(-4.016106, 11.66722, -0.0194764),
            MVector3D(-4.373949, 11.57889, -0.2365222),
        ).vector,
    ).all()


if __name__ == "__main__":
    pytest.main()
