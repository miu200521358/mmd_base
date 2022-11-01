import pytest


def test_MVector3D_length():
    from mlib.base.math import MVector3D

    assert 3.7416573867739413 == MVector3D(1, 2, 3).length()
    assert 9.291393867445294 == MVector3D(2.3, 0.2, 9).length()


def test_MVector3D_length_squared():
    from mlib.base.math import MVector3D

    assert 14.0 == MVector3D(1, 2, 3).length_squared()
    assert 86.33000000000001 == MVector3D(2.3, 0.2, 9).length_squared()


def test_MVector3D_effective():
    from math import inf, nan

    import numpy as np
    from mlib.base.math import MVector3D

    v = MVector3D(1, 2, 3.2)
    v.effective()
    assert np.isclose(
        MVector3D(1, 2, 3.2).vector,
        v.vector,
    ).all()

    v = MVector3D(1.2, nan, 3.2)
    v.effective()
    assert np.isclose(
        MVector3D(1.2, 0, 3.2).vector,
        v.vector,
    ).all()

    v = MVector3D(1.2, 0.45, inf)
    v.effective()
    assert np.isclose(
        MVector3D(1.2, 0.45, 0).vector,
        v.vector,
    ).all()

    v = MVector3D(1.2, 0.45, -inf)
    v.effective()
    assert np.isclose(
        MVector3D(1.2, 0.45, 0).vector,
        v.vector,
    ).all()


def test_MVector3D_normalized():
    import numpy as np
    from mlib.base.math import MVector3D

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
    from mlib.base.math import MVector3D

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
    from mlib.base.math import MVector2D, MVector3D

    assert 6.397655820689325 == MVector3D(1, 2, 3).distance(MVector3D(2.3, 0.2, 9))
    assert 6.484682804030502 == MVector3D(-1, -0.3, 3).distance(
        MVector3D(-2.3, 0.2, 9.33333333)
    )

    with pytest.raises(ValueError):
        MVector3D(-1, -0.3, 3).distance(MVector2D(-2.3, 0.2))


def test_MVector3D_to_log():
    from mlib.base.math import MVector3D

    assert "[x=1.0, y=2.0, z=3.0]" == MVector3D(1, 2, 3).to_log()
    assert "[x=1.23, y=2.56, z=3.568]" == MVector3D(1.23, 2.56, 3.56789).to_log()


def test_MVector3D_element():
    import numpy as np
    from mlib.base.math import MVector3D

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
    from mlib.base.math import MVector3D

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


def test_MVectorDict_distances():
    import numpy as np
    from mlib.base.math import MVector3D, MVectorDict

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


def test_operate_vector():
    import operator

    import numpy as np
    from mlib.base.math import MVector3D, operate_vector

    assert np.isclose(
        operate_vector(MVector3D(0.247, 0.021, 3), 2, operator.add).vector,
        MVector3D(2.247, 2.021, 5).vector,
    ).all()

    assert np.isclose(
        operate_vector(MVector3D(0.247, 0.021, -3), 2, operator.sub).vector,
        MVector3D(-1.753, -1.979, -5).vector,
    ).all()

    assert np.isclose(
        operate_vector(
            MVector3D(0.247, 0.021, -3), MVector3D(1, 2, 3), operator.mul
        ).vector,
        MVector3D(0.247, 0.042, -9).vector,
    ).all()


def test_MQuaternion_from_euler_degrees():
    import numpy as np
    from mlib.base.math import MQuaternion, MVector3D

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
    from mlib.base.math import MQuaternion

    assert np.isclose(
        np.array([0, 0, 0]),
        MQuaternion([1, 0, 0, 0]).to_euler_degrees().vector,
    ).all()

    assert np.isclose(
        np.array([10, 0, 0]),
        MQuaternion([0.9961946980917455, 0.08715574274765817, 0.0, 0.0])
        .to_euler_degrees()
        .vector,
    ).all()

    assert np.isclose(
        np.array([10, 20, 30]),
        MQuaternion(
            [
                0.9515485246437885,
                0.12767944069578063,
                0.14487812541736916,
                0.2392983377447303,
            ]
        )
        .to_euler_degrees()
        .vector,
    ).all()

    assert np.isclose(
        np.array([60, -20, -80]),
        MQuaternion(
            [
                0.7091446481376844,
                0.4738680537545347,
                0.20131048764138487,
                -0.48170221425083437,
            ]
        )
        .to_euler_degrees()
        .vector,
    ).all()


def test_MQuaternion_multiply():
    import numpy as np
    from mlib.base.math import MQuaternion

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
                [
                    0.7091446481376844,
                    0.4738680537545347,
                    0.20131048764138487,
                    -0.48170221425083437,
                ]
            )
            # 10, 20, 30
            * MQuaternion(
                [
                    0.9515485246437885,
                    0.12767944069578063,
                    0.14487812541736916,
                    0.2392983377447303,
                ]
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
                [
                    0.9515485246437885,
                    0.12767944069578063,
                    0.14487812541736916,
                    0.2392983377447303,
                ]
            )
            # 60, -20, -80
            * MQuaternion(
                [
                    0.7091446481376844,
                    0.4738680537545347,
                    0.20131048764138487,
                    -0.48170221425083437,
                ]
            )
        ).vector.components,
    ).all()


def test_MQuaternion_to_degrees():
    import numpy as np
    from mlib.base.math import MQuaternion

    assert np.isclose(
        # np.array([0, 0, 0]),
        0,
        MQuaternion([1, 0, 0, 0]).to_degrees(),
    ).all()

    assert np.isclose(
        # np.array([10, 0, 0]),
        10,
        MQuaternion([0.9961946980917455, 0.08715574274765817, 0.0, 0.0]).to_degrees(),
    ).all()

    assert np.isclose(
        # np.array([10, 20, 30]),
        35.81710117358426,
        MQuaternion(
            [
                0.9515485246437885,
                0.12767944069578063,
                0.14487812541736916,
                0.2392983377447303,
            ]
        ).to_degrees(),
    ).all()

    assert np.isclose(
        # np.array([60, -20, -80]),
        89.66927179998277,
        MQuaternion(
            [
                0.7091446481376844,
                0.4738680537545347,
                0.20131048764138487,
                -0.48170221425083437,
            ]
        ).to_degrees(),
    ).all()


def test_MQuaternion_to_signed_degrees():
    import numpy as np
    from mlib.base.math import MQuaternion, MVector3D

    assert np.isclose(
        # np.array([0, 0, 0]),
        0,
        MQuaternion([1, 0, 0, 0]).to_signed_degrees(MVector3D(1, 2, -3)),
    ).all()

    assert np.isclose(
        # np.array([10, 0, 0]),
        10,
        MQuaternion(
            [0.9961946980917455, 0.08715574274765817, 0.0, 0.0]
        ).to_signed_degrees(MVector3D(1, 2, -3)),
    ).all()

    assert np.isclose(
        # np.array([10, 20, 30]),
        -35.81710117358426,
        MQuaternion(
            [
                0.9515485246437885,
                0.12767944069578063,
                0.14487812541736916,
                0.2392983377447303,
            ]
        ).to_signed_degrees(MVector3D(1, 2, -3)),
    ).all()

    assert np.isclose(
        # np.array([60, -20, -80]),
        89.66927179998277,
        MQuaternion(
            [
                0.7091446481376844,
                0.4738680537545347,
                0.20131048764138487,
                -0.48170221425083437,
            ]
        ).to_signed_degrees(MVector3D(1, 2, -3)),
    ).all()


def test_MQuaternion_dot():
    import numpy as np
    from mlib.base.math import MQuaternion

    assert np.isclose(
        0.6491836986795888,
        MQuaternion(
            # np.array([60, -20, -80]),
            [
                0.7091446481376844,
                0.4738680537545347,
                0.20131048764138487,
                -0.48170221425083437,
            ]
        ).dot(
            # np.array([10, 20, 30]),
            MQuaternion(
                [
                    0.9515485246437885,
                    0.12767944069578063,
                    0.14487812541736916,
                    0.2392983377447303,
                ]
            )
        ),
    ).all()

    assert np.isclose(
        0.9992933154462645,
        MQuaternion(
            # np.array([10, 23, 45]),
            [
                0.908536845412201,
                0.1549093965157679,
                0.15080756177478563,
                0.3575205710320892,
            ]
        ).dot(
            # np.array([12, 20, 42]),
            MQuaternion(
                [
                    0.9208654879256133,
                    0.15799222008931638,
                    0.1243359045760714,
                    0.33404459937562386,
                ]
            )
        ),
    ).all()


def test_MQuaternion_nlerp():
    import numpy as np
    from mlib.base.math import MQuaternion

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
                [
                    0.7091446481376844,
                    0.4738680537545347,
                    0.20131048764138487,
                    -0.48170221425083437,
                ]
            ),
            (
                # np.array([10, 20, 30]),
                MQuaternion(
                    [
                        0.9515485246437885,
                        0.12767944069578063,
                        0.14487812541736916,
                        0.2392983377447303,
                    ]
                )
            ),
            0.3,
        ).vector.components,
    ).all()


def test_MQuaternion_slerp():
    import numpy as np
    from mlib.base.math import MQuaternion

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
                [
                    0.7091446481376844,
                    0.4738680537545347,
                    0.20131048764138487,
                    -0.48170221425083437,
                ]
            ),
            (
                # np.array([10, 20, 30]),
                MQuaternion(
                    [
                        0.9515485246437885,
                        0.12767944069578063,
                        0.14487812541736916,
                        0.2392983377447303,
                    ]
                )
            ),
            0.3,
        ).vector.components,
    ).all()


def test_MQuaternion_from_axis_angles():
    import numpy as np
    from mlib.base.math import MQuaternion, MVector3D

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
    from mlib.base.math import MQuaternion, MVector3D

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
    from mlib.base.math import MQuaternion, MVector3D

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
    from mlib.base.math import MQuaternion

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
            [
                0.7091446481376844,
                0.4738680537545347,
                0.20131048764138487,
                -0.48170221425083437,
            ]
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
            [
                0.4684709324967611,
                0.3734504874442106,
                0.7929168339527322,
                0.11114231087966482,
            ]
        )
        .to_matrix4x4()
        .vector,
    ).all()


def test_MQuaternion_mul():
    import numpy as np
    from mlib.base.math import MQuaternion, MVector3D

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
                [
                    0.7091446481376844,
                    0.4738680537545347,
                    0.20131048764138487,
                    -0.48170221425083437,
                ]
            )
            * MVector3D(10, 20, 30)
        ).vector,
    ).all()


def test_MMatrix4x4_inverse():
    import numpy as np
    from mlib.base.math import MMatrix4x4

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
            -0.28213944,
            0.48809647,
            0.82592928,
            0.0,
            0.69636424,
            0.69636424,
            -0.17364818,
            0.0,
            -0.65990468,
            0.52615461,
            -0.53636474,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
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
            0.45487413,
            0.87398231,
            -0.17101007,
            0.0,
            -0.49240388,
            0.08682409,
            -0.8660254,
            0.0,
            -0.74204309,
            0.47813857,
            0.46984631,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        .inverse()
        .vector,
    ).all()


def test_MMatrix4x4_rotate():
    import numpy as np
    from mlib.base.math import MMatrix4x4, MQuaternion

    m = MMatrix4x4(
        -0.28213944,
        0.48809647,
        0.82592928,
        0.0,
        0.69636424,
        0.69636424,
        -0.17364818,
        0.0,
        -0.65990468,
        0.52615461,
        -0.53636474,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
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
    from mlib.base.math import MMatrix4x4, MVector3D

    m = MMatrix4x4(
        -0.28213944,
        0.48809647,
        0.82592928,
        0.0,
        0.69636424,
        0.69636424,
        -0.17364818,
        0.0,
        -0.65990468,
        0.52615461,
        -0.53636474,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
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
    from mlib.base.math import MMatrix4x4, MVector3D

    m = MMatrix4x4(
        -0.28213944,
        0.48809647,
        0.82592928,
        0.0,
        0.69636424,
        0.69636424,
        -0.17364818,
        0.0,
        -0.65990468,
        0.52615461,
        -0.53636474,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )

    m.scale(MVector3D(10, 20, 30))
    assert np.isclose(
        np.array(
            [
                [-2.8213944, 9.7619294, 24.7778784, 0.0],
                [6.9636424, 13.9272848, -5.2094454, 0.0],
                [-6.5990468, 10.5230922, -16.0909422, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        m.vector,
    ).all()

    m.scale(MVector3D(-0.8, -0.12, 0.3))
    assert np.isclose(
        np.array(
            [
                [2.25711552, -1.17143153, 7.43336352, 0.0],
                [-5.57091392, -1.67127418, -1.56283362, 0.0],
                [5.27923744, -1.26277106, -4.82728266, 0.0],
                [-0.0, -0.0, 0.0, 1.0],
            ]
        ),
        m.vector,
    ).all()


def test_MMatrix4x4_to_quternion():
    import numpy as np
    from mlib.base.math import MMatrix4x4

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
            -0.28213944,
            0.48809647,
            0.82592928,
            0.0,
            0.69636424,
            0.69636424,
            -0.17364818,
            0.0,
            -0.65990468,
            0.52615461,
            -0.53636474,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        .to_quternion()
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
            0.84349327,
            -0.41841204,
            0.33682409,
            0.0,
            0.49240388,
            0.85286853,
            -0.17364818,
            0.0,
            -0.21461018,
            0.31232456,
            0.92541658,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        .to_quternion()
        .vector.components,
    ).all()


def test_MMatrix4x4_mul():
    import numpy as np
    from mlib.base.math import MMatrix4x4, MVector3D

    assert np.isclose(
        np.array([31.7184134, 15.6814818, -12.166896800000002]),
        (
            MMatrix4x4(
                -0.28213944,
                0.48809647,
                0.82592928,
                0.0,
                0.69636424,
                0.69636424,
                -0.17364818,
                0.0,
                -0.65990468,
                0.52615461,
                -0.53636474,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            )
            * MVector3D(10, 20, 30)
        ).vector,
    ).all()


def test_MVector3D_class():
    from mlib.base.math import MVector3D

    v1 = MVector3D(0, 1, 2)
    v2 = MVector3D(3, 4, 5)
    v3 = v1 + v2
    assert MVector3D == v3.__class__


if __name__ == "__main__":
    pytest.main()
