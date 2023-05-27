import operator
from functools import lru_cache
from math import acos, atan2, cos, degrees, pi, radians, sin, sqrt
from typing import Type, TypeVar, Union

import numpy as np
from numpy.linalg import inv, norm
from quaternion import one as qq_one
from quaternion import as_rotation_matrix, from_rotation_matrix, quaternion, slerp_evaluate

from .base import BaseModel


class MRect(BaseModel):

    """
    矩形クラス

    Parameters
    ----------
    x : int
        x座標
    y : int
        y座標
    width : int
        横幅
    height : int
        縦幅
    """

    def __init__(self, x: int = 0, y: int = 0, width: int = 0, height: int = 0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    @property
    def x(self) -> int:
        return int(self.x)

    @x.setter
    def x(self, v: int):
        self.x = int(v)

    @property
    def y(self) -> int:
        return int(self.y)

    @y.setter
    def y(self, v: int):
        self.y = int(v)

    @property
    def width(self) -> int:
        return int(self.width)

    @width.setter
    def width(self, v: int):
        self.width = int(v)

    @property
    def height(self) -> int:
        return int(self.height)

    @height.setter
    def height(self, v: int):
        self.height = int(v)


@lru_cache(maxsize=None)
def calc_v3_by_ratio(
    prev_x: float, prev_y: float, prev_z: float, next_x: float, next_y: float, next_z: float, ratio_x: float, ratio_y: float, ratio_z: float
) -> np.ndarray:
    prev_v = np.array([prev_x, prev_y, prev_z], dtype=np.float64)
    next_v = np.array([next_x, next_y, next_z], dtype=np.float64)
    ratio_v = np.array([ratio_x, ratio_y, ratio_z], dtype=np.float64)
    return prev_v + (next_v - prev_v) * ratio_v


MVectorT = TypeVar("MVectorT", bound="MVector")


class MVector(BaseModel):
    """ベクトル基底クラス"""

    __slots__ = ("vector",)

    def __init__(self, x: float = 0.0):
        self.vector = np.array([x], dtype=np.float64)

    def copy(self: MVectorT) -> MVectorT:
        return self.__class__(self.x)

    def length(self: MVectorT) -> float:
        """
        ベクトルの長さ
        """
        return float(norm(self.vector, ord=2))

    def length_squared(self: MVectorT) -> float:
        """
        ベクトルの長さの二乗
        """
        return float(norm(self.vector, ord=2) ** 2)

    def effective(self: MVectorT, rtol: float = 1e-05, atol: float = 1e-08) -> MVectorT:
        vector = np.copy(self.vector)
        vector[np.isinf(vector)] = 0
        vector[np.isnan(vector)] = 0
        vector[np.isclose(vector, 0, rtol=rtol, atol=atol)] = 0
        return self.__class__(*vector)

    def round(self: MVectorT, decimals: int) -> MVectorT:
        """
        丸め処理

        Parameters
        ----------
        decimals : int
            丸め桁数

        Returns
        -------
        MVector
        """
        return self.__class__(*np.round(self.vector, decimals=decimals))

    def normalized(self: MVectorT) -> MVectorT:
        """
        正規化した値を返す
        """
        if not self:
            return self.__class__()

        vector = self.vector
        l2 = np.sqrt(np.sum(vector**2, axis=-1, keepdims=True))
        normv = np.divide(vector, l2, out=np.zeros_like(vector), where=l2 != 0)
        return self.__class__(*normv)

    def normalize(self: MVectorT) -> None:
        """
        自分自身の正規化
        """
        self.vector = self.normalized().vector

    def distance(self: MVectorT, other) -> float:
        """
        他のベクトルとの距離

        Parameters
        ----------
        other : MVector
            他のベクトル

        Returns
        -------
        float
        """
        if not isinstance(other, self.__class__):
            raise ValueError("同じ型同士で計算してください")
        return self.__class__(*(self.vector - other.vector)).length()

    def abs(self: MVectorT) -> MVectorT:
        """
        絶対値変換
        """
        return self.__class__(*np.abs(self.vector))

    def one(self: MVectorT) -> MVectorT:
        """
        0を1に変える
        """
        return self.__class__(*np.where(np.isclose(self.vector, 0), 1, self.vector))

    def cross(self: MVectorT, other) -> MVectorT:
        """
        外積
        """
        return self.__class__(*np.cross(self.vector, other.vector))

    def inner(self: MVectorT, other) -> float:
        """
        内積（一次元配列）
        """
        return float(np.inner(self.vector, other.vector))

    def dot(self: MVectorT, other) -> float:
        """
        内積（二次元の場合、二次元のまま返す）
        """
        return float(np.dot(self.vector, other.vector))

    def __lt__(self: MVectorT, other) -> bool:
        if isinstance(other, MVector):
            return bool(np.all(np.less(self.vector, other.vector)))
        else:
            return bool(np.all(np.less(self.vector, other)))

    def __le__(self: MVectorT, other) -> bool:
        if isinstance(other, MVector):
            return bool(np.all(np.less_equal(self.vector, other.vector)))
        else:
            return bool(np.all(np.less_equal(self.vector, other)))

    def __eq__(self: MVectorT, other) -> bool:
        if isinstance(other, MVector):
            return bool(np.all(np.equal(self.vector, other.vector)))
        else:
            return bool(np.all(np.equal(self.vector, other)))

    def __ne__(self: MVectorT, other) -> bool:
        if isinstance(other, MVector):
            return bool(np.any(np.not_equal(self.vector, other.vector)))
        else:
            return bool(np.any(np.not_equal(self.vector, other)))

    def __gt__(self: MVectorT, other) -> bool:
        if isinstance(other, MVector):
            return bool(np.all(np.greater(self.vector, other.vector)))
        else:
            return bool(np.all(np.greater(self.vector, other)))

    def __ge__(self: MVectorT, other) -> bool:
        if isinstance(other, MVector):
            return bool(np.all(np.greater_equal(self.vector, other.vector)))
        else:
            return bool(np.all(np.greater_equal(self.vector, other)))

    def __bool__(self: MVectorT) -> bool:
        return bool(not np.all(self.vector == 0))

    def __add__(self: MVectorT, other) -> MVectorT:
        return operate_vector(self, other, operator.add)

    def __sub__(self: MVectorT, other) -> MVectorT:
        return operate_vector(self, other, operator.sub)

    def __mul__(self: MVectorT, other) -> MVectorT:
        return operate_vector(self, other, operator.mul)

    def __truediv__(self: MVectorT, other) -> MVectorT:
        if isinstance(other, MVector) and np.count_nonzero(other.vector) == 0:
            return self.__class__()
        elif np.count_nonzero(other) == 0:
            return self.__class__()

        return operate_vector(self, other, operator.truediv)

    def __floordiv__(self: MVectorT, other) -> MVectorT:
        if isinstance(other, MVector) and np.count_nonzero(other.vector) == 0:
            return self.__class__()
        elif np.count_nonzero(other) == 0:
            return self.__class__()

        return operate_vector(self, other, operator.floordiv)

    def __mod__(self: MVectorT, other) -> MVectorT:
        return operate_vector(self, other, operator.mod)

    def __iadd__(self: MVectorT, other) -> MVectorT:
        self.vector = operate_vector(self, other, operator.add).vector
        return self

    def __isub__(self: MVectorT, other) -> MVectorT:
        self.vector = operate_vector(self, other, operator.sub).vector
        return self

    def __imul__(self: MVectorT, other) -> MVectorT:
        self.vector = operate_vector(self, other, operator.mul).vector
        return self

    def __itruediv__(self: MVectorT, other) -> MVectorT:
        if isinstance(other, MVector) and np.count_nonzero(other.vector) == 0:
            self = self.__class__()
        elif np.count_nonzero(other) == 0:
            self = self.__class__()
        else:
            self.vector = operate_vector(self, other, operator.truediv).vector
        return self

    def __ifloordiv__(self: MVectorT, other) -> MVectorT:
        if isinstance(other, MVector) and np.count_nonzero(other.vector) == 0:
            self = self.__class__()
        elif np.count_nonzero(other) == 0:
            self = self.__class__()
        else:
            self.vector = operate_vector(self, other, operator.floordiv).vector
        return self

    def __imod__(self: MVectorT, other) -> MVectorT:
        self.vector = operate_vector(self, other, operator.mod).vector
        return self

    def __lshift__(self: MVectorT, other) -> MVectorT:
        return operate_vector(self, other, operator.lshift)

    def __rshift__(self: MVectorT, other) -> MVectorT:
        return operate_vector(self, other, operator.rshift)

    def __and__(self: MVectorT, other) -> MVectorT:
        return operate_vector(self, other, operator.and_)

    def __or__(self: MVectorT, other) -> MVectorT:
        return operate_vector(self, other, operator.or_)

    def __neg__(self: MVectorT) -> MVectorT:
        return self.__class__(*operator.neg(self.vector))

    def __pos__(self: MVectorT) -> MVectorT:
        return self.__class__(*operator.pos(self.vector))

    def __invert__(self: MVectorT) -> MVectorT:
        return self.__class__(*operator.invert(self.vector))

    def __hash__(self: MVectorT) -> int:
        return hash(tuple(self.vector.flatten()))

    @property
    def x(self: MVectorT) -> float:
        return self.vector[0]

    @x.setter
    def x(self: MVectorT, v: float) -> None:
        self.vector[0] = v

    def __getitem__(self: MVectorT, index: int) -> float:
        return self.vector[index]

    @classmethod
    def std_mean(cls: Type[MVectorT], values: list, range: float = 1.5) -> MVectorT:
        """標準偏差を加味したmean処理"""
        np_standard_vectors = np.array([v.vector for v in values])
        np_standard_lengths = np.array([v.length() for v in values])
        median_standard_values = np.median(np_standard_lengths)
        std_standard_values = np.std(np_standard_lengths)

        # 中央値から標準偏差の一定範囲までの値を取得
        filtered_standard_values = np_standard_vectors[
            (np_standard_lengths >= median_standard_values - range * std_standard_values)
            & (np_standard_lengths <= median_standard_values + range * std_standard_values)
        ]

        return cls(*np.mean(filtered_standard_values, axis=0))


class MVector2D(MVector):
    """
    2次元ベクトルクラス
    """

    def __init__(self, x: float = 0.0, y: float = 0.0):
        """
        初期化

        Parameters
        ----------
        x : float, optional
            X値, by default 0.0
        y : float, optional
            Y値, by default 0.0
        """
        self.vector = np.array([x, y], dtype=np.float64)

    def __str__(self) -> str:
        return f"[x={round(self.vector[0], 5)}, y={round(self.vector[1], 5)}]"

    def copy(self) -> "MVector2D":
        return self.__class__(self.x, self.y)

    @property
    def y(self) -> float:
        return self.vector[1]

    @y.setter
    def y(self, v) -> None:
        self.vector[1] = v

    @property
    def gl(self) -> "MVector2D":
        return MVector2D(-self.x, self.y)


class MVector3D(MVector):
    """
    3次元ベクトルクラス
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.vector = np.array([x, y, z], dtype=np.float64)

    def __str__(self) -> str:
        """
        ログ用文字列に変換
        """
        return f"[x={round(self.vector[0], 5)}, y={round(self.vector[1], 5)}, z={round(self.vector[2], 5)}]"

    def copy(self) -> "MVector3D":
        return self.__class__(self.x, self.y, self.z)

    def to_key(self, threshold=0.1) -> tuple[int, int, int]:
        """
        キー用値に変換

        Parameters
        ----------
        threshold : float, optional
            閾値, by default 0.1

        Returns
        -------
        tuple
            (x, y, z)
        """
        return (
            round(self.vector[0] / threshold),
            round(self.vector[1] / threshold),
            round(self.vector[2] / threshold),
        )

    @property
    def y(self) -> float:
        return self.vector[1]

    @y.setter
    def y(self, v: float) -> None:
        self.vector[1] = v

    @property
    def z(self) -> float:
        return self.vector[2]

    @z.setter
    def z(self, v: float) -> None:
        self.vector[2] = v

    @property
    def gl(self) -> "MVector3D":
        return MVector3D(-self.x, self.y, self.z)

    @property
    def vector4(self) -> np.ndarray:
        return np.array([self.vector[0], self.vector[1], self.vector[2], 0], dtype=np.float64)

    @staticmethod
    def calc_by_ratio(prev_v: "MVector3D", next_v: "MVector3D", x: float, y: float, z: float) -> "MVector3D":
        return MVector3D(*calc_v3_by_ratio(*prev_v.vector, *next_v.vector, x, y, z))

    def to_local_matrix4x4(self) -> "MMatrix4x4":
        """自身をローカル軸とした場合の回転行列を取得"""
        if not self:
            return MMatrix4x4()

        # ローカルX軸の方向ベクトル
        x_axis = self.normalized().vector
        norm_x_axis = norm(x_axis)
        if not norm_x_axis:
            return MMatrix4x4()
        x_axis = x_axis / norm_x_axis
        if np.all(np.isnan(x_axis)):
            return MMatrix4x4()

        # ローカルZ軸の方向ベクトル
        z_axis = np.array([0.0, 0.0, -1.0])

        # ローカルY軸の方向ベクトル
        y_axis = np.cross(z_axis, x_axis)
        norm_y_axis = norm(y_axis)
        if not norm_y_axis:
            return MMatrix4x4()
        y_axis /= norm_y_axis
        if np.all(np.isnan(y_axis)):
            return MMatrix4x4()

        # ローカル軸に合わせた回転行列を作成する
        rotation_matrix = MMatrix4x4()
        rotation_matrix.vector[0, :3] = x_axis
        rotation_matrix.vector[1, :3] = y_axis
        rotation_matrix.vector[2, :3] = z_axis

        return rotation_matrix


class MVector4D(MVector):
    """
    4次元ベクトルクラス
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        w: float = 0.0,
    ):
        self.vector = np.array([x, y, z, w], dtype=np.float64)

    def __str__(self) -> str:
        return (
            f"[x={round(self.vector[0], 5)}, y={round(self.vector[1], 5)}, "
            + f"z={round(self.vector[2], 5)}], w={round(self.vector[2], 5)}]"
        )

    def copy(self) -> "MVector4D":
        return self.__class__(self.x, self.y, self.z, self.w)

    @property
    def y(self) -> float:
        return self.vector[1]

    @y.setter
    def y(self, v: float) -> None:
        self.vector[1] = v

    @property
    def z(self) -> float:
        return self.vector[2]

    @z.setter
    def z(self, v: float) -> None:
        self.vector[2] = v

    @property
    def w(self) -> float:
        return self.vector[3]

    @w.setter
    def w(self, v: float) -> None:
        self.vector[3] = v

    @property
    def gl(self) -> "MVector4D":
        return MVector4D(-self.x, self.y, self.z, self.w)

    @property
    def xy(self) -> "MVector2D":
        return MVector2D(*self.vector[:2])  # type: ignore

    @property
    def xyz(self) -> "MVector3D":
        return MVector3D(*self.vector[:3])  # type: ignore


class MVectorDict:
    """ベクトル辞書基底クラス"""

    __slots__ = ("vectors",)

    def __init__(self) -> None:
        """初期化"""
        self.vectors: dict[int, np.ndarray] = {}

    def __iter__(self):
        return self.vectors.items()

    def keys(self) -> list:
        return list(self.vectors.keys())

    def values(self) -> np.ndarray:
        return np.array(list(self.vectors.values()), dtype=np.float64)

    def append(self, vkey: int, v: MVector) -> None:
        self.vectors[vkey] = v.vector

    def distances(self, v: MVector):
        return norm((self.values() - v.vector), ord=2, axis=1)

    def nearest_distance(self, v: MVectorT) -> float:
        """
        指定ベクトル直近値

        Parameters
        ----------
        v : MVector
            比較対象ベクトル

        Returns
        -------
        float
            直近距離
        """
        return float(np.min(self.distances(v)))

    def nearest_value(self, v: MVectorT):
        """
        指定ベクトル直近値

        Parameters
        ----------
        v : MVector
            比較対象ベクトル

        Returns
        -------
        MVector
            直近値
        """
        return v.__class__(*np.array(self.values())[np.argmin(self.distances(v))])

    def nearest_key(self, v: MVectorT) -> np.ndarray:
        """
        指定ベクトル直近キー

        Parameters
        ----------
        v : MVector
            比較対象ベクトル

        Returns
        -------
        直近キー
        """
        return np.array(self.keys())[np.argmin(self.distances(v))]


@lru_cache(maxsize=None)
def cache_slerp_evaluate(q1: quaternion, q2: quaternion, t) -> quaternion:
    return slerp_evaluate(q1, q2, t)


class MQuaternion(MVector):
    """
    クォータニオンクラス
    """

    def __init__(
        self,
        scalar: float = 1.0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ):
        self.vector: quaternion = quaternion(scalar, x, y, z)

    @property
    def scalar(self) -> float:
        return self.vector.components[0]  # type: ignore

    @scalar.setter
    def scalar(self, v: float) -> None:
        self.vector.components[0] = v

    @property
    def x(self) -> float:
        return self.vector.components[1]  # type: ignore

    @x.setter
    def x(self, v: float) -> None:
        self.vector.components[1] = v

    @property
    def y(self) -> float:
        return self.vector.components[2]  # type: ignore

    @y.setter
    def y(self, v: float) -> None:
        self.vector.components[2] = v

    @property
    def z(self) -> float:
        return self.vector.components[3]  # type: ignore

    @z.setter
    def z(self, v: float) -> None:
        self.vector.components[3] = v

    @property
    def xyz(self) -> MVector3D:
        return MVector3D(*self.vector.components[1:])  # type: ignore

    @property
    def theta(self) -> float:
        return 2 * acos(min(1, max(-1, self.scalar)))

    @property
    def gl(self) -> "MQuaternion":
        return MQuaternion(-self.scalar, -self.x, self.y, self.z)

    def __bool__(self) -> bool:
        return qq_one != self.vector

    def __str__(self) -> str:
        return f"[x={round(self.x, 5)}, y={round(self.y, 5)}, " + f"z={round(self.z, 5)}, scalar={round(self.scalar, 5)}]"

    def effective(self, rtol: float = 1e-05, atol: float = 1e-08) -> "MQuaternion":
        vector = np.copy(self.vector.components)
        vector[np.isnan(vector)] = 0
        vector[np.isinf(vector)] = 0
        return MQuaternion(*vector)

    def length(self) -> float:
        """
        ベクトルの長さ
        """
        return float(self.vector.abs())  # type: ignore

    def length_squared(self) -> float:
        """
        ベクトルの長さの二乗
        """
        return float(self.vector.abs() ** 2)  # type: ignore

    def inverse(self) -> "MQuaternion":
        """
        逆回転
        """
        return MQuaternion(*self.vector.inverse().components)

    def normalized(self) -> "MQuaternion":
        """
        正規化した値を返す
        """
        if not self:
            return MQuaternion()

        v = self.effective()
        l2 = norm(v.vector.components, ord=2, axis=-1, keepdims=True)
        l2[l2 == 0] = 1
        normv = v.vector.components / l2
        return MQuaternion(*normv)

    def normalize(self) -> None:
        """
        自分自身の正規化
        """
        v = self.effective()
        l2 = norm(v.vector.components, ord=2, axis=-1, keepdims=True)
        l2[l2 == 0] = 1
        v.vector.components /= l2
        self.vector = v.vector

    def to_vector4(self) -> MVector4D:
        return MVector4D(self.x, self.y, self.z, self.scalar)

    def copy(self) -> "MQuaternion":
        return MQuaternion(self.scalar, self.x, self.y, self.z)

    def dot(self, v: "MQuaternion") -> float:
        return np.sum(self.vector.components * v.vector.components)

    def to_euler_degrees(self) -> MVector3D:
        """
        クォータニオンをオイラー角に変換する
        """
        if not self:
            return MVector3D()

        xx = self.x * self.x
        xy = self.x * self.y
        xz = self.x * self.z
        xw = self.x * self.scalar
        yy = self.y * self.y
        yz = self.y * self.z
        yw = self.y * self.scalar
        zz = self.z * self.z
        zw = self.z * self.scalar
        lengthSquared = xx + yy + zz + self.scalar**2

        if not np.isclose([lengthSquared, lengthSquared - 1.0], 0).any():
            xx, xy, xz, xw, yy, yz, yw, zz, zw = np.array([xx, xy, xz, xw, yy, yz, yw, zz, zw], dtype=np.float64) / lengthSquared

        pitch = np.arcsin(max(-1, min(1, -2.0 * (yz - xw))))
        yaw = 0
        roll = 0

        if pitch < (np.pi / 2):
            if pitch > -(np.pi / 2):
                yaw = np.arctan2(2.0 * (xz + yw), 1.0 - 2.0 * (xx + yy))
                roll = np.arctan2(2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz))
            else:
                # not a unique solution
                roll = 0
                yaw = -np.arctan2(-2.0 * (xy - zw), 1.0 - 2.0 * (yy + zz))
        else:
            # not a unique solution
            roll = 0
            yaw = np.arctan2(-2.0 * (xy - zw), 1.0 - 2.0 * (yy + zz))

        return MVector3D(*np.degrees([pitch, yaw, roll]))

    def to_euler_degrees_mmd(self) -> MVector3D:
        """
        MMDの表記に合わせたオイラー角
        """
        euler = self.to_euler_degrees()
        return MVector3D(euler.x, -euler.y, -euler.z)

    def to_degrees(self) -> float:
        """
        角度に変換
        """
        return degrees(self.theta)

    def to_signed_degrees(self, local_axis: MVector3D) -> float:
        """
        軸による符号付き角度に変換
        """
        deg = degrees(self.theta)
        sign = np.sign(self.xyz.dot(local_axis)) * np.sign(self.scalar)

        if sign != 0:
            deg *= sign

        if 180 < abs(deg):
            # 180度を超してる場合、フリップなので、除去
            return (abs(deg) - 180) * np.sign(deg)

        return deg

    def to_theta(self, v: "MQuaternion") -> float:
        """
        自分ともうひとつの値vとのtheta（変位量）を返す
        """
        return acos(min(1, max(-1, self.normalized().dot(v.normalized()))))

    def to_matrix4x4(self) -> "MMatrix4x4":
        if not self:
            return MMatrix4x4()

        mat3x3 = as_rotation_matrix(self.vector)
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = mat3x3.flatten()
        return MMatrix4x4(m00, m01, m02, 0.0, m10, m11, m12, 0.0, m20, m21, m22, 0.0, 0.0, 0.0, 0.0, 1.0)

    def to_matrix4x4_axis(self, local_x_axis: MVector3D, local_z_axis: MVector3D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self:
            return np.eye(4), np.eye(4), np.eye(4)

        # クォータニオンを回転行列に変換
        R = self.to_matrix4x4().vector

        x_axis = local_x_axis.copy().vector
        z_axis = local_z_axis.copy().vector

        # X軸回転行列を作成
        x_axis = x_axis / norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / norm(y_axis)
        z_axis = z_axis / norm(z_axis)
        # X = np.array([[1, 0, 0, 0], [0, y_axis[2], -y_axis[1], 0], [0, y_axis[1], y_axis[2], 0], [0, 0, 0, 1]])
        theta = np.arctan2(x_axis[1], x_axis[0])
        RX = np.array([[1, 0, 0, 0], [0, np.cos(theta), np.sin(theta), 0], [0, -np.sin(theta), np.cos(theta), 0], [0, 0, 0, 1]])

        # Z軸回転行列を作成
        z_axis_X = np.dot(RX, np.concatenate((z_axis.tolist(), [0.0])))
        theta = np.arctan2(z_axis_X[1], z_axis_X[0])
        RZ = np.array([[np.cos(theta), np.sin(theta), 0, 0], [-np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Y軸回転行列を作成
        RY = np.dot(np.dot(np.linalg.inv(RX), np.linalg.inv(RZ)), np.dot(np.dot(R, RZ), RX))

        return RX, RY, RZ

    def __mul__(self, other: Union[float, MVector3D, "MQuaternion"]):
        if isinstance(other, MVector3D):
            # quaternion と vec3 のかけ算は vec3 を返す
            return self.to_matrix4x4() * other
        elif isinstance(other, MQuaternion):
            mat = MMatrix4x4(*self.to_matrix4x4().vector.flatten())
            mat.rotate(other)

            return mat.to_quaternion()

        return MQuaternion(*(self.vector.components * other))

    def multiply_factor(self, factor: float):
        if factor == 0:
            return MQuaternion()
        return MQuaternion(self.scalar / factor, self.x, self.y, self.z)

    def to_fixed_axis_quaternion(self, fixed_axis: MVector3D) -> "MQuaternion":
        """
        軸制限されたクォータニオンの回転

        Parameters
        ----------
        fixed_axis : MVector3D
            軸制限を表す3次元ベクトル

        Returns
        -------
        MQuaternion
        """
        normalized_fixed_axis = fixed_axis.normalized()
        theta = acos(max(-1, min(1, normalized_fixed_axis.dot(self.xyz.normalized()))))
        fixed_qq_axis: MVector3D = normalized_fixed_axis * (1 if theta < pi / 2 else -1) * self.xyz.length()
        return MQuaternion(self.scalar, fixed_qq_axis.x, fixed_qq_axis.y, fixed_qq_axis.z).normalized()

    @staticmethod
    def from_euler_degrees(a: Union[int, float, MVector3D], b=0, c=0):
        """
        オイラー角をクォータニオンに変換する
        """
        euler = np.zeros(3)
        if isinstance(a, (int, float)):
            euler = np.radians([a, b, c], dtype=np.double)
        else:
            euler = np.radians([a.x, a.y, a.z], dtype=np.double)

        euler *= 0.5

        c1, c2, c3 = np.cos([euler[1], euler[2], euler[0]])
        s1, s2, s3 = np.sin([euler[1], euler[2], euler[0]])
        w = c1 * c2 * c3 + s1 * s2 * s3
        x = c1 * c2 * s3 + s1 * s2 * c3
        y = s1 * c2 * c3 - c1 * s2 * s3
        z = c1 * s2 * c3 - s1 * c2 * s3

        return MQuaternion(w, x, y, z)

    @staticmethod
    def from_axis_angles(v: MVector3D, degree: float):
        """
        軸と角度からクォータニオンに変換する
        """
        vv = v.normalized()
        length = sqrt(vv.x**2 + vv.y**2 + vv.z**2)

        xyz = vv.vector
        if not np.isclose([length - 1.0, length], 0).any():
            xyz /= length

        radian = radians(degree / 2.0)
        return MQuaternion(cos(radian), *(xyz * sin(radian))).normalized()

    @staticmethod
    def from_direction(direction: MVector3D, up: MVector3D):
        """
        軸と角度からクォータニオンに変換する
        """
        if np.isclose(direction.vector, 0).all():
            return MQuaternion()

        z_axis = direction.normalized()
        x_axis = up.cross(z_axis).normalized()

        if np.isclose(x_axis.length_squared(), 0).all():
            # collinear or invalid up vector derive shortest arc to new direction
            return MQuaternion.rotate(MVector3D(0.0, 0.0, 1.0), z_axis)

        y_axis = z_axis.cross(x_axis)

        return MQuaternion.from_axes(x_axis, y_axis, z_axis)

    @staticmethod
    def rotate(from_v: MVector3D, to_v: MVector3D):
        """
        fromベクトルからtoベクトルまでの回転量
        """
        v0 = from_v.normalized()
        v1 = to_v.normalized()
        d = v0.dot(v1) + 1.0

        # if dest vector is close to the inverse of source vector, ANY axis of rotation is valid
        if np.isclose(d, 0).all():
            axis = MVector3D(1.0, 0.0, 0.0).cross(v0)
            if np.isclose(axis.length_squared(), 0).all():
                axis = MVector3D(0.0, 1.0, 0.0).cross(v0)
            axis.normalize()
            # same as MQuaternion.fromAxisAndAngle(axis, 180.0)
            return MQuaternion(0.0, axis.x, axis.y, axis.z).normalized()

        d = sqrt(2.0 * d)
        axis = v0.cross(v1) / d
        return MQuaternion(d * 0.5, axis.x, axis.y, axis.z).normalized()

    @staticmethod
    def from_axes(x_axis: MVector3D, y_axis: MVector3D, z_axis: MVector3D):
        return MQuaternion(
            *from_rotation_matrix(
                np.array(
                    [
                        [x_axis.x, y_axis.x, z_axis.x],
                        [x_axis.y, y_axis.y, z_axis.y],
                        [x_axis.z, y_axis.z, z_axis.z],
                    ],
                    dtype=np.float64,
                )
            ).components
        )

    @staticmethod
    def nlerp(q1: "MQuaternion", q2: "MQuaternion", t: float):
        """
        線形補間
        """
        # Handle the easy cases first.
        if 0.0 >= t:
            return q1
        elif 1.0 <= t:
            return q2

        q2b = MQuaternion(*q2.vector.components)
        d = q1.dot(q2)

        if 0.0 > d:
            q2b = -q2b

        return MQuaternion(*(q1.vector.components * (1.0 - t) + q2b.vector.components * t)).normalized()

    @staticmethod
    def slerp(q1: "MQuaternion", q2: "MQuaternion", t: float):
        """
        球形補間
        """
        return MQuaternion(*cache_slerp_evaluate(q1.vector, q2.vector, t).components)

    def separate_by_axis(self, global_axis: MVector3D):
        # ローカルZ軸ベースで求める場合
        local_z_axis = MVector3D(0, 0, -1)
        # X軸ベクトル
        global_x_axis = global_axis.normalized()
        # Y軸ベクトル
        global_y_axis = local_z_axis.cross(global_x_axis)
        if not global_y_axis:
            # ローカルZ軸ベースで求めるのに失敗した場合、ローカルY軸ベースで求め直す
            local_y_axis = MVector3D(0, 1, 0)
            # Z軸ベクトル
            global_z_axis = local_y_axis.cross(global_x_axis)
            # Y軸ベクトル
            global_y_axis = global_x_axis.cross(global_z_axis)
        else:
            # Z軸ベクトル
            global_z_axis = global_x_axis.cross(global_y_axis)

        # X成分を抽出する ------------

        # グローバル軸方向に伸ばす
        global_x_vec = self * global_x_axis

        # YZの回転量（自身のねじれを無視する）
        yz_qq = MQuaternion.rotate(global_x_axis, global_x_vec.normalized())

        # 元々の回転量 から YZ回転 を除去して、除去されたX成分を求める
        x_qq = self * yz_qq.inverse()

        # Y成分を抽出する ------------

        # グローバル軸方向に伸ばす
        global_y_vec = self * global_y_axis

        # XZの回転量（自身のねじれを無視する）
        xz_qq = MQuaternion.rotate(global_y_axis, global_y_vec.normalized())

        # 元々の回転量 から XZ回転 を除去して、除去されたY成分を求める
        y_qq = self * xz_qq.inverse()

        # Z成分を抽出する ------------

        # グローバル軸方向に伸ばす
        global_z_vec = self * global_z_axis

        # XYの回転量（自身のねじれを無視する）
        xy_qq = MQuaternion.rotate(global_z_axis, global_z_vec.normalized())

        # 元々の回転量 から XY回転 を除去して、除去されたZ成分を求める
        z_qq = self * xy_qq.inverse()

        return x_qq, y_qq, z_qq

    def separate_euler_degrees(self) -> MVector3D:
        """
        ZXYの回転順序でオイラー角度を求める
        https://programming-surgeon.com/script/euler-python-script/

        Returns
        -------
        ZXYローカル軸別のオイラー角度
        """
        mat = self.normalized().to_matrix4x4()
        z_radian = atan2(-mat[0, 1], mat[0, 0])
        x_radian = atan2(mat[2, 1] * cos(z_radian), mat[1, 1])
        y_radian = atan2(-mat[2, 0], mat[2, 2])

        return MVector3D(*np.degrees([x_radian, y_radian, z_radian]).tolist())


class MMatrix4x4(MVector):
    """
    4x4行列クラス
    """

    def __init__(
        self,
        m11: float = 1.0,
        m12: float = 0.0,
        m13: float = 0.0,
        m14: float = 0.0,
        m21: float = 0.0,
        m22: float = 1.0,
        m23: float = 0.0,
        m24: float = 0.0,
        m31: float = 0.0,
        m32: float = 0.0,
        m33: float = 1.0,
        m34: float = 0.0,
        m41: float = 0.0,
        m42: float = 0.0,
        m43: float = 0.0,
        m44: float = 1.0,
    ):
        self.vector = np.array(
            [m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34, m41, m42, m43, m44],
            dtype=np.float64,
        ).reshape(4, 4)

    def inverse(self) -> "MMatrix4x4":
        """
        逆行列
        """
        if self:
            return MMatrix4x4(*inv(self.vector).flatten())

        return MMatrix4x4()

    def rotate(self, q: MQuaternion) -> None:
        """
        回転行列
        """
        self.vector = self.vector @ q.to_matrix4x4().vector

    def rotate_x(self, q: MQuaternion) -> None:
        """
        X軸周りの回転行列
        """
        theta = q.theta
        yy = cos(theta)
        yz = -sin(theta)
        zy = sin(theta)
        zz = cos(theta)

        c = (yy + zz) / 2
        s = (yz - zy) / 2

        norm = sqrt(c**2 + s**2)

        if norm != 0:
            c /= norm
            s /= norm

        mat = np.eye(4, dtype=np.float64)
        mat[1, 1] = c
        mat[1, 2] = s
        mat[2, 1] = -s
        mat[2, 2] = c

        self.vector = self.vector @ mat

    def rotate_y(self, q: MQuaternion) -> None:
        """
        Y軸周りの回転行列
        """
        theta = q.theta
        xx = cos(theta)
        xz = sin(theta)
        zx = -sin(theta)
        zz = cos(theta)

        c = (xx + zz) / 2
        s = (xz - zx) / 2

        norm = sqrt(c**2 + s**2)

        if norm != 0:
            c /= norm
            s /= norm

        mat = np.eye(4, dtype=np.float64)
        mat[0, 0] = c
        mat[0, 2] = s
        mat[2, 0] = -s
        mat[2, 2] = c

        self.vector = self.vector @ mat

    def rotate_z(self, q: MQuaternion) -> None:
        """
        Z軸周りの回転行列
        """
        theta = q.theta
        xx = cos(theta)
        xy = -sin(theta)
        yx = sin(theta)
        yy = cos(theta)

        c = (xx + yy) / 2
        s = (xy - yx) / 2

        norm = sqrt(c**2 + s**2)

        if norm != 0:
            c /= norm
            s /= norm

        mat = np.eye(4, dtype=np.float64)
        mat[0, 0] = c
        mat[0, 1] = -s
        mat[1, 0] = s
        mat[1, 1] = c

        self.vector = self.vector @ mat

    def translate(self, v: MVector3D) -> None:
        """
        平行移動行列
        """
        vmat = np.eye(4)
        vmat[:3, 3] = v.vector
        self.vector = self.vector @ vmat

    def scale(self, v: Union[MVector3D, float]) -> None:
        """
        縮尺行列
        """
        vmat = np.eye(4)
        if isinstance(v, MVector3D):
            vmat = vmat * np.array([*v.vector, 1])
        else:
            vmat = vmat * v

        self.vector = self.vector @ vmat

    def identity(self) -> None:
        """
        初期化
        """
        self.vector = np.eye(4, dtype=np.float64)

    def look_at(self, eye: MVector3D, center: MVector3D, up: MVector3D) -> None:
        forward = center - eye
        forward.normalize()
        if np.isclose(forward.vector, 0).all():
            return

        side = forward.cross(up).normalized()
        upv = side.cross(forward).normalized()

        m = MMatrix4x4()
        m.vector[0, :-1] = side.vector
        m.vector[1, :-1] = upv.vector
        m.vector[2, :-1] = -forward.vector
        m.vector[-1, -1] = 1.0

        self *= m
        self.translate(-eye)

    def perspective(
        self,
        vertical_angle: float,
        aspect_ratio: float,
        near_plane: float,
        far_plane: float,
    ) -> None:
        """
        パースペクティブ行列
        """
        if near_plane == far_plane or aspect_ratio == 0:
            return

        rad = radians(vertical_angle / 2)
        sine = sin(rad)

        if sine == 0:
            return

        cotan = cos(rad) / sine
        clip = far_plane - near_plane

        m = MMatrix4x4()
        m.vector[0, 0] = cotan / aspect_ratio
        m.vector[1, 1] = cotan
        m.vector[2, 2] = -(near_plane + far_plane) / clip
        m.vector[2, 3] = -(2 * near_plane * far_plane) / clip
        m.vector[3, 2] = -1

        self *= m

    def map_vector(self, v: MVector3D) -> MVector3D:
        return MVector3D(*np.sum(v.vector * self.vector[:3, :3], axis=1))

    def to_quaternion(self) -> "MQuaternion":
        q = MQuaternion()
        v = self.vector

        # I removed + 1
        trace = v[0, 0] + v[1, 1] + v[2, 2]
        # I changed M_EPSILON to 0
        if 0 < trace:
            s = 0.5 / sqrt(trace + 1)
            q.scalar = 0.25 / s
            q.x = (v[2, 1] - v[1, 2]) * s
            q.y = (v[0, 2] - v[2, 0]) * s
            q.z = (v[1, 0] - v[0, 1]) * s
        else:
            if v[0, 0] > v[1, 1] and v[0, 0] > v[2, 2]:
                s = 2 * sqrt(1 + v[0, 0] - v[1, 1] - v[2, 2])
                q.scalar = (v[2, 1] - v[1, 2]) / s
                q.x = 0.25 * s
                q.y = (v[0, 1] + v[1, 0]) / s
                q.z = (v[0, 2] + v[2, 0]) / s
            elif v[1, 1] > v[2, 2]:
                s = 2 * sqrt(1 + v[1, 1] - v[0, 0] - v[2, 2])
                q.scalar = (v[0, 2] - v[2, 0]) / s
                q.x = (v[0, 1] + v[1, 0]) / s
                q.y = 0.25 * s
                q.z = (v[1, 2] + v[2, 1]) / s
            else:
                s = 2 * sqrt(1 + v[2, 2] - v[0, 0] - v[1, 1])
                q.scalar = (v[1, 0] - v[0, 1]) / s
                q.x = (v[0, 2] + v[2, 0]) / s
                q.y = (v[1, 2] + v[2, 1]) / s
                q.z = 0.25 * s

        q.normalize()

        return q

    def to_position(self) -> MVector3D:
        return MVector3D(*self.vector[:3, 3])

    def __mul__(self, other: Union["MMatrix4x4", "MVector3D", "MVector4D", float]):
        if isinstance(other, MMatrix4x4):
            # 行列同士のかけ算は matmul で演算
            raise ValueError("MMatrix4x4同士のかけ算は @ を使って下さい")
        elif isinstance(other, MVector3D):
            # vec3 とのかけ算は vec3 を返す
            s = np.sum(self.vector[:, :3] * other.vector, axis=1) + self.vector[:, 3]
            if s[3] == 1.0:
                return MVector3D(*s[:3])
            elif s[3] == 0.0:
                return MVector3D()
            else:
                return MVector3D(*(s[:3] / s[3]))
        elif isinstance(other, MVector4D):
            # vec4 とのかけ算は vec4 を返す
            return MVector4D(*np.sum(self.vector * other.vector, axis=1))
        return super().__mul__(other)

    def __matmul__(self, other):
        # 行列同士のかけ算
        return MMatrix4x4(np.matmul(self.vector, other.vector))

    def __imatmul__(self, other):
        # 行列同士のかけ算代入
        self.vector = np.matmul(self.vector, other.vector)
        return self

    def __getitem__(self, index) -> float:
        y, x = index
        return self.vector[y, x]

    def __setitem__(self, index, v: float) -> None:
        y, x = index
        self.vector[y, x] = v

    def __bool__(self) -> bool:
        return bool(not (self.vector == np.eye(4)).all())

    def copy(self) -> "MMatrix4x4":
        return self.__class__(*self.vector.flatten())


class MMatrix4x4List:
    """
    4x4行列クラスリスト
    """

    __slots__ = (
        "vector",
        "row",
        "col",
    )

    def __init__(self, row: int, col: int):
        """
        指定した行列の数だけ多次元Matrixを作成

        Parameters
        ----------
        row : int
            列数（キーフレ数）
        col : int
            行数（ボーン数）
        """
        self.row: int = row
        self.col: int = col
        self.vector: np.ndarray = np.tile(np.eye(4, dtype=np.float64), (row, col, 1, 1))

    def translate(self, vs: list[list[np.ndarray]]):
        """
        平行移動行列

        Parameters
        ----------
        vs : list[list[np.ndarray]]
            ベクトル(v.vector)
        """
        vmat = self.vector[..., :3] * np.array([v2 for v1 in vs for v2 in v1], dtype=np.float64).reshape(self.row, self.col, 1, 3)
        self.vector[..., 3] += np.sum(vmat, axis=-1)

    def rotate(self, qs: list[list[np.ndarray]]):
        """
        回転行列

        Parameters
        ----------
        qs : list[list[np.ndarray]]
            クォータニオンの回転行列(qq.to_matrix4x4().vector)
        """

        self.vector = self.vector @ np.array([q2 for q1 in qs for q2 in q1], dtype=np.float64).reshape(self.row, self.col, 4, 4)

    def scale(self, vs: list[list[np.ndarray]]):
        """
        縮尺行列

        Parameters
        ----------
        vs : list[list[np.ndarray]]
            ベクトル(v.vector)
        """
        # vec4に変換
        ones = np.ones((self.row, self.col, 1))
        vs4: np.ndarray = np.concatenate((vs, ones.tolist()), axis=2).reshape(self.row, self.col, 4, 1)
        # スケール行列に変換
        mat4 = np.full((self.row, self.col, 4, 4), np.eye(4)) * vs4

        self.vector = self.vector @ mat4

    def add(self, vs: np.ndarray):
        """
        行列をそのまま加算する

        Parameters
        ----------
        vs : list[list[np.ndarray]]
            ローカル行列
        """

        self.vector = self.vector + vs

    def matmul(self, vs: np.ndarray):
        """
        行列をそのままかける

        Parameters
        ----------
        vs : list[list[np.ndarray]]
            ローカル行列
        """

        self.vector = self.vector @ vs

    def inverse(self):
        """
        逆行列
        """
        new_mat = MMatrix4x4List(self.row, self.col)
        new_mat.vector = inv(self.vector)
        return new_mat

    def __matmul__(self, other):
        # 行列同士のかけ算
        new_mat = MMatrix4x4List(self.row, self.col)
        new_mat.vector = self.vector @ other.vector
        return new_mat

    def __imatmul__(self, other):
        # 行列同士のかけ算代入
        new_mat = MMatrix4x4List(self.row, self.col)
        new_mat.vector = self.vector @ other.vector
        self.vector = new_mat.vector
        return self

    def matmul_cols(self):
        # colを 行列積 するため、ひとつ次元を増やす
        tile_mats = np.tile(np.eye(4, dtype=np.float64), (self.row, self.col, self.col, 1, 1))
        # 斜めにセルを埋めていく
        for c in range(self.col):
            tile_mats[:, c:, c, :, :] = np.tile(self.vector[:, c], (self.col - c, 1)).reshape(self.row, self.col - c, 4, 4)
        # 行列積を求める
        result_mats = MMatrix4x4List(self.row, self.col)
        result_mats.vector = np.tile(np.eye(4, dtype=np.float64), (self.row, self.col, 1, 1))
        result_mats.vector = tile_mats[:, :, 0]
        for c in range(1, self.col):
            result_mats.vector = np.matmul(result_mats.vector, tile_mats[:, :, c])

        return result_mats

    def to_positions(self) -> np.ndarray:
        # 行列計算結果の位置
        return self.vector[..., :3, 3]

    def __getitem__(self, index) -> np.ndarray:
        y, x = index
        return self.vector[y, x]


def operate_vector(v: MVectorT, other: Union[MVectorT, float, int], op) -> MVectorT:
    """
    演算処理

    Parameters
    ----------
    v : MVector
        計算主対象
    other : Union[MVector, float, int]
        演算対象
    op : 演算処理

    Returns
    -------
    MVector
        演算結果
    """
    if isinstance(other, MVector):
        v1 = op(v.vector, other.vector)
    else:
        v1 = op(v.vector, other)

    if isinstance(v1, quaternion):
        v2 = v.__class__(*v1.components)
    else:
        v2 = v.__class__(*v1)
    return v2.effective()


def intersect_line_plane(line_point: MVector3D, line_direction: MVector3D, plane_point: MVector3D, plane_normal: MVector3D) -> MVector3D:
    """
    直線と平面の交点を求める処理

    Parameters
    ----------
    line_point : 直線上の任意の1点
    line_direction : 直線の方向ベクトル
    plane_point : 平面上の任意の1点
    plane_normal : 平面の法線ベクトル

    Returns
    -------
    交点位置
    """

    # 直線の方向ベクトルを正規化する
    line_direction.normalize()

    # 平面の法線ベクトルを正規化する
    plane_normal.normalize()

    denominator = line_direction.dot(plane_normal)

    # 直線と平面が平行である場合は交点が存在しない
    if np.abs(denominator) < 1e-6:
        return MVector3D()

    # 直線と平面が交わる点の位置ベクトルを計算する
    t = plane_normal.dot(plane_point - line_point) / denominator
    intersection = line_point + line_direction * t

    return intersection


import numpy as np


def align_triangle(
    a1: MVector3D,
    a2: MVector3D,
    a3: MVector3D,
    b1: MVector3D,
    b2: MVector3D,
) -> MVector3D:
    """
    三角形Aに形を合わせた三角形BのためのB3を求める（B1, B2は固定とする）

    Parameters
    ----------
    A1 : MVector3D
    A2 : MVector3D
    A3 : MVector3D
    B1 : MVector3D
    B2 : MVector3D

    Returns
    -------
    the new positions of B2 and B3 (3D vectors)
    """
    A1, A2, A3, B1, B2 = np.array([a1.vector, a2.vector, a3.vector, b1.vector, b2.vector])

    # A-1, A-2, B-1, B-2を固定したベクトルを作成
    v1 = A2 - A1
    v2 = B2 - B1

    # B-3'を求めるためのベクトルを計算
    v3_prime = (A3 - A1) / norm(v1) * norm(v2)

    # B-3'をB-1, B-2からオフセット
    B3_prime = B1 + v3_prime

    return MVector3D(*B3_prime)
