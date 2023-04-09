from functools import lru_cache
from typing import List, Tuple

import bezier
import numpy as np

from mlib.base.base import BaseModel
from mlib.base.math import MVector2D

# MMDでの補間曲線の最大値
IP_MAX = 127


class Interpolation(BaseModel):
    __slots__ = ["begin", "start", "end", "finish"]

    def __init__(self):
        """
        補間曲線
        """
        self.begin = MVector2D(0, 0)
        self.start = MVector2D(20, 20)
        self.end = MVector2D(107, 107)
        self.finish = MVector2D(IP_MAX, IP_MAX)

    def normalize(self):
        diff = self.finish - self.begin
        self.begin = Interpolation.round_mmd((self.begin - self.begin) / diff, MVector2D())
        self.start = Interpolation.round_mmd((self.start - self.begin) / diff, MVector2D())
        self.end = Interpolation.round_mmd((self.end - self.begin) / diff, MVector2D(IP_MAX, IP_MAX))
        self.finish = Interpolation.round_mmd((self.finish - self.begin) / diff, MVector2D(IP_MAX, IP_MAX))

    @classmethod
    def round_mmd(cls, t: MVector2D, s: MVector2D) -> MVector2D:
        t.x = Interpolation.round(t.x * IP_MAX)
        t.y = Interpolation.round(t.y * IP_MAX)

        if not (0 <= t.x and 0 <= t.y):
            # 範囲に収まってない場合、縮める
            v = (t - (t * ((s - t) / np.max((s - t).vector)))) * 0.95
            t.x = Interpolation.round(v.x)
            t.y = Interpolation.round(v.y)

        elif not (t.x <= IP_MAX and t.y <= IP_MAX):
            # 範囲に収まってない場合、縮める
            v = (t * IP_MAX / np.max(t.vector)) * 0.95
            t.x = Interpolation.round(v.x)
            t.y = Interpolation.round(v.y)

        return t

    @classmethod
    def round(cls, t: float) -> int:
        t2 = t * 1000000
        # pythonは偶数丸めなので、整数部で丸めた後、元に戻す
        return int(round(round(t2, -6) / 1000000))


def get_infections(values: List[float], threshold: float) -> np.ndarray:
    extract_idxs = get_threshold_infections(np.fromiter(values, dtype=np.float64, count=len(values)), threshold)
    if 2 > len(extract_idxs):
        return np.array([])
    extracts = np.fromiter(values, dtype=np.float64, count=len(values))[extract_idxs]
    f_prime = np.gradient(extracts)
    infections = extract_idxs[np.where(np.diff(np.sign(f_prime)))[0]]
    return infections


def get_fix_infections(values: List[float]) -> np.ndarray:
    return np.where(np.diff(np.where(np.isclose(np.abs(np.diff(values)), 0.0))[0]) > 2)[0]


def get_threshold_infections(values: np.ndarray, threshold: float) -> np.ndarray:
    extract_idxs = []
    start_idx = 0
    end_idx = 1
    while end_idx <= len(values) - 1:
        diff = np.abs(values[start_idx:end_idx]).ptp()
        if diff >= threshold:
            extract_idxs.append(start_idx)
            extract_idxs.append(end_idx - 1)
            start_idx = end_idx - 1
        else:
            end_idx += 1
    return np.fromiter(sorted(list(set(extract_idxs))), dtype=np.float64, count=len(extract_idxs))


def create_interpolation(values: List[float]):
    if 2 >= len(values) or 0.0001 >= abs(np.max(values) - np.min(values)):
        return Interpolation()

    # Xは次数（フレーム数）分移動
    xs = np.arange(0, len(values))

    # YはXの移動分を許容範囲とする
    ys = np.fromiter(sorted(list(set(values))), dtype=np.float64, count=len(values))

    # https://github.com/dhermes/bezier/issues/242
    s_vals = np.linspace(0, 1, len(values))
    representative = bezier.Curve.from_nodes(np.eye(4))
    transform = representative.evaluate_multi(s_vals).T
    nodes = np.vstack([xs, ys])
    reduced_t, _, _, _ = np.linalg.lstsq(transform, nodes.T, rcond=None)
    reduced = reduced_t.T
    joined_curve = bezier.Curve.from_nodes(reduced)

    nodes = joined_curve.nodes

    # 次数を減らしたベジェ曲線をMMD用補間曲線に変換
    org_ip = Interpolation()
    org_ip.begin = MVector2D(nodes[0, 0], nodes[1, 0])
    org_ip.start = MVector2D(nodes[0, 1], nodes[1, 1])
    org_ip.end = MVector2D(nodes[0, 2], nodes[1, 2])
    org_ip.finish = MVector2D(nodes[0, 3], nodes[1, 3])
    org_ip.normalize()

    return org_ip


# https://pomax.github.io/bezierinfo
# https://shspage.hatenadiary.org/entry/20140625/1403702735
# https://bezier.readthedocs.io/en/stable/python/reference/bezier.curve.html#bezier.curve.Curve.evaluate
# https://edvakf.hatenadiary.org/entry/20111016/1318716097
def evaluate(interpolation: Interpolation, start: int, now: int, end: int) -> Tuple[float, float, float]:
    """
    補間曲線を求める

    Parameters
    ----------
    interpolation : Interpolation
        補間曲線
    start : int
        開始キーフレ
    now : int
        計算キーフレ
    end : int
        終端キーフレ

    Returns
    -------
    tuple[float, float, float]
        x（計算キーフレ時点のX値）, y（計算キーフレ時点のY値）, t（計算キーフレまでの変化量）
    """
    if 0 == (now - start) or 0 == (end - start):
        return 0.0, 0.0, 0.0

    x = (now - start) / (end - start)
    x1 = interpolation.start.x / IP_MAX
    y1 = interpolation.start.y / IP_MAX
    x2 = interpolation.end.x / IP_MAX
    y2 = interpolation.end.y / IP_MAX

    if 1 <= x:
        return 1.0, 1.0, 1.0

    return cache_evaluate(x, x1, y1, x2, y2)


@lru_cache(maxsize=None)
def cache_evaluate(x: float, x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float]:
    t = newton(x1, x2, x)
    s = 1 - t

    y = (3 * (s**2) * t * y1) + (3 * s * (t**2) * y2) + (t**3)

    return x, y, t


# 解を求める関数
def func_f(x1: float, x2: float, x: float, t: float):
    t1 = 1 - t
    return 3 * (t1**2) * t * x1 + 3 * t1 * (t**2) * x2 + (t**3) - x


@lru_cache(maxsize=None)
def cached_func_f(x1, x2, x, t):
    return func_f(x1, x2, x, t)


# Newton法（方程式の関数項、探索の開始点、微小量、誤差範囲、最大反復回数）
def newton(x1, x2, x, t0=0.5, eps=1e-10, error=1e-10):
    derivative = 2 * eps
    for _ in range(10):
        func_f_value = cached_func_f(x1, x2, x, t0)
        # 中心差分による微分値
        func_df = (cached_func_f(x1, x2, x, t0 + eps) - cached_func_f(x1, x2, x, t0 - eps)) / derivative
        if eps >= abs(func_df):
            break
        # 次の解を計算
        t1 = t0 - func_f_value / func_df
        if error >= abs(t1 - t0):
            # 「誤差範囲が一定値以下」ならば終了
            break
        # 解を更新
        t0 = t1
    return t0
