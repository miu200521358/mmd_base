from mlib.base.base import BaseModel
from mlib.math import MVector2D

# MMDでの補間曲線の最大値
INTERPOLATION_MMD_MAX = 127


class Interpolation(BaseModel):
    def __init__(
        self,
        start: MVector2D = None,
        end: MVector2D = None,
    ):
        """
        補間曲線

        Parameters
        ----------
        start : MVector2D, optional
            補間曲線開始, by default None
        end : MVector2D, optional
            補間曲線終了, by default None
        """
        self.start: MVector2D = start or MVector2D()
        self.end: MVector2D = end or MVector2D()


# http://d.hatena.ne.jp/edvakf/20111016/1318716097
# https://pomax.github.io/bezierinfo
# https://shspage.hatenadiary.org/entry/20140625/1403702735
# https://bezier.readthedocs.io/en/stable/python/reference/bezier.curve.html#bezier.curve.Curve.evaluate
def evaluate(
    interpolation: Interpolation, start: int, now: int, end: int
) -> tuple[float, float, float]:
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
    if (now - start) == 0 or (end - start) == 0:
        return 0, 0, 0

    x = (now - start) / (end - start)
    x1 = interpolation.start.x / INTERPOLATION_MMD_MAX
    y1 = interpolation.start.y / INTERPOLATION_MMD_MAX
    x2 = interpolation.end.x / INTERPOLATION_MMD_MAX
    y2 = interpolation.end.y / INTERPOLATION_MMD_MAX

    t = 0.5
    s = 0.5

    # 二分法
    for i in range(15):
        ft = (3 * (s * s) * t * x1) + (3 * s * (t * t) * x2) + (t * t * t) - x

        if ft > 0:
            t -= 1 / (4 << i)
        else:
            t += 1 / (4 << i)

        s = 1 - t

    y = (3 * (s * s) * t * y1) + (3 * s * (t * t) * y2) + (t * t * t)

    return x, y, t
