from typing import Optional

from mlib.base.base import BaseModel
from mlib.base.bezier import Interpolation, evaluate
from mlib.base.math import MQuaternion, MVector3D
from mlib.base.part import BaseIndexNameModel, BaseRotationModel


class BaseVmdNameFrame(BaseIndexNameModel):
    """
    VMD用基底クラス(名前あり)

    Parameters
    ----------
    index : int, optional
        キーフレ, by default None
    name : str, optional
        名前, by default None
    register : bool, optional
        登録対象か否か, by default None
    read : bool, optional
        VMDデータから読み込んだデータか, by default None
    """

    def __init__(
        self,
        index: int = -1,
        name: str = "",
        register: bool = False,
        read: bool = False,
    ):
        super().__init__(index, name)
        self.register = register
        self.read = read


# https://hariganep.seesaa.net/article/201103article_1.html
class BoneInterpolations(BaseModel):
    """
    ボーンキーフレ用補間曲線

    Parameters
    ----------
    translation_x : Interpolation, optional
        移動X, by default None
    translation_y : Interpolation, optional
        移動Y, by default None
    translation_z : Interpolation, optional
        移動Z, by default None
    rotation : Interpolation, optional
        回転, by default None
    """

    __slots__ = [
        "translation_x",
        "translation_y",
        "translation_z",
        "rotation",
        "vals",
    ]

    def __init__(
        self,
    ):
        self.translation_x = Interpolation()
        self.translation_y = Interpolation()
        self.translation_z = Interpolation()
        self.rotation = Interpolation()
        self.vals = [
            20,
            20,
            0,
            0,
            20,
            20,
            20,
            20,
            107,
            107,
            107,
            107,
            107,
            107,
            107,
            107,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            107,
            107,
            107,
            107,
            107,
            107,
            107,
            107,
            0,
            20,
            20,
            20,
            20,
            20,
            20,
            107,
            107,
            107,
            107,
            107,
            107,
            107,
            107,
            0,
            0,
            20,
            20,
            20,
            20,
            20,
            107,
            107,
            107,
            107,
            107,
            107,
            107,
            107,
            0,
            0,
            0,
        ]

    def evaluate(self, prev_index: int, index: int, next_index: int) -> tuple[float, float, float, float]:
        # 補間結果Yは、FKキーフレ内で計算する
        _, ry, _ = evaluate(self.rotation, prev_index, index, next_index)
        _, xy, _ = evaluate(self.translation_x, prev_index, index, next_index)
        _, yy, _ = evaluate(self.translation_y, prev_index, index, next_index)
        _, zy, _ = evaluate(self.translation_z, prev_index, index, next_index)

        return ry, xy, yy, zy

    def merge(self) -> list[int]:
        return [
            self.translation_x.start.x,
            self.vals[1],
            self.vals[2],
            self.vals[3],
            self.translation_x.start.y,
            self.vals[5],
            self.vals[6],
            self.vals[7],
            self.translation_x.end.x,
            self.vals[9],
            self.vals[10],
            self.vals[11],
            self.translation_x.end.y,
            self.vals[13],
            self.vals[14],
            self.vals[15],
            self.translation_y.start.x,
            self.vals[17],
            self.vals[18],
            self.vals[19],
            self.translation_y.start.y,
            self.vals[21],
            self.vals[22],
            self.vals[23],
            self.translation_y.end.x,
            self.vals[25],
            self.vals[26],
            self.vals[27],
            self.translation_y.end.y,
            self.vals[29],
            self.vals[30],
            self.vals[31],
            self.translation_z.start.x,
            self.vals[33],
            self.vals[34],
            self.vals[35],
            self.translation_z.start.y,
            self.vals[37],
            self.vals[38],
            self.vals[39],
            self.translation_z.end.x,
            self.vals[41],
            self.vals[42],
            self.vals[43],
            self.translation_z.end.y,
            self.vals[45],
            self.vals[46],
            self.vals[47],
            self.rotation.start.x,
            self.vals[49],
            self.vals[50],
            self.vals[51],
            self.rotation.start.y,
            self.vals[53],
            self.vals[54],
            self.vals[55],
            self.rotation.end.x,
            self.vals[57],
            self.vals[58],
            self.vals[59],
            self.rotation.end.y,
            self.vals[61],
            self.vals[62],
            self.vals[63],
        ]


class VmdBoneFrame(BaseVmdNameFrame):
    """
    VMDのボーン1フレーム

    Parameters
    ----------
    name : str, optional
        ボーン名, by default None
    index : int, optional
        キーフレ, by default None
    position : MVector3D, optional
        位置, by default None
    rotation : MQuaternion, optional
        回転, by default None
    interpolations : Interpolations, optional
        補間曲線, by default None
    register : bool, optional
        登録対象か否か, by default None
    read : bool, optional
        VMDデータから読み込んだデータか, by default None
    """

    __slots__ = [
        "name",
        "index",
        "register",
        "read",
        "position",
        "rotation",
        "interpolations",
        "ik_rotation",
        "ik_target_rotation",
        "correct_rotation",
    ]

    def __init__(
        self,
        name: str = "",
        index: int = -1,
        register: bool = False,
        read: bool = False,
    ):
        super().__init__(index, name, register, read)
        self.position = MVector3D()
        self.rotation = MQuaternion()
        self.interpolations = BoneInterpolations()
        self.ik_rotation: Optional[MQuaternion] = None
        self.correct_rotation: Optional[MQuaternion] = None


class VmdMorphFrame(BaseVmdNameFrame):
    """
    VMDのモーフ1フレーム

    Parameters
    ----------
    name : str, optional
        モーフ名, by default None
    index : int, optional
        キーフレ, by default None
    ratio : float, optional
        変化量, by default None
    register : bool, optional
        登録対象か否か, by default None
    read : bool, optional
        VMDデータから読み込んだデータか, by default None
    """

    __slots__ = [
        "name",
        "index",
        "register",
        "read",
        "ratio",
    ]

    def __init__(
        self,
        index: int = -1,
        name: str = "",
        ratio: float = 0.0,
        register: bool = False,
        read: bool = False,
    ):
        super().__init__(index, name, register, read)
        self.ratio = ratio


class CameraInterpolations(BaseModel):
    """
    カメラ補間曲線

    Parameters
    ----------
    translation_x : Interpolation, optional
        移動X, by default None
    translation_y : Interpolation, optional
        移動Y, by default None
    translation_z : Interpolation, optional
        移動Z, by default None
    rotation : Interpolation, optional
        回転, by default None
    distance : Interpolation, optional
        距離, by default None
    viewing_angle : Interpolation, optional
        視野角, by default None
    """

    __slots__ = [
        "translation_x",
        "translation_y",
        "translation_z",
        "rotation",
        "distance",
        "viewing_angle",
    ]

    def __init__(
        self,
        translation_x: Optional[Interpolation] = None,
        translation_y: Optional[Interpolation] = None,
        translation_z: Optional[Interpolation] = None,
        rotation: Optional[Interpolation] = None,
        distance: Optional[Interpolation] = None,
        viewing_angle: Optional[Interpolation] = None,
    ):
        self.translation_x = translation_x or Interpolation()
        self.translation_y = translation_y or Interpolation()
        self.translation_z = translation_z or Interpolation()
        self.rotation = rotation or Interpolation()
        self.distance = distance or Interpolation()
        self.viewing_angle = viewing_angle or Interpolation()


class VmdCameraFrame(BaseVmdNameFrame):
    """
    カメラキーフレ

    Parameters
    ----------
    index : int, optional
        キーフレ, by default None
    position : MVector3D, optional
        位置, by default None
    rotation : BaseRotationModel, optional
        回転, by default None
    distance : float, optional
        距離, by default None
    viewing_angle : int, optional
        視野角, by default None
    perspective : bool, optional
        パース, by default None
    interpolations : CameraInterpolations, optional
        補間曲線, by default None
    register : bool, optional
        登録対象か否か, by default None
    read : bool, optional
        VMDデータから読み込んだデータか, by default None
    """

    __slots__ = [
        "index",
        "register",
        "read",
        "position",
        "rotation",
        "distance",
        "viewing_angle",
        "perspective",
        "interpolations",
    ]

    def __init__(
        self,
        index: int = -1,
        position: Optional[MVector3D] = None,
        rotation: Optional[BaseRotationModel] = None,
        distance: float = 0.0,
        viewing_angle: int = 0,
        perspective: bool = False,
        interpolations: Optional[CameraInterpolations] = None,
        register: bool = False,
        read: bool = False,
    ):
        super().__init__(index, "カメラ", register, read)
        self.position = position or MVector3D()
        self.rotation = rotation or BaseRotationModel()
        self.distance = distance
        self.viewing_angle = viewing_angle
        self.perspective = perspective
        self.interpolations = interpolations or CameraInterpolations()


class VmdLightFrame(BaseVmdNameFrame):
    """
    照明キーフレ

    Parameters
    ----------
    index : int, optional
        キーフレ, by default None
    color : MVector3D, optional
        色, by default None
    position : MVector3D, optional
        位置, by default None
    register : bool, optional
        登録対象か否か, by default None
    read : bool, optional
        VMDデータから読み込んだデータか, by default None
    """

    __slots__ = [
        "index",
        "register",
        "read",
        "color",
        "position",
    ]

    def __init__(
        self,
        index: int = -1,
        color: Optional[MVector3D] = None,
        position: Optional[MVector3D] = None,
        register: bool = False,
        read: bool = False,
    ):
        super().__init__(index, "照明", register, read)
        self.color = color or MVector3D()
        self.position = position or MVector3D()


class VmdShadowFrame(BaseVmdNameFrame):
    """
    セルフ影キーフレ

    Parameters
    ----------
    index : int, optional
        キーフレ, by default None
    mode : int, optional
        セルフ影モード, by default None
    distance : float, optional
        影範囲距離, by default None
    register : bool, optional
        登録対象か否か, by default None
    read : bool, optional
        VMDデータから読み込んだデータか, by default None
    """

    __slots__ = [
        "index",
        "register",
        "read",
        "type",
        "distance",
    ]

    def __init__(
        self,
        index: int = -1,
        mode: int = 0,
        distance: float = 0.0,
        register: bool = False,
        read: bool = False,
    ):
        super().__init__(index, "セルフ影", register, read)
        self.type = mode
        self.distance = distance


class VmdIkOnOff(BaseModel):
    """
    IKのONOFF

    Parameters
    ----------
    name : str, optional
        IK名, by default None
    onoff : bool, optional
        ON,OFF, by default None
    """

    __slots__ = [
        "name",
        "onoff",
    ]

    def __init__(
        self,
        name: str = "",
        onoff: bool = True,
    ):
        super().__init__()
        self.name = name
        self.onoff = onoff


class VmdShowIkFrame(BaseVmdNameFrame):
    """
    IKキーフレ

    Parameters
    ----------
    index : int, optional
        キーフレ, by default None
    show : bool, optional
        表示有無, by default None
    iks : list[VmdIk], optional
        IKリスト, by default None
    register : bool, optional
        登録対象か否か, by default None
    read : bool, optional
        VMDデータから読み込んだデータか, by default None
    """

    __slots__ = [
        "index",
        "register",
        "read",
        "show",
        "iks",
    ]

    def __init__(
        self,
        index: int = -1,
        show: bool = True,
        iks: Optional[list[VmdIkOnOff]] = None,
        register: bool = False,
        read: bool = False,
    ):
        super().__init__(index, "IK", register, read)
        self.show = show
        self.iks = iks or []
