from typing import Optional

from mlib.base.base import BaseModel
from mlib.base.bezier import Interpolation
from mlib.base.math import MQuaternion, MVector3D
from mlib.base.part import (BaseIndexModel, BaseIndexNameModel,
                            BaseRotationModel)


class BaseVmdFrame(BaseIndexModel):
    """
    VMD用基底クラス

    Parameters
    ----------
    index : int, optional
        キーフレ, by default None
    registerer : bool, optional
        登録対象か否か, by default None
    read : bool, optional
        VMDデータから読み込んだデータか, by default None
    """

    def __init__(
        self,
        index: int = -1,
        registerer: bool = False,
        read: bool = False,
    ):
        super().__init__(index)
        self.registerer = registerer
        self.read = read


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

    def __init__(
        self,
        translation_x: Interpolation = None,
        translation_y: Interpolation = None,
        translation_z: Interpolation = None,
        rotation: Interpolation = None,
        vals: list[int] = None,
    ):
        self.translation_x: Interpolation = translation_x or Interpolation()
        self.translation_y: Interpolation = translation_y or Interpolation()
        self.translation_z: Interpolation = translation_z or Interpolation()
        self.rotation: Interpolation = rotation or Interpolation()
        self.vals = vals or [
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

    def merge(self) -> list[int]:
        return [
            self.translation_x.start.x,
            self.translation_y.start.x,
            self.translation_z.start.x,
            self.rotation.start.x,
            self.translation_x.start.y,
            self.translation_y.start.y,
            self.translation_z.start.y,
            self.rotation.start.y,
            self.translation_x.end.x,
            self.translation_y.end.x,
            self.translation_z.end.x,
            self.rotation.end.x,
            self.translation_x.end.y,
            self.translation_y.end.y,
            self.translation_z.end.y,
            self.rotation.end.y,
            self.translation_y.start.x,
            self.translation_z.start.x,
            self.rotation.start.x,
            self.translation_x.start.y,
            self.translation_y.start.y,
            self.translation_z.start.y,
            self.rotation.start.y,
            self.translation_x.end.x,
            self.translation_y.end.x,
            self.translation_z.end.x,
            self.rotation.end.x,
            self.translation_x.end.y,
            self.translation_y.end.y,
            self.translation_z.end.y,
            self.rotation.end.y,
            self.vals[31],
            self.translation_z.start.x,
            self.rotation.start.x,
            self.translation_x.start.y,
            self.translation_y.start.y,
            self.translation_z.start.y,
            self.rotation.start.y,
            self.translation_x.end.x,
            self.translation_y.end.x,
            self.translation_z.end.x,
            self.rotation.end.x,
            self.translation_x.end.y,
            self.translation_y.end.y,
            self.translation_z.end.y,
            self.rotation.end.y,
            self.vals[46],
            self.vals[47],
            self.rotation.start.x,
            self.translation_x.start.y,
            self.translation_y.start.y,
            self.translation_z.start.y,
            self.rotation.start.y,
            self.translation_x.end.x,
            self.translation_y.end.x,
            self.translation_z.end.x,
            self.rotation.end.x,
            self.translation_x.end.y,
            self.translation_y.end.y,
            self.translation_z.end.y,
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

    def __init__(
        self,
        name: str = "",
        index: int = -1,
        position: MVector3D = None,
        rotation: MQuaternion = None,
        interpolations: BoneInterpolations = None,
        register: bool = False,
        read: bool = False,
    ):
        super().__init__(index, name, register, read)
        self.position = position or MVector3D()
        self.rotation = rotation or MQuaternion()
        self.interpolations = interpolations or BoneInterpolations()
        self.ik_rotation: Optional[MQuaternion] = None
        self.ik_target_rotation: Optional[MQuaternion] = None
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

    def __init__(
        self,
        translation_x: Interpolation = None,
        translation_y: Interpolation = None,
        translation_z: Interpolation = None,
        rotation: Interpolation = None,
        distance: Interpolation = None,
        viewing_angle: Interpolation = None,
    ):
        self.translation_x = translation_x or Interpolation()
        self.translation_y = translation_y or Interpolation()
        self.translation_z = translation_z or Interpolation()
        self.rotation = rotation or Interpolation()
        self.distance = distance or Interpolation()
        self.viewing_angle = viewing_angle or Interpolation()


class VmdCameraFrame(BaseVmdFrame):
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

    def __init__(
        self,
        index: int = -1,
        position: MVector3D = None,
        rotation: BaseRotationModel = None,
        distance: float = 0.0,
        viewing_angle: int = 0,
        perspective: bool = False,
        interpolations: CameraInterpolations = None,
        register: bool = False,
        read: bool = False,
    ):
        super().__init__(index, register, read)
        self.position = position or MVector3D()
        self.rotation = rotation or BaseRotationModel()
        self.distance = distance
        self.viewing_angle = viewing_angle
        self.perspective = perspective
        self.interpolations = interpolations or CameraInterpolations()


class VmdLightFrame(BaseVmdFrame):
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

    def __init__(
        self,
        index: int = -1,
        color: MVector3D = None,
        position: MVector3D = None,
        register: bool = False,
        read: bool = False,
    ):
        super().__init__(index, register, read)
        self.color = color or MVector3D()
        self.position = position or MVector3D()


class VmdShadowFrame(BaseVmdFrame):
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

    def __init__(
        self,
        index: int = -1,
        mode: int = 0,
        distance: float = 0.0,
        register: bool = False,
        read: bool = False,
    ):
        super().__init__(index, register, read)
        self.type = mode
        self.distance = distance


class VmdIkOnoff(BaseModel):
    """
    IKのONOFF

    Parameters
    ----------
    name : str, optional
        IK名, by default None
    onoff : bool, optional
        ONOFF, by default None
    """

    def __init__(
        self,
        name: str = "",
        onoff: bool = True,
    ):
        super().__init__()
        self.name = name
        self.onoff = onoff


class VmdShowIkFrame(BaseVmdFrame):
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

    def __init__(
        self,
        index: int = -1,
        show: bool = True,
        iks: list[VmdIkOnoff] = None,
        register: bool = False,
        read: bool = False,
    ):
        super().__init__(index, register, read)
        self.show = show
        self.iks = iks or []
