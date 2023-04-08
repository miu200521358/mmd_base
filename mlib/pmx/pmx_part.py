import os
from abc import ABC, abstractmethod
from enum import Flag, IntEnum, unique
from typing import Optional, Union

import numpy as np
import OpenGL.GL as gl
from PIL import Image, ImageOps

from mlib.base.base import BaseModel
from mlib.base.exception import MViewerException
from mlib.base.math import MMatrix4x4, MQuaternion, MVector2D, MVector3D, MVector4D
from mlib.base.part import BaseIndexModel, BaseIndexNameModel, BaseRotationModel, Switch


@unique
class DeformType(IntEnum):
    """ウェイト変形方式"""

    BDEF1 = 0
    """0:BDEF1"""
    BDEF2 = 1
    """1:BDEF2"""
    BDEF4 = 2
    """2:BDEF4"""
    SDEF = 3
    """3:SDEF"""


class Deform(BaseModel, ABC):
    """
    デフォーム基底クラス

    Parameters
    ----------
    indexes : list[int]
        ボーンINDEXリスト
    weights : list[float]
        ウェイトリスト
    count : int
        デフォームボーン個数
    """

    def __init__(self, indexes: list[int], weights: list[float], count: int):
        super().__init__()
        self.indexes = np.fromiter(indexes, dtype=np.int32, count=len(indexes))
        self.weights = np.fromiter(weights, dtype=np.float64, count=len(weights))
        self.count: int = count

    def get_indexes(self, weight_threshold: float = 0) -> np.ndarray:
        """
        デフォームボーンINDEXリスト取得

        Parameters
        ----------
        weight_threshold : float, optional
            ウェイト閾値, by default 0
            指定された場合、このweight以上のウェイトを持っているINDEXのみを取得する

        Returns
        -------
        np.ndarray
            デフォームボーンINDEXリスト
        """
        return self.indexes[self.weights >= weight_threshold]

    def get_weights(self, weight_threshold: float = 0) -> np.ndarray:
        """
        デフォームウェイトリスト取得

        Parameters
        ----------
        weight_threshold : float, optional
            ウェイト閾値, by default 0
            指定された場合、このweight以上のウェイトを持っているウェイトのみを取得する

        Returns
        -------
        np.ndarray
            デフォームウェイトリスト
        """
        return self.weights[self.weights >= weight_threshold]

    def normalize(self, align=False):
        """
        ウェイト正規化

        Parameters
        ----------
        align : bool, optional
            countのボーン数に揃えるか, by default False
        """
        if align:
            # 揃える必要がある場合
            # 数が足りるよう、かさ増しする
            ilist = np.fromiter(self.indexes.tolist() + [0, 0, 0, 0], count=(len(self.indexes) + 4), dtype=np.int32)
            wlist = np.fromiter(self.weights.tolist() + [0, 0, 0, 0], count=(len(self.weights) + 4), dtype=np.float64)
            # 正規化
            wlist /= wlist.sum(axis=0, keepdims=True)

            # ウェイトの大きい順に指定個数までを対象とする
            self.indexes = ilist[np.argsort(-wlist)][: self.count]
            self.weights = wlist[np.argsort(-wlist)][: self.count]

        # ウェイト正規化
        self.weights /= self.weights.sum(axis=0, keepdims=True)

    def normalized_deform(self) -> list:
        """
        ウェイト正規化して4つのボーンINDEXとウェイトを返す（合計8個）
        """
        # 揃える必要がある場合
        # 数が足りるよう、かさ増しする
        ilist = np.fromiter(
            np.fromiter(
                self.indexes.tolist() + [0, 0, 0, 0],
                dtype=np.int32,
                count=len(self.indexes) + 4,
            ),
            dtype=np.int32,
            count=len(self.indexes) + 4,
        )
        wlist = np.fromiter(
            np.fromiter(
                self.weights.tolist() + [0, 0, 0, 0],
                dtype=np.float64,
                count=len(self.weights) + 4,
            ),
            dtype=np.float64,
            count=len(self.weights) + 4,
        )
        # 正規化
        wlist /= wlist.sum(axis=0, keepdims=True)

        # ウェイトの大きい順に指定個数までを対象とする
        indexes = ilist[np.argsort(-wlist)][:4]
        weights = wlist[np.argsort(-wlist)][:4]

        # ウェイト正規化
        weights /= weights.sum(axis=0, keepdims=True)

        normalized_deform = []
        normalized_deform.extend(indexes.tolist())
        normalized_deform.extend(weights.tolist())

        return normalized_deform

    @abstractmethod
    def type(self) -> int:
        """
        デフォームタイプ
        """
        return -1


class Bdef1(Deform):
    __slots__ = ["indexes", "weights", "count", "index0"]

    def __init__(self, index0: int):
        super().__init__([index0], [1.0], 1)

    def type(self) -> int:
        return 0


class Bdef2(Deform):
    __slots__ = ["indexes", "weights", "count", "index0", "index1", "weight0"]

    def __init__(self, index0: int, index1: int, weight0: float):
        super().__init__([index0, index1], [weight0, 1 - weight0], 2)

    def type(self) -> int:
        return 1


class Bdef4(Deform):
    __slots__ = [
        "indexes",
        "weights",
        "count",
        "index0",
        "index1",
        "index2",
        "index3",
        "weight0",
        "weight1",
        "weight2",
        "weight3",
    ]

    def __init__(
        self,
        index0: int,
        index1: int,
        index2: int,
        index3: int,
        weight0: float,
        weight1: float,
        weight2: float,
        weight3: float,
    ):
        super().__init__([index0, index1, index2, index3], [weight0, weight1, weight2, weight3], 4)

    def type(self) -> int:
        return 2


class Sdef(Deform):
    __slots__ = [
        "indexes",
        "weights",
        "count",
        "index0",
        "index1",
        "weight0",
        "sdef_c_x",
        "sdef_c_y",
        "sdef_c_z",
        "sdef_r0_x",
        "sdef_r0_y",
        "sdef_r0_z",
        "sdef_r1_x",
        "sdef_r1_y",
        "sdef_r1_z",
    ]

    def __init__(
        self,
        index0: int,
        index1: int,
        weight0: float,
        sdef_c_x: float,
        sdef_c_y: float,
        sdef_c_z: float,
        sdef_r0_x: float,
        sdef_r0_y: float,
        sdef_r0_z: float,
        sdef_r1_x: float,
        sdef_r1_y: float,
        sdef_r1_z: float,
    ):
        super().__init__([index0, index1], [weight0, 1 - weight0], 2)
        self.sdef_c = MVector3D(sdef_c_x, sdef_c_y, sdef_c_z)
        self.sdef_r0 = MVector3D(sdef_r0_x, sdef_r0_y, sdef_r0_z)
        self.sdef_r1 = MVector3D(sdef_r1_x, sdef_r1_y, sdef_r1_z)

    def type(self) -> int:
        return 3


class Vertex(BaseIndexModel):
    """
    頂点

    Parameters
    ----------
    position : MVector3D, optional
        頂点位置, by default MVector3D()
    normal : MVector3D, optional
        頂点法線, by default MVector3D()
    uv : MVector2D, optional
        UV, by default MVector2D()
    extended_uvs : list[MVector4D], optional
        追加UV, by default []
    deform_type: DeformType, optional
        ウェイト変形方式 0:BDEF1 1:BDEF2 2:BDEF4 3:SDEF, by default DeformType.BDEF1
    deform : Deform, optional
        デフォーム, by default Deform([], [], 0)
    edge_factor : float, optional
        エッジ倍率, by default 0
    """

    __slots__ = [
        "index",
        "position",
        "normal",
        "uv",
        "extended_uvs",
        "deform_type",
        "deform",
        "edge_factor",
    ]

    def __init__(
        self,
        index: int = -1,
    ):
        super().__init__(index=index)
        self.position = MVector3D()
        self.normal = MVector3D()
        self.uv = MVector2D()
        self.extended_uvs: list[MVector4D] = []
        self.deform_type = DeformType.BDEF1
        self.deform: Union[Bdef1, Bdef2, Bdef4, Sdef] = Bdef1(-1)
        self.edge_factor = 0.0


class Face(BaseIndexModel):
    """
    面データ

    Parameters
    ----------
    vertex_index0 : int
        頂点0
    vertex_index1 : int
        頂点1
    vertex_index2 : int
        頂点2
    """

    __slots__ = [
        "index",
        "vertex_index0",
        "vertex_index1",
        "vertex_index2",
    ]

    def __init__(
        self,
        index: int = -1,
        vertex_index0: int = -1,
        vertex_index1: int = -1,
        vertex_index2: int = -1,
    ):
        super().__init__(index=index)
        self.vertices = [vertex_index0, vertex_index1, vertex_index2]


@unique
class TextureType(IntEnum):
    TEXTURE = 0
    TOON = 1
    SPHERE = 2


class Texture(BaseIndexNameModel):
    """
    テクスチャ
    """

    __slots__ = [
        "index",
        "name",
        "for_draw",
        "image",
        "texture_type",
        "texture",
        "texture_id",
    ]

    def __init__(self, index: int = -1, name: str = ""):
        super().__init__(index=index, name=name)
        self.for_draw = False
        self.valid = True

    def init_draw(self, model_path: str, texture_type: TextureType, is_individual: bool = True):
        if self.for_draw:
            # 既にフラグが立ってたら描画初期化済み
            return

        # global texture
        if is_individual:
            tex_path = os.path.abspath(os.path.join(os.path.dirname(model_path), self.name))
        else:
            tex_path = self.name

        # テクスチャがちゃんとある場合のみ初期化処理実施
        self.valid = os.path.exists(tex_path) & os.path.isfile(tex_path)
        if self.valid:
            try:
                self.image = Image.open(tex_path).convert("RGBA")
                self.image = ImageOps.flip(self.image)
            except:
                self.valid = False

            if self.valid:
                self.texture_type = texture_type

                # テクスチャオブジェクト生成
                self.texture = gl.glGenTextures(1)
                error_code = gl.glGetError()
                if error_code != gl.GL_NO_ERROR:
                    raise MViewerException(f"glGenTextures Failure\n{error_code}")

                self.texture_type = texture_type
                self.texture_id = (
                    gl.GL_TEXTURE0 if texture_type == TextureType.TEXTURE else gl.GL_TEXTURE1 if texture_type == TextureType.TOON else gl.GL_TEXTURE2
                )
                self.set_texture()

        # 描画初期化
        self.for_draw = True

    def set_texture(self):
        self.bind()
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            self.image.size[0],
            self.image.size[1],
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            self.image.tobytes(),
        )

        error_code = gl.glGetError()
        if error_code != gl.GL_NO_ERROR:
            raise MViewerException(f"Texture set_texture Failure\n{error_code}")

        self.unbind()

    def bind(self) -> None:
        gl.glActiveTexture(self.texture_id)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)

        if self.texture_type == TextureType.TOON:
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)

        error_code = gl.glGetError()
        if error_code != gl.GL_NO_ERROR:
            raise MViewerException(f"Texture bind Failure\n{error_code}")

    def unbind(self) -> None:
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        error_code = gl.glGetError()
        if error_code != gl.GL_NO_ERROR:
            raise MViewerException(f"Texture unbind Failure\n{error_code}")


@unique
class SphereMode(IntEnum):
    """スフィアモード"""

    INVALID = 0
    """0:無効"""
    MULTIPLICATION = 1
    """1:乗算(sph)"""
    ADDITION = 2
    """2:加算(spa)"""
    SUBTEXTURE = 3
    """3:サブテクスチャ(追加UV1のx,yをUV参照して通常テクスチャ描画を行う)"""


@unique
class DrawFlg(Flag):
    """描画フラグ"""

    NONE = 0x0000
    """"初期値"""
    DOUBLE_SIDED_DRAWING = 0x0001
    """0x01:両面描画"""
    GROUND_SHADOW = 0x0002
    """0x02:地面影"""
    DRAWING_ON_SELF_SHADOW_MAPS = 0x0004
    """0x04:セルフシャドウマップへの描画"""
    DRAWING_SELF_SHADOWS = 0x0008
    """0x08:セルフシャドウの描画"""
    DRAWING_EDGE = 0x0010
    """0x10:エッジ描画"""


@unique
class ToonSharing(IntEnum):
    """スフィアモード"""

    INDIVIDUAL = 0
    """0:継続値は個別Toon"""
    SHARING = 1
    """1:継続値は共有Toon"""


class Material(BaseIndexNameModel):
    """
    材質

    Parameters
    ----------
    name : str, optional
        材質名, by default ""
    english_name : str, optional
        材質名英, by default ""
    diffuse_color : MVector4D, optional
        Diffuse (R,G,B,A)(拡散色＋非透過度), by default MVector4D()
    specular_color : MVector3D, optional
        Specular (R,G,B)(反射色), by default MVector3D()
    specular_factor : float, optional
        Specular係数(反射強度), by default 0
    ambient_color : MVector3D, optional
        Ambient (R,G,B)(環境色), by default MVector3D()
    draw_flg : DrawFlg, optional
        描画フラグ(8bit) - 各bit 0:OFF 1:ON
        0x01:両面描画, 0x02:地面影, 0x04:セルフシャドウマップへの描画, 0x08:セルフシャドウの描画, 0x10:エッジ描画, by default DrawFlg.NONE
    edge_color : MVector4D, optional
        エッジ色 (R,G,B,A), by default MVector4D()
    edge_size : float, optional
        エッジサイズ, by default 0
    texture_index : int, optional
        通常テクスチャINDEX, by default -1
    sphere_texture_index : int, optional
        スフィアテクスチャINDEX, by default -1
    sphere_mode : SphereMode, optional
        スフィアモード 0:無効 1:乗算(sph) 2:加算(spa) 3:サブテクスチャ(追加UV1のx,yをUV参照して通常テクスチャ描画を行う), by default INVALID
    toon_sharing_flg : Switch, optional
        共有Toonフラグ 0:継続値は個別Toon 1:継続値は共有Toon, by default OFF
    toon_texture_index : int, optional
        ToonテクスチャINDEX, by default -1
    comment : str, optional
        メモ, by default ""
    vertices_count : int, optional
        材質に対応する面(頂点)数 (必ず3の倍数になる), by default 0
    """

    __slots__ = [
        "index",
        "name",
        "english_name",
        "diffuse",
        "specular",
        "specular_factor",
        "ambient",
        "draw_flg",
        "edge_color",
        "edge_size",
        "texture_index",
        "sphere_texture_index",
        "sphere_mode",
        "toon_sharing_flg",
        "toon_texture_index",
        "comment",
        "vertices_count",
    ]

    def __init__(
        self,
        index: int = -1,
        name: str = "",
        english_name: str = "",
    ):
        super().__init__(index=index, name=name, english_name=english_name)
        self.diffuse = MVector4D()
        self.specular = MVector3D()
        self.specular_factor = 0.0
        self.ambient = MVector3D()
        self.draw_flg = DrawFlg.NONE
        self.edge_color = MVector4D()
        self.edge_size = 0.0
        self.texture_index = -1
        self.sphere_texture_index = -1
        self.sphere_mode = SphereMode.INVALID
        self.toon_sharing_flg = ToonSharing.SHARING
        self.toon_texture_index = -1
        self.comment = ""
        self.vertices_count = 0


class IkLink(BaseModel):
    """
    IKリンク

    Parameters
    ----------
    bone_index : int, optional
        リンクボーンのボーンIndex, by default -1
    angle_limit : bool, optional
        角度制限 0:OFF 1:ON, by default False
    min_angle_limit_radians : MVector3D, optional
        下限 (x,y,z) -> ラジアン角, by default MVector3D()
    max_angle_limit_radians : MVector3D, optional
        上限 (x,y,z) -> ラジアン角, by default MVector3D()
    """

    __slots__ = [
        "bone_index",
        "angle_limit",
        "min_angle_limit_radians",
        "max_angle_limit_radians",
    ]

    def __init__(
        self,
    ):
        super().__init__()
        self.bone_index = -1
        self.angle_limit = False
        self.min_angle_limit = BaseRotationModel()
        self.max_angle_limit = BaseRotationModel()


class Ik(BaseModel):
    """
    IK

    Parameters
    ----------
    bone_index : int, optional
        IKターゲットボーンのボーンIndex, by default -1
    loop_count : int, optional
        IKループ回数 (最大255), by default 0
    unit_radians : float, optional
        IKループ計算時の1回あたりの制限角度 -> ラジアン角, by default 0
        unit_rotation の x に値が入っている
    links : list[IkLink], optional
        IKリンクリスト, by default []
    """

    __slots__ = [
        "bone_index",
        "loop_count",
        "unit_radians",
        "links",
    ]

    def __init__(
        self,
    ):
        super().__init__()
        self.bone_index = -1
        self.loop_count = 0
        self.unit_rotation = BaseRotationModel()
        self.links: list[IkLink] = []


@unique
class BoneFlg(Flag):
    """ボーンフラグ"""

    NONE = 0x0000
    """"初期値"""
    TAIL_IS_BONE = 0x0001
    """接続先(PMD子ボーン指定)表示方法 -> 0:座標オフセットで指定 1:ボーンで指定"""
    CAN_ROTATE = 0x0002
    """回転可能"""
    CAN_TRANSLATE = 0x0004
    """移動可能"""
    IS_VISIBLE = 0x0008
    """表示"""
    CAN_MANIPULATE = 0x0010
    """操作可"""
    IS_IK = 0x0020
    """IK"""
    IS_EXTERNAL_LOCAL = 0x0080
    """ローカル付与 | 付与対象 0:ユーザー変形値／IKリンク／多重付与 1:親のローカル変形量"""
    IS_EXTERNAL_ROTATION = 0x0100
    """回転付与"""
    IS_EXTERNAL_TRANSLATION = 0x0200
    """移動付与"""
    HAS_FIXED_AXIS = 0x0400
    """軸固定"""
    HAS_LOCAL_COORDINATE = 0x0800
    """ローカル軸"""
    IS_AFTER_PHYSICS_DEFORM = 0x1000
    """物理後変形"""
    IS_EXTERNAL_PARENT_DEFORM = 0x2000
    """外部親変形"""


class Bone(BaseIndexNameModel):
    """
    ボーン

    Parameters
    ----------
    name : str, optional
        ボーン名, by default ""
    english_name : str, optional
        ボーン名英, by default ""
    position : MVector3D, optional
        位置, by default MVector3D()
    parent_index : int, optional
        親ボーンのボーンIndex, by default -1
    layer : int, optional
        変形階層, by default 0
    bone_flg : BoneFlg, optional
        ボーンフラグ(16bit) 各bit 0:OFF 1:ON, by default BoneFlg.NONE
    tail_position : MVector3D, optional
        接続先:0 の場合 座標オフセット, ボーン位置からの相対分, by default MVector3D()
    tail_index : int, optional
        接続先:1 の場合 接続先ボーンのボーンIndex, by default -1
    effect_index : int, optional
        回転付与:1 または 移動付与:1 の場合 付与親ボーンのボーンIndex, by default -1
    effect_factor : float, optional
        付与率, by default 0
    fixed_axis : MVector3D, optional
        軸固定:1 の場合 軸の方向ベクトル, by default MVector3D()
    local_x_vector : MVector3D, optional
        ローカル軸:1 の場合 X軸の方向ベクトル, by default MVector3D()
    local_z_vector : MVector3D, optional
        ローカル軸:1 の場合 Z軸の方向ベクトル, by default MVector3D()
    external_key : int, optional
        外部親変形:1 の場合 Key値, by default -1
    ik : Optional[Ik], optional
        IK:1 の場合 IKデータを格納, by default None
    is_system : bool, optional
        システム計算用追加ボーン, by default False
    """

    __slots__ = [
        "index",
        "name",
        "english_name",
        "position",
        "parent_index",
        "layer",
        "bone_flg",
        "tail_position",
        "tail_index",
        "effect_index",
        "effect_factor",
        "fixed_axis",
        "local_x_vector",
        "local_z_vector",
        "external_key",
        "ik",
        "display",
        "is_system",
        "local_y_vector",
        "correct_local_z_vector",
        "local_axis",
        "ik_link_indexes",
        "ik_target_indexes",
        "offset_matrix",
        "init_matrix",
        "parent_relative_position",
        "tail_relative_position",
        "weighted_scales",
    ]

    def __init__(
        self,
        index: int = -1,
        name: str = "",
        english_name: str = "",
    ):
        super().__init__(index=index, name=name, english_name=english_name)
        self.position = MVector3D()
        self.parent_index = -1
        self.layer = 0
        self.bone_flg = BoneFlg.NONE
        self.tail_position = MVector3D()
        self.tail_index = -1
        self.effect_index = -1
        self.effect_factor = 0.0
        self.fixed_axis = MVector3D()
        self.local_x_vector = MVector3D()
        self.local_z_vector = MVector3D()
        self.external_key = -1
        self.ik: Optional[Ik] = None
        self.display: bool = False
        self.is_system: bool = False
        self.ik_link_indexes: list[int] = []
        self.ik_target_indexes: list[int] = []

        self.local_y_vector = self.local_z_vector.cross(self.local_x_vector)
        self.correct_local_z_vector = self.local_x_vector.cross(self.local_y_vector)

        self.parent_relative_position = MVector3D()
        self.tail_relative_position = MVector3D()
        self.local_axis = MVector3D()

        self.offset_matrix = MMatrix4x4()
        self.init_matrix = MMatrix4x4()
        self.weighted_scales: dict[int, MVector3D] = {}

    @property
    def is_tail_bone(self) -> bool:
        """表示先がボーンであるか"""
        return BoneFlg.TAIL_IS_BONE in self.bone_flg

    @property
    def can_rotate(self) -> bool:
        """回転可能であるか"""
        return BoneFlg.CAN_ROTATE in self.bone_flg

    @property
    def can_translate(self) -> bool:
        """移動可能であるか"""
        return BoneFlg.CAN_TRANSLATE in self.bone_flg

    @property
    def is_visible(self) -> bool:
        """表示であるか"""
        return BoneFlg.IS_VISIBLE in self.bone_flg

    @property
    def can_manipulate(self) -> bool:
        """操作可であるか"""
        return BoneFlg.CAN_MANIPULATE in self.bone_flg

    @property
    def is_ik(self) -> bool:
        """IKであるか"""
        return BoneFlg.IS_IK in self.bone_flg

    @property
    def is_external_local(self) -> bool:
        """ローカル付与であるか"""
        return BoneFlg.IS_EXTERNAL_LOCAL in self.bone_flg

    @property
    def is_external_rotation(self) -> bool:
        """回転付与であるか"""
        return BoneFlg.IS_EXTERNAL_ROTATION in self.bone_flg

    @property
    def is_external_translation(self) -> bool:
        """移動付与であるか"""
        return BoneFlg.IS_EXTERNAL_TRANSLATION in self.bone_flg

    @property
    def has_fixed_axis(self) -> bool:
        """軸固定であるか"""
        return BoneFlg.HAS_FIXED_AXIS in self.bone_flg

    @property
    def has_local_coordinate(self) -> bool:
        """ローカル軸を持つか"""
        return BoneFlg.HAS_LOCAL_COORDINATE in self.bone_flg

    @property
    def is_after_physics_deform(self) -> bool:
        """物理後変形であるか"""
        return BoneFlg.IS_AFTER_PHYSICS_DEFORM in self.bone_flg

    @property
    def is_external_parent_deform(self) -> bool:
        """外部親変形であるか"""
        return BoneFlg.IS_EXTERNAL_PARENT_DEFORM in self.bone_flg


class MorphOffset(BaseModel):
    """モーフオフセット基底クラス"""

    def __init__(self) -> None:
        super().__init__()


class VertexMorphOffset(MorphOffset):
    """
    頂点モーフ

    Parameters
    ----------
    vertex_index : int
        頂点INDEX
    position_offset : MVector3D
        座標オフセット量(x,y,z)
    """

    __slots__ = ["vertex_index", "position_offset"]

    def __init__(self, vertex_index: int, position_offset: MVector3D):
        super().__init__()
        self.vertex_index = vertex_index
        self.position_offset = position_offset


class UvMorphOffset(MorphOffset):
    """
    UVモーフ

    Parameters
    ----------
    vertex_index : int
        頂点INDEX
    uv : MVector4D
        UVオフセット量(x,y,z,w) ※通常UVはz,wが不要項目になるがモーフとしてのデータ値は記録しておく
    """

    __slots__ = ["vertex_index", "uv"]

    def __init__(self, vertex_index: int, uv: MVector4D):
        super().__init__()
        self.vertex_index = vertex_index
        self.uv = uv


class BoneMorphOffset(MorphOffset):
    """
    ボーンモーフ

    Parameters
    ----------
    bone_index : int
        ボーンIndex
    position : MVector3D
        移動量(x,y,z)
    qq : MQuaternion
        回転量-クォータニオン(x,y,z,w)
    """

    __slots__ = ["bone_index", "position", "rotation"]

    def __init__(self, bone_index: int, position: MVector3D, qq: MQuaternion):
        super().__init__()
        self.bone_index = bone_index
        self.position = position
        self.rotation = BaseRotationModel()
        self.rotation.qq = qq


class GroupMorphOffset(MorphOffset):
    """
    グループモーフ

    Parameters
    ----------
    morph_index : int
        モーフINDEX
    morph_factor : float
        モーフ変動量
    """

    __slots__ = ["morph_index", "morph_factor"]

    def __init__(self, morph_index: int, morph_factor: float):
        super().__init__()
        self.morph_index = morph_index
        self.morph_factor = morph_factor


@unique
class MaterialMorphCalcMode(IntEnum):
    """材質モーフ：計算モード"""

    MULTIPLICATION = 0
    """0:乗算"""
    ADDITION = 1
    """1:加算"""


class MaterialMorphOffset(MorphOffset):
    """
    材質モーフ

    Parameters
    ----------
    material_index : int
        材質Index -> -1:全材質対象
    calc_mode : CalcMode
        0:乗算, 1:加算
    diffuse : MVector4D
        Diffuse (R,G,B,A)
    specular : MVector3D
        Specular (R,G,B)
    specular_factor : float
        Specular係数
    ambient : MVector3D
        Ambient (R,G,B)
    edge_color : MVector4D
        エッジ色 (R,G,B,A)
    edge_size : float
        エッジサイズ
    texture_factor : MVector4D
        テクスチャ係数 (R,G,B,A)
    sphere_texture_factor : MVector4D
        スフィアテクスチャ係数 (R,G,B,A)
    toon_texture_factor : MVector4D
        Toonテクスチャ係数 (R,G,B,A)
    """

    __slots__ = [
        "material_index",
        "calc_mode",
        "diffuse",
        "specular",
        "specular_factor",
        "ambient",
        "edge_color",
        "edge_size",
        "texture_factor",
        "sphere_texture_factor",
        "toon_texture_factor",
    ]

    def __init__(
        self,
        material_index: int,
        calc_mode: MaterialMorphCalcMode,
        diffuse: MVector4D,
        specular: MVector3D,
        specular_factor: float,
        ambient: MVector3D,
        edge_color: MVector4D,
        edge_size: float,
        texture_factor: MVector4D,
        sphere_texture_factor: MVector4D,
        toon_texture_factor: MVector4D,
    ):
        super().__init__()
        self.material_index = material_index
        self.calc_mode = calc_mode
        self.diffuse = diffuse
        self.specular = specular
        self.specular_factor = specular_factor
        self.ambient = ambient
        self.edge_color = edge_color
        self.edge_size = edge_size
        self.texture_factor = texture_factor
        self.sphere_texture_factor = sphere_texture_factor
        self.toon_texture_factor = toon_texture_factor


class ShaderMaterial:
    """
    材質モーフを加味したシェーダー用材質情報
    """

    __slots__ = [
        "diffuse",
        "specular",
        "specular_factor",
        "ambient",
        "edge_color",
        "edge_size",
        "texture_factor",
        "sphere_texture_factor",
        "toon_texture_factor",
    ]

    def __init__(
        self,
        material: Material,
        light_ambient4: MVector4D,
        texture_factor: Optional[MVector4D] = None,
        toon_texture_factor: Optional[MVector4D] = None,
        sphere_texture_factor: Optional[MVector4D] = None,
    ):
        super().__init__()
        self.diffuse: MVector4D = material.diffuse * light_ambient4 + MVector4D(
            material.ambient.x,
            material.ambient.y,
            material.ambient.z,
            material.diffuse.w,
        )
        self.ambient: MVector3D = material.diffuse.xyz * light_ambient4.xyz
        self.specular: MVector4D = MVector4D(
            *(material.specular * light_ambient4.xyz).vector,
        )
        self.specular.w = material.specular_factor

        self.edge_color: MVector4D = material.edge_color
        self.edge_size = material.edge_size
        self.texture_factor: MVector4D = texture_factor or MVector4D(1, 1, 1, 1)
        self.sphere_texture_factor: MVector4D = toon_texture_factor or MVector4D(1, 1, 1, 1)
        self.toon_texture_factor: MVector4D = sphere_texture_factor or MVector4D(1, 1, 1, 1)

    def __imul__(self, v: Union[float, "ShaderMaterial"]):
        if isinstance(v, float):
            self.diffuse *= v
            self.ambient *= v
            self.specular *= v
            self.edge_color *= v
            self.edge_size *= v
            self.texture_factor *= v
            self.sphere_texture_factor *= v
            self.toon_texture_factor *= v
        else:
            self.diffuse *= v.diffuse
            self.ambient *= v.ambient
            self.specular *= v.specular
            self.edge_color *= v.edge_color
            self.edge_size *= v.edge_size
            self.texture_factor *= v.texture_factor
            self.sphere_texture_factor *= v.sphere_texture_factor
            self.toon_texture_factor *= v.toon_texture_factor
        return self

    def __iadd__(self, v: Union[float, "ShaderMaterial"]):
        if isinstance(v, float):
            self.diffuse += v
            self.ambient += v
            self.specular += v
            self.edge_color += v
            self.edge_size += v
            self.texture_factor += v
            self.sphere_texture_factor += v
            self.toon_texture_factor += v
        else:
            self.diffuse += v.diffuse
            self.ambient += v.ambient
            self.specular += v.specular
            self.edge_color += v.edge_color
            self.edge_size += v.edge_size
            self.texture_factor += v.texture_factor
            self.sphere_texture_factor += v.sphere_texture_factor
            self.toon_texture_factor += v.toon_texture_factor
        return self


@unique
class MorphPanel(IntEnum):
    """操作パネル"""

    SYSTEM = 0
    """0:システム予約"""
    EYEBROW_LOWER_LEFT = 1
    """1:眉(左下)"""
    EYE_UPPER_LEFT = 2
    """2:目(左上)"""
    LIP_UPPER_RIGHT = 3
    """3:口(右上)"""
    OTHER_LOWER_RIGHT = 4
    """4:その他(右下)"""

    @property
    def panel_name(self):
        if 1 == self.value:
            return "眉"
        elif 2 == self.value:
            return "目"
        elif 3 == self.value:
            return "口"
        elif 4 == self.value:
            return "他"
        else:
            return "システム"


@unique
class MorphType(IntEnum):
    """モーフ種類"""

    GROUP = 0
    """0:グループ"""
    VERTEX = 1
    """1:頂点"""
    BONE = 2
    """2:ボーン"""
    UV = 3
    """3:UV"""
    EXTENDED_UV1 = 4
    """4:追加UV1"""
    EXTENDED_UV2 = 5
    """5:追加UV2"""
    EXTENDED_UV3 = 6
    """6:追加UV3"""
    EXTENDED_UV4 = 7
    """7:追加UV4"""
    MATERIAL = 8
    """"8:材質"""


class Morph(BaseIndexNameModel):
    """
    _summary_

    Parameters
    ----------
    name : str, optional
        モーフ名, by default ""
    english_name : str, optional
        モーフ名英, by default ""
    panel : MorphPanel, optional
        モーフパネル, by default MorphPanel.UPPER_LEFT_EYE
    morph_type : MorphType, optional
        モーフ種類, by default MorphType.GROUP
    offsets : list[TMorphOffset], optional
        モーフオフセット
    """

    __slots__ = [
        "index",
        "name",
        "english_name",
        "panel",
        "morph_type",
        "offsets",
    ]

    def __init__(
        self,
        index: int = -1,
        name: str = "",
        english_name: str = "",
    ):
        super().__init__(index=index, name=name, english_name=english_name)
        self.panel = MorphPanel.EYE_UPPER_LEFT
        self.morph_type = MorphType.GROUP
        self.offsets: list[VertexMorphOffset | UvMorphOffset | BoneMorphOffset | GroupMorphOffset | MaterialMorphOffset] = []


@unique
class DisplayType(IntEnum):
    """表示枠要素タイプ"""

    BONE = 0
    """0:ボーン"""
    MORPH = 1
    """1:モーフ"""


class DisplaySlotReference(BaseModel):
    """
    表示枠要素

    Parameters
    ----------
    display_type : DisplayType, optional
        要素対象 0:ボーン 1:モーフ, by default DisplayType.BONE
    display_index : int, optional
        ボーンIndex or モーフIndex, by default -1
    """

    __slots__ = ["display_type", "display_index"]

    def __init__(
        self,
    ):
        super().__init__()
        self.display_type = DisplayType.BONE
        self.display_index = -1


class DisplaySlot(BaseIndexNameModel):
    """
    表示枠

    Parameters
    ----------
    name : str, optional
        枠名, by default ""
    english_name : str, optional
        枠名英, by default ""
    special_flg : Switch, optional
        特殊枠フラグ - 0:通常枠 1:特殊枠, by default Switch.OFF
    references : list[DisplaySlotReference], optional
        表示枠要素, by default []
    """

    __slots__ = ["index", "name", "english_name", "special_flg", "references"]

    def __init__(
        self,
        index: int = -1,
        name: str = "",
        english_name: str = "",
    ):
        super().__init__(index=index, name=name, english_name=english_name)
        self.special_flg = Switch.OFF
        self.references: list[DisplaySlotReference] = []


class RigidBodyParam(BaseModel):
    """
    剛体パラ

    Parameters
    ----------
    mass : float, optional
        質量, by default 0
    linear_damping : float, optional
        移動減衰, by default 0
    angular_damping : float, optional
        回転減衰, by default 0
    restitution : float, optional
        反発力, by default 0
    friction : float, optional
        摩擦力, by default 0
    """

    __slots__ = [
        "mass",
        "linear_damping",
        "angular_damping",
        "restitution",
        "friction",
    ]

    def __init__(
        self,
        mass: float = 0,
        linear_damping: float = 0,
        angular_damping: float = 0,
        restitution: float = 0,
        friction: float = 0,
    ) -> None:
        super().__init__()
        self.mass = mass
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        self.restitution = restitution
        self.friction = friction


@unique
class RigidBodyShape(IntEnum):
    """剛体の形状"""

    SPHERE = 0
    """0:球"""
    BOX = 1
    """1:箱"""
    CAPSULE = 2
    """2:カプセル"""


@unique
class RigidBodyMode(IntEnum):
    """剛体物理の計算モード"""

    STATIC = 0
    """0:ボーン追従(static)"""
    DYNAMIC = 1
    """1:物理演算(dynamic)"""
    DYNAMIC_BONE = 2
    """2:物理演算 + Bone位置合わせ"""


@unique
class RigidBodyCollisionGroup(Flag):
    """剛体の衝突グループ"""

    NONE = 0x0000
    """0:グループなし"""
    GROUP01 = 0x0001
    GROUP02 = 0x0002
    GROUP03 = 0x0004
    GROUP04 = 0x0008
    GROUP05 = 0x0010
    GROUP06 = 0x0020
    GROUP07 = 0x0040
    GROUP08 = 0x0080
    GROUP09 = 0x0100
    GROUP10 = 0x0200
    GROUP11 = 0x0400
    GROUP12 = 0x0800
    GROUP13 = 0x1000
    GROUP14 = 0x2000
    GROUP15 = 0x4000
    GROUP16 = 0x8000


class RigidBody(BaseIndexNameModel):
    """
    剛体

    Parameters
    ----------
    name : str, optional
        剛体名, by default ""
    english_name : str, optional
        剛体名英, by default ""
    bone_index : int, optional
        関連ボーンIndex, by default -1
    collision_group : int, optional
        グループ, by default 0
    no_collision_group : RigidBodyCollisionGroup, optional
        非衝突グループフラグ, by default 0
    shape_type : RigidBodyShape, optional
        形状, by default RigidBodyShape.SPHERE
    shape_size : MVector3D, optional
        サイズ(x,y,z), by default MVector3D()
    shape_position : MVector3D, optional
        位置(x,y,z), by default MVector3D()
    shape_rotation_radians : MVector3D, optional
        回転(x,y,z) -> ラジアン角, by default MVector3D()
    mass : float, optional
        質量, by default 0
    linear_damping : float, optional
        移動減衰, by default 0
    angular_damping : float, optional
        回転減衰, by default 0
    restitution : float, optional
        反発力, by default 0
    friction : float, optional
        摩擦力, by default 0
    mode : RigidBodyMode, optional
        剛体の物理演算, by default RigidBodyMode.STATIC
    """

    __slots__ = [
        "index",
        "name",
        "english_name",
        "bone_index",
        "collision_group",
        "no_collision_group",
        "shape_type",
        "shape_size",
        "shape_position",
        "shape_rotation",
        "param",
        "mode",
        "x_direction",
        "y_direction",
        "z_direction",
    ]

    def __init__(
        self,
        index: int = -1,
        name: str = "",
        english_name: str = "",
    ) -> None:
        super().__init__(index=index, name=name, english_name=english_name)
        self.bone_index = -1
        self.collision_group = 0
        self.no_collision_group = RigidBodyCollisionGroup.NONE
        self.shape_type = RigidBodyShape.SPHERE
        self.shape_size = MVector3D()
        self.shape_position = MVector3D()
        self.shape_rotation = BaseRotationModel()
        self.param = RigidBodyParam()
        self.mode = RigidBodyMode.STATIC
        # 軸方向
        self.x_direction = MVector3D()
        self.y_direction = MVector3D()
        self.z_direction = MVector3D()


class JointLimitParam(BaseModel):
    """
    ジョイント制限パラメーター

    Parameters
    ----------
    limit_min : MVector3D, optional
        制限最小角度, by default MVector3D()
    limit_max : MVector3D, optional
        制限最大角度, by default MVector3D()
    """

    __slots__ = ["limit_min", "limit_max"]

    def __init__(
        self,
        limit_min: MVector3D,
        limit_max: MVector3D,
    ) -> None:
        super().__init__()
        self.limit_min = limit_min or MVector3D()
        self.limit_max = limit_max or MVector3D()


class JointParam(BaseModel):
    """
    ジョイントパラメーター

    Parameters
    ----------
    translation_limit_min : MVector3D, optional
        移動制限-下限(x,y,z), by default MVector3D()
    translation_limit_max : MVector3D, optional
        移動制限-上限(x,y,z), by default MVector3D()
    rotation_limit_min : BaseRotationModel, optional
        回転制限-下限(x,y,z) -> ラジアン角, by default BaseRotationModel()
    rotation_limit_max : BaseRotationModel, optional
        回転制限-上限(x,y,z) -> ラジアン角, by default BaseRotationModel()
    spring_constant_translation : MVector3D, optional
        バネ定数-移動(x,y,z), by default MVector3D()
    spring_constant_rotation : MVector3D, optional
        バネ定数-回転(x,y,z), by default MVector3D()
    """

    __slots__ = [
        "translation_limit_min",
        "translation_limit_max",
        "rotation_limit_min_radians",
        "rotation_limit_max_radians",
        "spring_constant_translation",
        "spring_constant_rotation",
    ]

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.translation_limit_min = MVector3D()
        self.translation_limit_max = MVector3D()
        self.rotation_limit_min = BaseRotationModel()
        self.rotation_limit_max = BaseRotationModel()
        self.spring_constant_translation = MVector3D()
        self.spring_constant_rotation = MVector3D()


class Joint(BaseIndexNameModel):
    """
    ジョイント

    Parameters
    ----------
    name : str, optional
        Joint名, by default ""
    english_name : str, optional
        Joint名英, by default ""
    joint_type : int, optional
        Joint種類, by default 0
    rigidbody_index_a : int, optional
        関連剛体AのIndex, by default -1
    rigidbody_index_b : int, optional
        関連剛体BのIndex, by default -1
    position : MVector3D, optional
        位置(x,y,z), by default MVector3D()
    rotation : BaseRotationModel, optional
        回転(x,y,z) -> ラジアン角, by default BaseRotationModel()
    translation_limit_min : MVector3D, optional
        移動制限-下限(x,y,z), by default MVector3D()
    translation_limit_max : MVector3D, optional
        移動制限-上限(x,y,z), by default MVector3D()
    rotation_limit_min : BaseRotationModel, optional
        回転制限-下限(x,y,z) -> ラジアン角, by default BaseRotationModel()
    rotation_limit_max : BaseRotationModel, optional
        回転制限-上限(x,y,z) -> ラジアン角, by default BaseRotationModel()
    spring_constant_translation : MVector3D, optional
        バネ定数-移動(x,y,z), by default MVector3D()
    spring_constant_rotation : MVector3D, optional
        バネ定数-回転(x,y,z), by default MVector3D()
    """

    __slots__ = [
        "index",
        "name",
        "english_name",
        "joint_type",
        "rigidbody_index_a",
        "rigidbody_index_b",
        "position",
        "rotation",
        "param",
    ]

    def __init__(
        self,
        index: int = -1,
        name: str = "",
        english_name: str = "",
    ) -> None:
        super().__init__(index=index, name=name, english_name=english_name)
        self.joint_type = 0
        self.rigidbody_index_a = -1
        self.rigidbody_index_b = -1
        self.position = MVector3D()
        self.rotation = BaseRotationModel()
        self.param = JointParam()
