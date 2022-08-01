from abc import ABC, abstractmethod
from enum import Enum, Flag, IntEnum, unique
from typing import Optional

import numpy as np
from mlib.math import MQuaternion, MVector2D, MVector3D, MVector4D
from mlib.model.base import (
    BaseHashModel,
    BaseIndexListModel,
    BaseIndexModel,
    BaseIndexNameListModel,
    BaseIndexNameModel,
    BaseModel,
    BaseRotationModel,
    IntPair,
    Switch,
)


class Deform(BaseModel, ABC):
    """デフォーム基底クラス"""

    def __init__(self, indecies: list[int], weights: list[float], count: int):
        """
        デフォーム基底クラス初期化

        Parameters
        ----------
        indecies : list[int]
            ボーンINDEXリスト
        weights : list[float]
            ウェイトリスト
        count : int
            デフォームボーン個数
        """
        super().__init__()
        self.indecies = np.array(indecies, dtype=np.int32)
        self.weights = np.array(weights, dtype=np.float64)
        self.count: int = count

    def get_indecies(self, weight_threshold: float = 0) -> np.ndarray:
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
        return self.indecies[self.weights >= weight_threshold]

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
            ilist = np.array(self.indices.tolist() + [0, 0, 0, 0])
            wlist = np.array(self.weights.tolist() + [0, 0, 0, 0])
            # 正規化
            wlist /= wlist.sum(axis=0, keepdims=1)

            # ウェイトの大きい順に指定個数までを対象とする
            self.indecies = ilist[np.argsort(-wlist)][: self.count]
            self.weights = wlist[np.argsort(-wlist)][: self.count]

        # ウェイト正規化
        self.weights /= self.weights.sum(axis=0, keepdims=1)

    @abstractmethod
    def type(self) -> int:
        """
        デフォームタイプ
        """
        return -1


class Bdef1(Deform):
    def __init__(self, index0: int):
        super().__init__([index0], [1.0], 1)

    def type(self) -> int:
        return 0


class Bdef2(Deform):
    def __init__(self, index0: int, index1: int, weight0: float):
        super().__init__([index0, index1], [weight0, 1 - weight0], 2)

    def type(self) -> int:
        return 1


class Bdef4(Deform):
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
        super().__init__(
            [index0, index1, index2, index3], [weight0, weight1, weight2, weight3], 4
        )

    def type(self) -> int:
        return 2


class Sdef(Deform):
    def __init__(
        self,
        index0: int,
        index1: int,
        weight0: float,
        sdef_c: float,
        sdef_r0: float,
        sdef_r1: float,
    ):
        super().__init__([index0, index1], [weight0, 1 - weight0], 2)
        self.sdef_c = sdef_c
        self.sdef_r0 = sdef_r0
        self.sdef_r1 = sdef_r1

    def type(self) -> int:
        return 3


class Vertex(BaseIndexModel):
    def __init__(
        self,
        position: MVector3D = MVector3D(),
        normal: MVector3D = MVector3D(),
        uv: MVector2D = MVector2D(),
        extended_uvs: list[MVector4D] = [],
        deform: Deform = Bdef1(-1),
        edge_factor: float = 0,
    ):
        """
        頂点初期化

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
        deform : Deform, optional
            デフォーム, by default Deform([], [], 0)
        edge_factor : float, optional
            エッジ倍率, by default 0
        """
        super().__init__()
        self.position = position
        self.normal = normal
        self.uv = uv
        self.extended_uvs = extended_uvs
        self.deform = deform
        self.edge_factor = edge_factor


class Vertices(BaseIndexListModel):
    def __init__(self, list_: list[Vertex] = []):
        """
        頂点リスト

        Parameters
        ----------
        list_ : list[Vertex], optional
            リスト, by default []
        """
        super().__init__(list_)


class Surface(BaseIndexModel):
    def __init__(self, vertex_index0: int, vertex_index1: int, vertex_index2: int):
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
        super().__init__()
        self.vertices = [vertex_index0, vertex_index1, vertex_index2]


class Surfaces(BaseIndexListModel):
    def __init__(self, list_: list[Surface] = []):
        """
        面リスト

        Parameters
        ----------
        list_ : list[Surface], optional
            リスト, by default []
        """
        super().__init__(list_)


class Texture(BaseIndexModel):
    def __init__(self, texture_path: str):
        """
        テクスチャ

        Parameters
        ----------
        texture_path : str
            テクスチャパス
        """
        super().__init__()
        self.texture_path = texture_path


class Textures(BaseIndexListModel):
    def __init__(self, list_: list[Texture] = []):
        """
        テクスチャリスト

        Parameters
        ----------
        list_ : list[Texture], optional
            リスト, by default []
        """
        super().__init__(list_)


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


class Material(BaseIndexNameModel):
    def __init__(
        self,
        name: str = "",
        english_name: str = "",
        diffuse_color: MVector4D = MVector4D(),
        specular_color: MVector3D = MVector3D(),
        specular_factor: float = 0,
        ambient_color: MVector3D = MVector3D(),
        draw_flg: DrawFlg = DrawFlg.NONE,
        edge_color: MVector4D = MVector4D(),
        edge_size: float = 0,
        texture_index: int = -1,
        sphere_texture_index: int = -1,
        sphere_mode: SphereMode = SphereMode.INVALID,
        toon_sharing_flg: Switch = Switch.OFF,
        toon_texture_index: int = -1,
        comment="",
        vertices_count=0,
    ):
        """
        材質初期化

        Parameters
        ----------
        name : str, optional
            材質名, by default ""
        english_name : str, optional
            材質名英, by default ""
        diffuse_color : MVector4D, optional
            Diffuse (R,G,B,A), by default MVector4D()
        specular_color : MVector3D, optional
            Specular (R,G,B), by default MVector3D()
        specular_factor : float, optional
            Specular係数, by default 0
        ambient_color : MVector3D, optional
            Ambient (R,G,B), by default MVector3D()
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
        super().__init__(name=name, english_name=english_name)
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.specular_factor = specular_factor
        self.ambient_color = ambient_color
        self.draw_flg = draw_flg
        self.edge_color = edge_color
        self.edge_size = edge_size
        self.texture_index = texture_index
        self.sphere_texture_index = sphere_texture_index
        self.sphere_mode = sphere_mode
        self.toon_sharing_flg = toon_sharing_flg
        self.toon_texture_index = toon_texture_index
        self.comment = comment
        self.vertices_count = vertices_count


class Materials(BaseIndexNameListModel):
    def __init__(self, list_: list[Material] = []):
        """
        材質リスト

        Parameters
        ----------
        list_ : list[Material], optional
            リスト, by default []
        """
        super().__init__(list_)


class IkLink(BaseModel):
    def __init__(
        self,
        bone_index: int = -1,
        angle_limit: bool = False,
        min_angle_limit_radians: MVector3D = MVector3D(),
        max_angle_limit_radians: MVector3D = MVector3D(),
    ):
        """
        IKリンク初期化

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
        super().__init__()
        self.bone_index = bone_index
        self.angle_limit = angle_limit
        self.min_angle_limit: BaseRotationModel = BaseRotationModel(
            min_angle_limit_radians
        )
        self.max_angle_limit: BaseRotationModel = BaseRotationModel(
            max_angle_limit_radians
        )


class Ik(BaseModel):
    def __init__(
        self,
        bone_index: int = -1,
        loop_count: int = 0,
        limit_radians: float = 0,
        links: list[IkLink] = [],
    ):
        """
        IK初期化

        Parameters
        ----------
        bone_index : int, optional
            IKターゲットボーンのボーンIndex, by default -1
        loop_count : int, optional
            IKループ回数 (最大255), by default 0
        limit_radians : float, optional
            IKループ計算時の1回あたりの制限角度 -> ラジアン角, by default 0
        links : list[IkLink], optional
            IKリンクリスト, by default []
        """
        super().__init__()
        self.bone_index = bone_index
        self.loop_count = loop_count
        self.limit_radians = limit_radians
        self.links = links


@unique
class BoneFlag(Flag):
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
    def __init__(
        self,
        name: str = "",
        english_name: str = "",
        position: MVector3D = MVector3D(),
        parent_index: int = -1,
        layer: int = 0,
        bone_flg: BoneFlag = BoneFlag.NONE,
        tail_position: MVector3D = MVector3D(),
        tail_index: int = -1,
        effect_index: int = -1,
        effect_factor: float = 0,
        fixed_axis: MVector3D = MVector3D(),
        local_x_vector: MVector3D = MVector3D(),
        local_z_vector: MVector3D = MVector3D(),
        external_key: int = -1,
        ik: Optional[Ik] = None,
        display: bool = False,
        is_sizing: bool = False,
    ):
        """
        ボーン初期化

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
        bone_flg : BoneFlag, optional
            ボーンフラグ(16bit) 各bit 0:OFF 1:ON, by default BoneFlag.NONE
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
        is_sizing : bool, optional
            サイジング計算用追加ボーン, by default False
        """
        super().__init__(name=name, english_name=english_name)
        self.position = position
        self.parent_index = parent_index
        self.layer = layer
        self.bone_flg = bone_flg
        self.tail_position = tail_position
        self.tail_index = tail_index
        self.effect_index = effect_index
        self.effect_factor = effect_factor
        self.fixed_axis = fixed_axis
        self.local_x_vector = local_x_vector
        self.local_z_vector = local_z_vector
        self.external_key = external_key
        self.ik = ik
        self.display = display
        self.is_sizing = is_sizing


class Bones(BaseIndexNameListModel):
    def __init__(self, list_: list[Bone] = []):
        """
        ボーンリスト

        Parameters
        ----------
        list_ : list[Bone], optional
            リスト, by default []
        """
        super().__init__(list_)


class MorphOffset(BaseModel):
    def __init__(self):
        super().__init__()


class VertexMorphOffset(MorphOffset):
    def __init__(self, vertex_index: int, position_offset: MVector3D):
        """
        頂点モーフ

        Parameters
        ----------
        vertex_index : int
            頂点INDEX
        position_offset : MVector3D
            座標オフセット量(x,y,z)
        """
        self.vertex_index = vertex_index
        self.position_offset = position_offset


class UvMorphOffset(MorphOffset):
    def __init__(self, vertex_index: int, uv: MVector4D):
        """
        UVモーフ

        Parameters
        ----------
        vertex_index : int
            頂点INDEX
        uv : MVector4D
            UVオフセット量(x,y,z,w) ※通常UVはz,wが不要項目になるがモーフとしてのデータ値は記録しておく
        """
        self.vertex_index = vertex_index
        self.uv = uv


class BoneMorphOffset(MorphOffset):
    def __init__(self, bone_index: int, position: MVector3D, rotation: MQuaternion):
        """
        ボーンモーフ

        Parameters
        ----------
        bone_index : int
            ボーンIndex
        position : MVector3D
            移動量(x,y,z)
        rotation : MQuaternion
            回転量-クォータニオン(x,y,z,w)
        """
        self.bone_index = bone_index
        self.position = position
        self.rotation = rotation


@unique
class CalcMode(IntEnum):
    MULTIPLICATION = 0
    """0:乗算"""
    ADDITION = 1
    """1:加算"""


class MaterialMorphOffset(MorphOffset):
    def __init__(
        self,
        material_index: int,
        calc_mode: CalcMode,
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


@unique
class MorphPanel(Enum):
    """操作パネル"""

    SYSTEM = IntPair("システム", 0)
    """0:システム予約"""
    LOWER_LEFT_EYEBROW = IntPair("眉", 1)
    """1:眉(左下)"""
    UPPER_LEFT_EYE = IntPair("目", 2)
    """2:目(左上)"""
    UPPER_RIGHT_LIP = IntPair("口", 3)
    """3:口(右上)"""
    LOWER_RIGHT_OTHER = IntPair("他", 4)
    """4:その他(右下)"""


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
    def __init__(
        self,
        name: str = "",
        english_name: str = "",
        panel: MorphPanel = MorphPanel.UPPER_LEFT_EYE,
        morph_type: MorphType = MorphType.GROUP,
        offsets: list[MorphOffset] = [],
    ):
        super().__init__(name=name, english_name=english_name)
        self.panel = panel
        self.morph_type = morph_type
        self.offsets = offsets


class Morphs(BaseIndexNameListModel):
    def __init__(self, list_: list[Morph] = []):
        """
        モーフリスト

        Parameters
        ----------
        list_ : list[Morph], optional
            リスト, by default []
        """
        super().__init__(list_)


class DisplayType(IntEnum):
    BONE = 0
    """0:ボーン"""
    MORPH = 1
    """1:モーフ"""


class DisplaySlotReference(BaseModel):
    def __init__(
        self, display_type: DisplayType = DisplayType.BONE, display_index: int = -1
    ):
        """
        表示枠要素

        Parameters
        ----------
        display_type : DisplayType, optional
            要素対象 0:ボーン 1:モーフ, by default DisplayType.BONE
        display_index : int, optional
            ボーンIndex or モーフIndex, by default -1
        """
        super().__init__()
        self.display_type = display_type
        self.display_index = display_index


class DisplaySlot(BaseIndexNameModel):
    def __init__(
        self,
        name: str = "",
        english_name: str = "",
        special_flag: Switch = Switch.OFF,
        references: list[DisplaySlotReference] = [],
    ):
        """
        表示枠初期化

        Parameters
        ----------
        name : str, optional
            枠名, by default ""
        english_name : str, optional
            枠名英, by default ""
        special_flag : Switch, optional
            特殊枠フラグ - 0:通常枠 1:特殊枠, by default Switch.OFF
        references : list[DisplaySlotReference], optional
            表示枠要素, by default []
        """
        super().__init__(name=name, english_name=english_name)
        self.special_flag = special_flag
        self.references = references


class DisplaySlots(BaseIndexNameListModel):
    def __init__(
        self,
        list_: list[DisplaySlot] = [],
    ):
        """
        表示枠リスト

        Parameters
        ----------
        list_ : list[DisplaySlot], optional
            リスト
        """
        super().__init__(list_)
        # システム表示枠は必ず必要なので、初期化時に追加
        self.append(
            DisplaySlot(
                "Root", "Root", Switch.ON, [DisplaySlotReference(DisplayType.BONE, 0)]
            )
        )
        self.append(DisplaySlot("表情", "Exp", Switch.ON, []))


class RigidBodyParam(BaseModel):
    def __init__(
        self,
        mass: float = 0,
        linear_damping: float = 0,
        angular_damping: float = 0,
        restitution: float = 0,
        friction: float = 0,
    ) -> None:
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
        super().__init__()
        self.mass = mass
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        self.restitution = restitution
        self.friction = friction


class RigidBodyShape(IntEnum):
    SPHERE = 0
    """0:球"""
    BOX = 1
    """1:箱"""
    CAPSULE = 2
    """2:カプセル"""


class RigidBodyMode(IntEnum):
    STATIC = 0
    """0:ボーン追従(static)"""
    DYNAMIC = 1
    """1:物理演算(dynamic)"""
    DYNAMIC_BONE = 2
    """2:物理演算 + Bone位置合わせ"""


class RigidBody(BaseIndexNameModel):
    def __init__(
        self,
        name: str = "",
        english_name: str = "",
        bone_index: int = -1,
        collision_group: int = 0,
        no_collision_group: int = 0,
        shape_type: RigidBodyShape = RigidBodyShape.SPHERE,
        shape_size: MVector3D = MVector3D(),
        shape_position: MVector3D = MVector3D(),
        shape_rotation_radians: MVector3D = MVector3D(),
        mass: float = 0,
        linear_damping: float = 0,
        angular_damping: float = 0,
        restitution: float = 0,
        friction: float = 0,
        mode: RigidBodyMode = RigidBodyMode.STATIC,
    ) -> None:
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
        no_collision_group : int, optional
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
        super().__init__(name=name, english_name=english_name)
        self.bone_index = bone_index
        self.collision_group = collision_group
        self.no_collision_group = no_collision_group
        self.shape_type = shape_type
        self.shape_size = shape_size
        self.shape_position = shape_position
        self.shape_rotation: BaseRotationModel = BaseRotationModel(
            shape_rotation_radians
        )
        self.param = RigidBodyParam(
            mass, linear_damping, angular_damping, restitution, friction
        )
        self.mode = mode
        # 軸方向
        self.x_direction = MVector3D()
        self.y_direction = MVector3D()
        self.z_direction = MVector3D()


class RigidBodies(BaseIndexNameListModel):
    def __init__(self, list_: list[RigidBody] = []):
        """
        剛体リスト

        Parameters
        ----------
        list_ : list[RigidBody], optional
            リスト, by default []
        """
        super().__init__(list_)


class JointLimitParam(BaseModel):
    def __init__(
        self,
        limit_min: MVector3D = MVector3D(),
        limit_max: MVector3D = MVector3D(),
    ) -> None:
        """
        ジョイント制限パラメーター

        Parameters
        ----------
        limit_min : MVector3D, optional
            制限最小角度, by default MVector3D()
        limit_max : MVector3D, optional
            制限最大角度, by default MVector3D()
        """
        super().__init__()
        self.limit_min = limit_min
        self.limit_max = limit_max


class JointParam(BaseModel):
    def __init__(
        self,
        translation_limit_min: MVector3D = MVector3D(),
        translation_limit_max: MVector3D = MVector3D(),
        rotation_limit_min: BaseRotationModel = BaseRotationModel(),
        rotation_limit_max: BaseRotationModel = BaseRotationModel(),
        spring_constant_translation: MVector3D = MVector3D(),
        spring_constant_rotation: MVector3D = MVector3D(),
    ) -> None:
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
        super().__init__()
        self.translation_limit_min = translation_limit_min
        self.translation_limit_max = translation_limit_max
        self.rotation_limit_min = rotation_limit_min
        self.rotation_limit_max = rotation_limit_max
        self.spring_constant_translation = spring_constant_translation
        self.spring_constant_rotation = spring_constant_rotation


class Joint(BaseIndexNameModel):
    def __init__(
        self,
        name: str = "",
        english_name: str = "",
        joint_type: int = 0,
        rigidbody_index_a: int = -1,
        rigidbody_index_b: int = -1,
        position: MVector3D = MVector3D(),
        rotation: BaseRotationModel = BaseRotationModel(),
        translation_limit_min: MVector3D = MVector3D(),
        translation_limit_max: MVector3D = MVector3D(),
        rotation_limit_min: BaseRotationModel = BaseRotationModel(),
        rotation_limit_max: BaseRotationModel = BaseRotationModel(),
        spring_constant_translation: MVector3D = MVector3D(),
        spring_constant_rotation: MVector3D = MVector3D(),
    ) -> None:
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
        super().__init__(name=name, english_name=english_name)
        self.joint_type = joint_type
        self.rigidbody_index_a = rigidbody_index_a
        self.rigidbody_index_b = rigidbody_index_b
        self.position = position
        self.rotation = rotation
        self.param = JointParam(
            translation_limit_min,
            translation_limit_max,
            rotation_limit_min,
            rotation_limit_max,
            spring_constant_translation,
            spring_constant_rotation,
        )


class Joints(BaseIndexNameListModel):
    def __init__(self, list_: list[Joint] = []):
        """
        ジョイントリスト

        Parameters
        ----------
        list_ : list[Joint], optional
            リスト, by default []
        """
        super().__init__(list_)


class PmxModel(BaseHashModel):
    def __init__(
        self,
        path: str = "",
        name: str = "",
        english_name: str = "",
        comment: str = "",
        english_comment: str = "",
        json_data: dict = {},
    ):
        """
        Pmxモデルデータ

        Parameters
        ----------
        path : str, optional
            パス, by default ""
        name : str, optional
            モデル名, by default ""
        english_name : str, optional
            モデル名英, by default ""
        comment : str, optional
            コメント, by default ""
        english_comment : str, optional
            コメント英, by default ""
        json_data : dict, optional
            JSONデータ（vroidデータ用）, by default {}
        """
        super().__init__(path)
        self.name = name
        self.english_name = english_name
        self.comment = comment
        self.english_comment = english_comment
        self.json_data = json_data
        self.vertices = Vertices()
        self.surfaces = Surfaces()
        self.textures = Textures()
        self.materials = Materials()
        self.bones = Bones()
        self.morphs = Morphs()
        self.display_slots = DisplaySlots()
        self.rigid_bodies = RigidBodies()
        self.joints = Joints()
        # ハッシュ値は初期化時に持っておく
        self.digest = self.hexdigest()
