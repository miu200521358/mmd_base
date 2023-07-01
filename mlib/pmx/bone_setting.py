from enum import Enum, Flag, unique
from typing import Iterable, Union

from mlib.base.math import MVector3D


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


class BoneSetting:
    """ボーン設定"""

    def __init__(
        self,
        name: str,
        parents: Iterable[str],
        relatives: Union[MVector3D, Iterable[str]],
        tails: Iterable[str],
        flag: BoneFlg,
        axis: MVector3D,
    ) -> None:
        """
        name : 準標準ボーン名
        parents : 親ボーン名候補リスト
        relative : 軸計算時の相対位置
            vector の場合はそのまま使う。名前リストの場合、該当ボーンの位置との相対位置
        tails : 末端ボーン名候補リスト
        flag : ボーンの特性
        axis : ボーンの仮想軸
        """
        self.name = name
        self.parents = parents
        self.relatives = relatives
        self.tails = tails
        self.flag = flag
        self.axis = axis


class BoneSettings(Enum):
    """準標準ボーン設定一覧"""

    ROOT = BoneSetting(
        name="全ての親",
        parents=[],
        relatives=MVector3D(0, 1, 0),
        tails=("センター",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_TRANSLATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE,
        axis=MVector3D(0, 1, 0),
    )
    CENTER = BoneSetting(
        name="センター",
        parents=("全ての親",),
        relatives=MVector3D(0, 1, 0),
        tails=("上半身", "下半身"),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_TRANSLATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE,
        axis=MVector3D(0, 1, 0),
    )
    GROOVE = BoneSetting(
        name="グルーブ",
        parents=("センター",),
        relatives=MVector3D(0, -1, 0),
        tails=("上半身", "下半身"),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_TRANSLATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE,
        axis=MVector3D(0, -1, 0),
    )
    WAIST = BoneSetting(
        name="腰",
        parents=("グルーブ", "センター"),
        relatives=MVector3D(0, -1, 0),
        tails=("上半身", "下半身"),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_TRANSLATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE,
        axis=MVector3D(0, -1, 0),
    )
    LOWER = BoneSetting(
        name="下半身",
        parents=("腰", "グルーブ", "センター"),
        relatives=("足中心",),
        tails=("足中心",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE,
        axis=MVector3D(0, -1, 0),
    )
    LEG_CENTER = BoneSetting(
        name="足中心",
        parents=("下半身", "腰", "グルーブ", "センター"),
        relatives=MVector3D(0, -1, 0),
        tails=[],
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE,
        axis=MVector3D(0, -1, 0),
    )
    UPPER = BoneSetting(
        name="上半身",
        parents=("腰", "グルーブ", "センター"),
        relatives=("上半身2", "首根元"),
        tails=("上半身2", "首根元"),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(0, 1, 0),
    )
    UPPER2 = BoneSetting(
        name="上半身2",
        parents=("上半身",),
        relatives=("上半身3", "首根元"),
        tails=("上半身3", "首根元"),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(0, 1, 0),
    )
    UPPER3 = BoneSetting(
        name="上半身3",
        parents=("上半身2",),
        relatives=("首根元",),
        tails=("首根元",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(0, 1, 0),
    )
    ARM_CENTER = BoneSetting(
        name="首根元",
        parents=("上半身3", "上半身2", "上半身"),
        relatives=("首",),
        tails=("首",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(0, 1, 0),
    )
    NECK = BoneSetting(
        name="首",
        parents=("首根元",),
        relatives=("頭",),
        tails=("頭",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(0, 1, 0),
    )
    HEAD = BoneSetting(
        name="頭",
        parents=("首",),
        relatives=MVector3D(0, 1, 0),
        tails=[],
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE,
        axis=MVector3D(0, 1, 0),
    )
    EYES = BoneSetting(
        name="両目",
        parents=("頭",),
        relatives=MVector3D(0, 1, 0),
        tails=("左目", "右目"),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE,
        axis=MVector3D(0, 1, 0),
    )
    LEFT_EYE = BoneSetting(
        name="左目",
        parents=("頭",),
        relatives=MVector3D(0, 1, 0),
        tails=[],
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.IS_EXTERNAL_ROTATION,
        axis=MVector3D(0, 1, 0),
    )
    RIGHT_EYE = BoneSetting(
        name="右目",
        parents=("頭",),
        relatives=MVector3D(0, 1, 0),
        tails=[],
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.IS_EXTERNAL_ROTATION,
        axis=MVector3D(0, 1, 0),
    )

    RIGHT_BUST = BoneSetting(
        name="右胸",
        parents=("上半身3", "上半身2", "上半身"),
        relatives=("上半身3", "上半身2", "上半身"),
        tails=[],
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE,
        axis=MVector3D(0, 0, -1),
    )
    RIGHT_SHOULDER_P = BoneSetting(
        name="右肩P",
        parents=("首根元", "上半身3", "上半身2", "上半身"),
        relatives=MVector3D(0, 1, 0),
        tails=("右肩",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_SHOULDER = BoneSetting(
        name="右肩",
        parents=("右肩P", "首根元", "上半身3", "上半身2", "上半身"),
        relatives=("右腕",),
        tails=("右腕",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_SHOULDER_C = BoneSetting(
        name="右肩C",
        parents=("右肩",),
        relatives=MVector3D(0, 1, 0),
        tails=("右ひじ",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.IS_EXTERNAL_ROTATION,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_ARM = BoneSetting(
        name="右腕",
        parents=("右肩C", "右肩"),
        relatives=("右ひじ",),
        tails=("右ひじ",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_ARM_TWIST = BoneSetting(
        name="右腕捩",
        parents=("右腕",),
        relatives=MVector3D(0, 1, 0),
        tails=("右ひじ",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.HAS_FIXED_AXIS,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_ARM_TWIST1 = BoneSetting(
        name="右腕捩1",
        parents=("右腕捩",),
        relatives=MVector3D(0, 1, 0),
        tails=("右ひじ",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.HAS_FIXED_AXIS,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_ARM_TWIST2 = BoneSetting(
        name="右腕捩2",
        parents=("右腕捩",),
        relatives=MVector3D(0, 1, 0),
        tails=("右ひじ",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.HAS_FIXED_AXIS,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_ARM_TWIST3 = BoneSetting(
        name="右腕捩3",
        parents=("右腕捩",),
        relatives=MVector3D(0, 1, 0),
        tails=("右ひじ",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.HAS_FIXED_AXIS,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_ELBOW = BoneSetting(
        name="右ひじ",
        parents=("右腕捩", "右腕"),
        relatives=("右手首",),
        tails=("右手首",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_HAND_TWIST = BoneSetting(
        name="右手捩",
        parents=("右ひじ",),
        relatives=MVector3D(0, 1, 0),
        tails=("右手首",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.HAS_FIXED_AXIS,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_HAND_TWIST1 = BoneSetting(
        name="右手捩1",
        parents=("右手捩",),
        relatives=MVector3D(0, 1, 0),
        tails=("右手首",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.HAS_FIXED_AXIS,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_HAND_TWIST2 = BoneSetting(
        name="右手捩2",
        parents=("右手捩",),
        relatives=MVector3D(0, 1, 0),
        tails=("右手首",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.HAS_FIXED_AXIS,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_HAND_TWIST3 = BoneSetting(
        name="右手捩3",
        parents=("右手捩",),
        relatives=MVector3D(0, 1, 0),
        tails=("右手首",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.HAS_FIXED_AXIS,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_WRIST = BoneSetting(
        name="右手首",
        parents=("右手捩", "右ひじ"),
        relatives=("右中指１", "右人指１", "右薬指１", "右小指１"),
        tails=("右中指１", "右人指１", "右薬指１", "右小指１"),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_THUMB0 = BoneSetting(
        name="右親指０",
        parents=("右手首",),
        relatives=("右親指１",),
        tails=("右親指１",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_THUMB1 = BoneSetting(
        name="右親指１",
        parents=("右親指０",),
        relatives=("右親指２",),
        tails=("右親指２",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_THUMB2 = BoneSetting(
        name="右親指２",
        parents=("右親指１",),
        relatives=MVector3D(0, 1, 0),
        tails=("右親指先",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_THUMB_TAIL = BoneSetting(
        name="右親指先",
        parents=("右親指２",),
        relatives=MVector3D(0, 1, 0),
        tails=[],
        flag=BoneFlg.CAN_ROTATE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_INDEX0 = BoneSetting(
        name="右人指１",
        parents=("右手首",),
        relatives=("右人指２",),
        tails=("右人指３",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_INDEX1 = BoneSetting(
        name="右人指２",
        parents=("右人指１",),
        relatives=("右人指３",),
        tails=("右人指３",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_INDEX2 = BoneSetting(
        name="右人指３",
        parents=("右人指２",),
        relatives=MVector3D(0, 1, 0),
        tails=("右人指先",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_INDEX_TAIL = BoneSetting(
        name="右人指先",
        parents=("右人指３",),
        relatives=MVector3D(0, 1, 0),
        tails=[],
        flag=BoneFlg.CAN_ROTATE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_MIDDLE0 = BoneSetting(
        name="右中指１",
        parents=("右手首",),
        relatives=("右中指２",),
        tails=("右中指３",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_MIDDLE1 = BoneSetting(
        name="右中指２",
        parents=("右中指１",),
        relatives=("右中指３",),
        tails=("右中指３",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_MIDDLE2 = BoneSetting(
        name="右中指３",
        parents=("右中指２",),
        relatives=MVector3D(0, 1, 0),
        tails=("右中指先",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_MIDDLE_TAIL = BoneSetting(
        name="右中指先",
        parents=("右中指３",),
        relatives=MVector3D(0, 1, 0),
        tails=[],
        flag=BoneFlg.CAN_ROTATE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_RING0 = BoneSetting(
        name="右薬指１",
        parents=("右手首",),
        relatives=("右薬指２",),
        tails=("右薬指３",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_RING1 = BoneSetting(
        name="右薬指２",
        parents=("右薬指１",),
        relatives=("右薬指３",),
        tails=("右薬指３",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_RING2 = BoneSetting(
        name="右薬指３",
        parents=("右薬指２",),
        relatives=MVector3D(0, 1, 0),
        tails=("右薬指先",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_RING_TAIL = BoneSetting(
        name="右薬指先",
        parents=("右薬指３",),
        relatives=MVector3D(0, 1, 0),
        tails=[],
        flag=BoneFlg.CAN_ROTATE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_PINKY0 = BoneSetting(
        name="右小指１",
        parents=("右手首",),
        relatives=("右小指２",),
        tails=("右小指３",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_PINKY1 = BoneSetting(
        name="右小指２",
        parents=("右小指１",),
        relatives=("右小指３",),
        tails=("右小指３",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_PINKY2 = BoneSetting(
        name="右小指３",
        parents=("右小指２",),
        relatives=MVector3D(0, 1, 0),
        tails=("右小指先",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_PINKY_TAIL = BoneSetting(
        name="右小指先",
        parents=("右小指３",),
        relatives=MVector3D(0, 1, 0),
        tails=[],
        flag=BoneFlg.CAN_ROTATE,
        axis=MVector3D(-1, 0, 0),
    )
    RIGHT_WRIST_CANCEL = BoneSetting(
        name="腰キャンセル右",
        parents=("足中心", "下半身"),
        relatives=MVector3D(0, -1, 0),
        tails=("右足",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_EXTERNAL_ROTATION,
        axis=MVector3D(0, -1, 0),
    )
    RIGHT_LEG = BoneSetting(
        name="右足",
        parents=("腰キャンセル右", "足中心", "下半身"),
        relatives=("右ひざ",),
        tails=("右ひざ",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(0, -1, 0),
    )
    RIGHT_KNEE = BoneSetting(
        name="右ひざ",
        parents=("右足",),
        relatives=("右足首",),
        tails=("右足首",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(0, -1, 0),
    )
    RIGHT_ANKLE = BoneSetting(
        name="右足首",
        parents=("右ひざ",),
        relatives=("右つま先",),
        tails=("右つま先",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE,
        axis=MVector3D(0, -1, 0),
    )
    RIGHT_TOE = BoneSetting(
        name="右つま先",
        parents=("右足首",),
        relatives=MVector3D(0, -1, 0),
        tails=[],
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE,
        axis=MVector3D(0, -1, 0),
    )
    RIGHT_LEG_D = BoneSetting(
        name="右足D",
        parents=("腰キャンセル右", "下半身"),
        relatives=("右ひざD",),
        tails=("右ひざD",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.IS_EXTERNAL_ROTATION | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(0, -1, 0),
    )
    RIGHT_KNEE_D = BoneSetting(
        name="右ひざD",
        parents=("右足D",),
        relatives=("右足首D",),
        tails=("右足首D",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.IS_EXTERNAL_ROTATION | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(0, -1, 0),
    )
    RIGHT_ANKLE_D = BoneSetting(
        name="右足首D",
        parents=("右ひざD",),
        relatives=("右足先EX",),
        tails=("右足先EX",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.IS_EXTERNAL_ROTATION | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(0, -1, 0),
    )
    RIGHT_TOE_EX = BoneSetting(
        name="右足先EX",
        parents=("右足首D",),
        relatives=MVector3D(0, -1, 0),
        tails=[],
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE,
        axis=MVector3D(0, -1, 0),
    )
    RIGHT_LEG_IK_PARENT = BoneSetting(
        name="右足IK親",
        parents=("全ての親",),
        relatives=MVector3D(0, 1, 0),
        tails=("右足ＩＫ",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_TRANSLATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE,
        axis=MVector3D(0, 1, 0),
    )
    RIGHT_LEG_IK = BoneSetting(
        name="右足ＩＫ",
        parents=("右足IK親", "全ての親"),
        relatives=("右つま先ＩＫ",),
        tails=("右つま先ＩＫ",),
        flag=BoneFlg.CAN_ROTATE
        | BoneFlg.CAN_TRANSLATE
        | BoneFlg.CAN_MANIPULATE
        | BoneFlg.IS_VISIBLE
        | BoneFlg.IS_IK
        | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(0, 1, 0),
    )
    RIGHT_TOE_IK = BoneSetting(
        name="右つま先ＩＫ",
        parents=("右足ＩＫ",),
        relatives=MVector3D(0, -1, 0),
        tails=[],
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_TRANSLATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.IS_IK,
        axis=MVector3D(0, -1, 0),
    )

    LEFT_BUST = BoneSetting(
        name="左胸",
        parents=("上半身3", "上半身2", "上半身"),
        relatives=("上半身3", "上半身2", "上半身"),
        tails=[],
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE,
        axis=MVector3D(0, 0, -1),
    )
    LEFT_SHOULDER_P = BoneSetting(
        name="左肩P",
        parents=("首根元", "上半身3", "上半身2", "上半身"),
        relatives=MVector3D(0, 1, 0),
        tails=("左肩",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_SHOULDER = BoneSetting(
        name="左肩",
        parents=("左肩P", "首根元", "上半身3", "上半身2", "上半身"),
        relatives=("左腕",),
        tails=("左腕",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_SHOULDER_C = BoneSetting(
        name="左肩C",
        parents=("左肩",),
        relatives=MVector3D(0, 1, 0),
        tails=("左ひじ",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.IS_EXTERNAL_ROTATION,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_ARM = BoneSetting(
        name="左腕",
        parents=("左肩C", "左肩"),
        relatives=("左ひじ",),
        tails=("左ひじ",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_ARM_TWIST = BoneSetting(
        name="左腕捩",
        parents=("左腕",),
        relatives=MVector3D(0, 1, 0),
        tails=("左ひじ",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.HAS_FIXED_AXIS,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_ARM_TWIST1 = BoneSetting(
        name="左腕捩1",
        parents=("左腕捩",),
        relatives=MVector3D(0, 1, 0),
        tails=("左ひじ",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.HAS_FIXED_AXIS,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_ARM_TWIST2 = BoneSetting(
        name="左腕捩2",
        parents=("左腕捩",),
        relatives=MVector3D(0, 1, 0),
        tails=("左ひじ",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.HAS_FIXED_AXIS,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_ARM_TWIST3 = BoneSetting(
        name="左腕捩3",
        parents=("左腕捩",),
        relatives=MVector3D(0, 1, 0),
        tails=("左ひじ",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.HAS_FIXED_AXIS,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_ELBOW = BoneSetting(
        name="左ひじ",
        parents=("左腕捩", "左腕"),
        relatives=("左手首",),
        tails=("左手首",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_HAND_TWIST = BoneSetting(
        name="左手捩",
        parents=("左ひじ",),
        relatives=MVector3D(0, 1, 0),
        tails=("左手首",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.HAS_FIXED_AXIS,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_HAND_TWIST1 = BoneSetting(
        name="左手捩1",
        parents=("左手捩",),
        relatives=MVector3D(0, 1, 0),
        tails=("左手首",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.HAS_FIXED_AXIS,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_HAND_TWIST2 = BoneSetting(
        name="左手捩2",
        parents=("左手捩",),
        relatives=MVector3D(0, 1, 0),
        tails=("左手首",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.HAS_FIXED_AXIS,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_HAND_TWIST3 = BoneSetting(
        name="左手捩3",
        parents=("左手捩",),
        relatives=MVector3D(0, 1, 0),
        tails=("左手首",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.HAS_FIXED_AXIS,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_WRIST = BoneSetting(
        name="左手首",
        parents=("左手捩", "左ひじ"),
        relatives=("左中指１", "左人指１", "左薬指１", "左小指１"),
        tails=("左中指１", "左人指１", "左薬指１", "左小指１"),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_THUMB0 = BoneSetting(
        name="左親指０",
        parents=("左手首",),
        relatives=("左親指１",),
        tails=("左親指１",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_THUMB1 = BoneSetting(
        name="左親指１",
        parents=("左親指０",),
        relatives=("左親指２",),
        tails=("左親指２",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_THUMB2 = BoneSetting(
        name="左親指２",
        parents=("左親指１",),
        relatives=MVector3D(0, 1, 0),
        tails=("左親指先",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_THUMB_TAIL = BoneSetting(
        name="左親指先",
        parents=("左親指２",),
        relatives=MVector3D(0, 1, 0),
        tails=[],
        flag=BoneFlg.CAN_ROTATE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_INDEX0 = BoneSetting(
        name="左人指１",
        parents=("左手首",),
        relatives=("左人指２",),
        tails=("左人指３",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_INDEX1 = BoneSetting(
        name="左人指２",
        parents=("左人指１",),
        relatives=("左人指３",),
        tails=("左人指３",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_INDEX2 = BoneSetting(
        name="左人指３",
        parents=("左人指２",),
        relatives=MVector3D(0, 1, 0),
        tails=("左人指先",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_INDEX_TAIL = BoneSetting(
        name="左人指先",
        parents=("左人指３",),
        relatives=MVector3D(0, 1, 0),
        tails=[],
        flag=BoneFlg.CAN_ROTATE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_MIDDLE0 = BoneSetting(
        name="左中指１",
        parents=("左手首",),
        relatives=("左中指２",),
        tails=("左中指３",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_MIDDLE1 = BoneSetting(
        name="左中指２",
        parents=("左中指１",),
        relatives=("左中指３",),
        tails=("左中指３",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_MIDDLE2 = BoneSetting(
        name="左中指３",
        parents=("左中指２",),
        relatives=MVector3D(0, 1, 0),
        tails=("左中指先",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_MIDDLE_TAIL = BoneSetting(
        name="左中指先",
        parents=("左中指３",),
        relatives=MVector3D(0, 1, 0),
        tails=[],
        flag=BoneFlg.CAN_ROTATE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_RING0 = BoneSetting(
        name="左薬指１",
        parents=("左手首",),
        relatives=("左薬指２",),
        tails=("左薬指３",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_RING1 = BoneSetting(
        name="左薬指２",
        parents=("左薬指１",),
        relatives=("左薬指３",),
        tails=("左薬指３",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_RING2 = BoneSetting(
        name="左薬指３",
        parents=("左薬指２",),
        relatives=MVector3D(0, 1, 0),
        tails=("左薬指先",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_RING_TAIL = BoneSetting(
        name="左薬指先",
        parents=("左薬指３",),
        relatives=MVector3D(0, 1, 0),
        tails=[],
        flag=BoneFlg.CAN_ROTATE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_PINKY0 = BoneSetting(
        name="左小指１",
        parents=("左手首",),
        relatives=("左小指２",),
        tails=("左小指３",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_PINKY1 = BoneSetting(
        name="左小指２",
        parents=("左小指１",),
        relatives=("左小指３",),
        tails=("左小指３",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_PINKY2 = BoneSetting(
        name="左小指３",
        parents=("左小指２",),
        relatives=MVector3D(0, 1, 0),
        tails=("左小指先",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_PINKY_TAIL = BoneSetting(
        name="左小指先",
        parents=("左小指３",),
        relatives=MVector3D(0, 1, 0),
        tails=[],
        flag=BoneFlg.CAN_ROTATE,
        axis=MVector3D(-1, 0, 0),
    )
    LEFT_WRIST_CANCEL = BoneSetting(
        name="腰キャンセル左",
        parents=("足中心", "下半身"),
        relatives=MVector3D(0, -1, 0),
        tails=("左足",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_EXTERNAL_ROTATION,
        axis=MVector3D(0, -1, 0),
    )
    LEFT_LEG = BoneSetting(
        name="左足",
        parents=("腰キャンセル左", "足中心", "下半身"),
        relatives=("左ひざ",),
        tails=("左ひざ",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(0, -1, 0),
    )
    LEFT_KNEE = BoneSetting(
        name="左ひざ",
        parents=("左足",),
        relatives=("左足首",),
        tails=("左足首",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(0, -1, 0),
    )
    LEFT_ANKLE = BoneSetting(
        name="左足首",
        parents=("左ひざ",),
        relatives=("左つま先",),
        tails=("左つま先",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE,
        axis=MVector3D(0, -1, 0),
    )
    LEFT_TOE = BoneSetting(
        name="左つま先",
        parents=("左足首",),
        relatives=MVector3D(0, -1, 0),
        tails=[],
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE,
        axis=MVector3D(0, -1, 0),
    )
    LEFT_LEG_D = BoneSetting(
        name="左足D",
        parents=("腰キャンセル左", "下半身"),
        relatives=("左ひざD",),
        tails=("左ひざD",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.IS_EXTERNAL_ROTATION | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(0, -1, 0),
    )
    LEFT_KNEE_D = BoneSetting(
        name="左ひざD",
        parents=("左足D",),
        relatives=("左足首D",),
        tails=("左足首D",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.IS_EXTERNAL_ROTATION | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(0, -1, 0),
    )
    LEFT_ANKLE_D = BoneSetting(
        name="左足首D",
        parents=("左ひざD",),
        relatives=("左足先EX",),
        tails=("左足先EX",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.IS_EXTERNAL_ROTATION | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(0, -1, 0),
    )
    LEFT_TOE_EX = BoneSetting(
        name="左足先EX",
        parents=("左足首D",),
        relatives=MVector3D(0, -1, 0),
        tails=[],
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE,
        axis=MVector3D(0, -1, 0),
    )
    LEFT_LEG_IK_PARENT = BoneSetting(
        name="左足IK親",
        parents=("全ての親",),
        relatives=MVector3D(0, 1, 0),
        tails=("左足ＩＫ",),
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_TRANSLATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE,
        axis=MVector3D(0, 1, 0),
    )
    LEFT_LEG_IK = BoneSetting(
        name="左足ＩＫ",
        parents=("左足IK親", "全ての親"),
        relatives=("左つま先ＩＫ",),
        tails=("左つま先ＩＫ",),
        flag=BoneFlg.CAN_ROTATE
        | BoneFlg.CAN_TRANSLATE
        | BoneFlg.CAN_MANIPULATE
        | BoneFlg.IS_VISIBLE
        | BoneFlg.IS_IK
        | BoneFlg.TAIL_IS_BONE,
        axis=MVector3D(0, 1, 0),
    )
    LEFT_TOE_IK = BoneSetting(
        name="左つま先ＩＫ",
        parents=("左足ＩＫ",),
        relatives=MVector3D(0, -1, 0),
        tails=[],
        flag=BoneFlg.CAN_ROTATE | BoneFlg.CAN_TRANSLATE | BoneFlg.CAN_MANIPULATE | BoneFlg.IS_VISIBLE | BoneFlg.IS_IK,
        axis=MVector3D(0, -1, 0),
    )


STANDARD_BONE_NAMES: dict[str, BoneSetting] = dict([(bs.value.name, bs.value) for bs in BoneSettings])
"""準標準ボーン名前とEnumのキーの辞書"""
