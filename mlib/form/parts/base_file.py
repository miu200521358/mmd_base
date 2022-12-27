import glob
import os
import re
import wx

from enum import Enum, unique

from mlib.pmx.pmx_collection import PmxModel
from mlib.pmx.pmx_reader import PmxReader
from mlib.pmx.pmx_writer import PmxWriter


@unique
class FileType(Enum):
    """ファイル種別"""

    VMD_VPD = "VMD/VPDファイル (*.vmd, *.vpd)|*.vmd;*.vpd|すべてのファイル (*.*)|*.*"
    VMD = "VMDファイル (*.vmd)|*.vmd|すべてのファイル (*.*)|*.*"
    PMX = "PMXファイル (*.pmx)|*.pmx|すべてのファイル (*.*)|*.*"
    CSV = "CSVファイル (*.csv)|*.csv|すべてのファイル (*.*)|*.*"
    VRM = "VRMファイル (*.vrm)|*.vrm|すべてのファイル (*.*)|*.*"


class BaseFilePickerCtrl:
    def __init__(self) -> None:
        pass
