import os
from typing import Callable, Optional

import wx

from mlib.core.logger import MLogger
from mlib.service.form.base_panel import BasePanel
from mlib.utils.file_utils import get_path

logger = MLogger(os.path.basename(__file__))
__ = logger.get_text


class ImageButton(wx.BitmapButton):
    def __init__(
        self,
        panel: BasePanel,
        image_path: str,
        size: wx.Size,
        click_evt: Callable,
        tooltip: Optional[str] = None,
    ):
        self.parent = panel
        self.image_name = image_path
        self.exec_evt = click_evt

        # 画像を読み込む
        image = wx.Image(get_path(image_path), wx.BITMAP_TYPE_ANY)

        # 画像をサイズに合わせてリサイズ
        image = image.Scale(size.x, size.y)

        # リサイズした画像をビットマップに変換
        bitmap = image.ConvertToBitmap()

        super().__init__(panel, bitmap=bitmap)
        self.Bind(wx.EVT_BUTTON, self._click)
        self.SetToolTip(tooltip)

    def _click(self, event: wx.Event):
        if self.exec_evt is not None:
            self.exec_evt(event)
