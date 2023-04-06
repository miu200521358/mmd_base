import os
import sys

import wx

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mlib.form.base_frame import BaseFrame
from mlib.form.base_panel import BasePanel
from mlib.form.parts.base_file_ctrl import BaseFilePickerCtrl
from mlib.pmx.pmx_reader import PmxReader


class TestPanel(BasePanel):
    def __init__(self, frame: BaseFrame, tab_idx: int, *args, **kw):
        super().__init__(frame, tab_idx, *args, **kw)

        pmx_reader = PmxReader()
        self.file_ctrl = BaseFilePickerCtrl(
            self,
            pmx_reader,
            title="テスト",
            is_show_name=True,
            name_spacer=20,
            is_save=False,
            tooltip="なんか色々",
        )
        self.file_ctrl.set_parent_sizer(self.root_sizer)

        self.SetSizer(self.root_sizer)
        self.Layout()
        self.fit()


class TestFrame(BaseFrame):
    def __init__(self, app) -> None:
        super().__init__(app, title="Mu Test Frame", size=wx.Size(800, 600))

        # ファイルタブ
        self.file_panel = TestPanel(self, 0)
        self.notebook.AddPage(self.file_panel, "ファイル", False)


class MuApp(wx.App):
    def __init__(self, redirect=False, filename=None, useBestVisual=False, clearSigInt=True):
        super().__init__(redirect, filename, useBestVisual, clearSigInt)
        self.frame = TestFrame(self)
        self.frame.Show()


if __name__ == "__main__":
    app = MuApp()
    app.MainLoop()
