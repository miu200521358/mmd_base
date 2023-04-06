import os
import sys

import wx

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mlib.form.base_frame import BaseFrame
from mlib.form.base_panel import BasePanel
from mlib.form.parts.base_file_ctrl import MFilePickerCtrl
from mlib.pmx.pmx_reader import PmxReader


class TestPanel(BasePanel):
    def __init__(self, frame: BaseFrame, tab_idx: int, *args, **kw):
        super().__init__(frame, tab_idx, *args, **kw)

        pmx_reader = PmxReader()

        self.model_pmx_ctrl = MFilePickerCtrl(
            self.frame,
            self,
            pmx_reader,
            key="model_pmx",
            title="テスト",
            is_show_name=True,
            name_spacer=20,
            is_save=False,
            tooltip="なんか色々",
        )
        self.model_pmx_ctrl.set_parent_sizer(self.root_sizer)

        self.output_pmx_ctrl = MFilePickerCtrl(
            self.frame,
            self,
            pmx_reader,
            key="model_pmx",
            title="出力先",
            is_show_name=False,
            is_save=True,
            tooltip="なんか色々",
        )
        self.output_pmx_ctrl.set_parent_sizer(self.root_sizer)

        self.root_sizer.Add(wx.StaticLine(self, wx.ID_ANY), wx.GROW)

        self.SetSizer(self.root_sizer)
        self.Layout()
        self.fit()


class TestFrame(BaseFrame):
    def __init__(self, app) -> None:
        super().__init__(
            app,
            history_keys=["model_pmx"],
            title="Mu Test Frame",
            size=wx.Size(800, 600),
        )

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
