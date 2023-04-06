import os
import sys
from datetime import datetime

import wx

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mlib.base.logger import MLogger, LoggingDecoration
from mlib.form.base_frame import BaseFrame
from mlib.form.base_panel import BasePanel
from mlib.form.parts.file_ctrl import MFilePickerCtrl
from mlib.pmx.pmx_reader import PmxReader
from mlib.utils.file_utils import separate_path
from mlib.form.parts.console_ctrl import ConsoleCtrl

logger = MLogger(os.path.basename(__file__))
__ = logger.get_text


class FilePanel(BasePanel):
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
            event=self.on_change_model_pmx,
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

        self.console_ctrl = ConsoleCtrl(self.frame, self, rows=200)
        self.console_ctrl.set_parent_sizer(self.root_sizer)

        self.root_sizer.Add(wx.StaticLine(self, wx.ID_ANY), wx.GROW)

        self.SetSizer(self.root_sizer)
        self.Layout()
        self.fit()

    def on_change_model_pmx(self, event: wx.Event):
        self.model_pmx_ctrl.read_name()
        dir_path, file_name, file_ext = separate_path(self.model_pmx_ctrl.path)
        self.output_pmx_ctrl.path = os.path.join(dir_path, f"{file_name}_{datetime.now():%Y%m%d_%H%M%S}{file_ext}")
        logger.info(self.model_pmx_ctrl.path, decoration=LoggingDecoration.DECORATION_BOX, func="on_change_model_pmx", lno=10)


class ConfigPanel(BasePanel):
    def __init__(self, frame: BaseFrame, tab_idx: int, *args, **kw):
        super().__init__(frame, tab_idx, *args, **kw)


class TestFrame(BaseFrame):
    def __init__(self, app) -> None:
        super().__init__(
            app,
            history_keys=["model_pmx"],
            title="Mu Test Frame",
            size=wx.Size(800, 600),
        )

        # ファイルタブ
        self.file_panel = FilePanel(self, 0)
        self.notebook.AddPage(self.file_panel, __("ファイル"), True)

        # 設定タブ
        self.config_panel = ConfigPanel(self, 1)
        self.notebook.AddPage(self.config_panel, __("設定"), False)


class MuApp(wx.App):
    def __init__(self, redirect=False, filename=None, useBestVisual=False, clearSigInt=True):
        super().__init__(redirect, filename, useBestVisual, clearSigInt)
        self.frame = TestFrame(self)
        self.frame.Show()


if __name__ == "__main__":
    app = MuApp()
    app.MainLoop()
