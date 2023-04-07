import os
import sys
from datetime import datetime
from typing import Any, Optional

import wx

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mlib.pmx.pmx_collection import PmxModel

from mlib.pmx.canvas import PmxCanvas
from mlib.form.base_worker import BaseWorker
from mlib.base.logger import MLogger
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

        self.pmx_reader = PmxReader()

        self._initialize_ui()

    def _initialize_ui(self):
        self.model_pmx_ctrl = MFilePickerCtrl(
            self.frame,
            self,
            self.pmx_reader,
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
            self.pmx_reader,
            key="model_pmx",
            title="出力先",
            is_show_name=False,
            is_save=True,
            tooltip="なんか色々",
        )
        self.output_pmx_ctrl.set_parent_sizer(self.root_sizer)

        self.console_ctrl = ConsoleCtrl(self.frame, self, rows=300)
        self.console_ctrl.set_parent_sizer(self.root_sizer)

        self.root_sizer.Add(wx.StaticLine(self, wx.ID_ANY), wx.GROW)
        self.fit()

    def on_change_model_pmx(self, event: wx.Event):
        if self.model_pmx_ctrl.read_name():
            self.model_pmx_ctrl.read_digest()
            dir_path, file_name, file_ext = separate_path(self.model_pmx_ctrl.path)
            model_path = os.path.join(dir_path, f"{file_name}_{datetime.now():%Y%m%d_%H%M%S}{file_ext}")
            self.output_pmx_ctrl.path = model_path


class PmxLoadWorker(BaseWorker):
    def __init__(self, panel: BasePanel, result_event: wx.Event) -> None:
        super().__init__(panel, result_event)

    def thread_execute(self):
        file_panel: FilePanel = self.panel

        self.result_data = file_panel.pmx_reader.read_by_filepath(file_panel.model_pmx_ctrl.path)


class ConfigPanel(BasePanel):
    def __init__(self, frame: BaseFrame, tab_idx: int, *args, **kw):
        super().__init__(frame, tab_idx, *args, **kw)

        self._initialize_ui()

    def _initialize_ui(self):
        self.config_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.canvas = PmxCanvas(self, 400, 600)
        self.config_sizer.Add(self.canvas, 0, wx.EXPAND | wx.ALL, 0)

        # キーフレ
        self.config_sizer.Add(self.canvas.frame_ctrl, 0, wx.ALL, 5)

        self.root_sizer.Add(self.config_sizer, 0, wx.ALL, 0)

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
        self.file_panel = FilePanel(self, 0)
        self.notebook.AddPage(self.file_panel, __("ファイル"), False)

        # 設定タブ
        self.config_panel = ConfigPanel(self, 1)
        self.notebook.AddPage(self.config_panel, __("設定"), False)

        self.worker = PmxLoadWorker(self.file_panel, self.on_result)

    def on_change_tab(self, event: wx.Event):
        if self.notebook.GetSelection() == self.config_panel.tab_idx:
            self.notebook.ChangeSelection(self.file_panel.tab_idx)
            if not self.worker.started:
                if not self.file_panel.model_pmx_ctrl.data:
                    # 設定タブにうつった時に読み込む
                    self.worker.start()
                else:
                    # 既に読み取りが完了していたらそのまま表示
                    self.notebook.ChangeSelection(self.config_panel.tab_idx)

    def on_result(self, result: bool, data: Optional[Any], elapsed_time: str):
        self.file_panel.console_ctrl.write(f"\n----------------\n{elapsed_time}")

        if not (result and data):
            return

        model: PmxModel = data
        self.file_panel.model_pmx_ctrl.data = model

        try:
            self.config_panel.canvas.set_context()
            self.config_panel.canvas.set_model(self.file_panel.model_pmx_ctrl.data)
            self.config_panel.canvas.model.init_draw(self.config_panel.canvas.shader)
            self.config_panel.canvas.Refresh()
            self.notebook.ChangeSelection(self.config_panel.tab_idx)
        except:
            logger.critical(__("モデル描画初期化処理失敗"))


class MuApp(wx.App):
    def __init__(self, redirect=False, filename=None, useBestVisual=False, clearSigInt=True):
        super().__init__(redirect, filename, useBestVisual, clearSigInt)
        self.frame = TestFrame(self)
        self.frame.Show()


if __name__ == "__main__":
    app = MuApp()
    app.MainLoop()
