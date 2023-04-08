import os
import sys
from datetime import datetime
from typing import Any, Optional

import wx

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mlib.form.parts.spin_ctrl import WheelSpinCtrl, WheelSpinCtrlDouble
from mlib.pmx.canvas import CanvasPanel
from mlib.base.logger import MLogger
from mlib.form.base_frame import BaseFrame
from mlib.form.base_panel import BasePanel
from mlib.form.base_worker import BaseWorker
from mlib.form.parts.console_ctrl import ConsoleCtrl
from mlib.form.parts.file_ctrl import MFilePickerCtrl
from mlib.pmx.pmx_collection import PmxModel
from mlib.pmx.pmx_reader import PmxReader
from mlib.utils.file_utils import save_histories, separate_path
from mlib.vmd.vmd_collection import VmdMotion
from mlib.vmd.vmd_reader import VmdReader

logger = MLogger(os.path.basename(__file__))
__ = logger.get_text


class FilePanel(BasePanel):
    def __init__(self, frame: BaseFrame, tab_idx: int, *args, **kw):
        super().__init__(frame, tab_idx, *args, **kw)

        self.pmx_reader = PmxReader()
        self.vmd_reader = VmdReader()

        self._initialize_ui()

    def _initialize_ui(self):
        self.model_ctrl = MFilePickerCtrl(
            self.frame,
            self,
            self.pmx_reader,
            key="model_pmx",
            title="表示モデル",
            is_show_name=True,
            name_spacer=20,
            is_save=False,
            tooltip="PMXモデル",
            file_change_event=self.on_change_model_pmx,
        )
        self.model_ctrl.set_parent_sizer(self.root_sizer)

        self.dress_ctrl = MFilePickerCtrl(
            self.frame,
            self,
            self.pmx_reader,
            key="dress_pmx",
            title="衣装モデル",
            is_show_name=True,
            name_spacer=20,
            is_save=False,
            tooltip="PMX衣装モデル",
            file_change_event=self.on_change_dress_pmx,
        )
        self.dress_ctrl.set_parent_sizer(self.root_sizer)

        self.motion_ctrl = MFilePickerCtrl(
            self.frame,
            self,
            self.vmd_reader,
            key="motion_vmd",
            title="表示モーション",
            is_show_name=True,
            name_spacer=20,
            is_save=False,
            tooltip="VMDモーションデータ",
            file_change_event=self.on_change_motion,
        )
        self.motion_ctrl.set_parent_sizer(self.root_sizer)

        self.output_pmx_ctrl = MFilePickerCtrl(
            self.frame,
            self,
            self.pmx_reader,
            title="出力先",
            is_show_name=False,
            is_save=True,
            tooltip="実際は動いてないよ",
        )
        self.output_pmx_ctrl.set_parent_sizer(self.root_sizer)

        self.console_ctrl = ConsoleCtrl(self.frame, self, rows=300)
        self.console_ctrl.set_parent_sizer(self.root_sizer)

        self.root_sizer.Add(wx.StaticLine(self, wx.ID_ANY), wx.GROW)
        self.fit()

    def on_change_model_pmx(self, event: wx.Event):
        self.model_ctrl.unwrap()
        if self.model_ctrl.read_name():
            self.model_ctrl.read_digest()
            dir_path, file_name, file_ext = separate_path(self.model_ctrl.path)
            model_path = os.path.join(dir_path, f"{file_name}_{datetime.now():%Y%m%d_%H%M%S}{file_ext}")
            self.output_pmx_ctrl.path = model_path

    def on_change_dress_pmx(self, event: wx.Event):
        self.dress_ctrl.unwrap()
        if self.dress_ctrl.read_name():
            self.dress_ctrl.read_digest()

    def on_change_motion(self, event: wx.Event):
        self.motion_ctrl.unwrap()
        if self.motion_ctrl.read_name():
            self.motion_ctrl.read_digest()


class PmxLoadWorker(BaseWorker):
    def __init__(self, panel: BasePanel, result_event: wx.Event) -> None:
        super().__init__(panel, result_event)

    def thread_execute(self):
        file_panel: FilePanel = self.panel

        self.result_data = (
            file_panel.pmx_reader.read_by_filepath(file_panel.model_ctrl.path),
            file_panel.pmx_reader.read_by_filepath(file_panel.dress_ctrl.path),
            file_panel.vmd_reader.read_by_filepath(file_panel.motion_ctrl.path),
        )


class ConfigPanel(CanvasPanel):
    def __init__(self, frame: BaseFrame, tab_idx: int, *args, **kw):
        super().__init__(frame, tab_idx, 500, 800, *args, **kw)

        self._initialize_ui()
        self._initialize_event()

    def _initialize_ui(self):
        self.config_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.config_sizer.Add(self.canvas, 0, wx.EXPAND | wx.ALL, 0)

        self.btn_sizer = wx.BoxSizer(wx.VERTICAL)

        # キーフレ
        self.frame_ctrl = WheelSpinCtrl(self, change_event=self.on_change_frame, initial=0, min=-100, max=10000, size=wx.Size(80, -1))
        self.btn_sizer.Add(self.frame_ctrl, 0, wx.ALL, 5)

        # キーフレ
        self.double_ctrl = WheelSpinCtrlDouble(self, change_event=self.on_change_frame, initial=0, min=-100, max=10000, inc=0.01, size=wx.Size(80, -1))
        self.btn_sizer.Add(self.double_ctrl, 0, wx.ALL, 5)

        # 再生
        self.play_btn = wx.Button(self, wx.ID_ANY, "Play", wx.DefaultPosition, wx.Size(100, 50))
        self.btn_sizer.Add(self.play_btn, 0, wx.ALL, 5)

        self.config_sizer.Add(self.btn_sizer, 0, wx.ALL, 0)
        self.root_sizer.Add(self.config_sizer, 0, wx.ALL, 0)

        self.fit()

    def _initialize_event(self):
        self.play_btn.Bind(wx.EVT_BUTTON, self.on_play)

    def on_play(self, event: wx.Event):
        self.canvas.on_play(event)
        self.play_btn.SetLabelText("Stop" if self.canvas.playing else "Play")

    def on_change_frame(self, event: wx.Event):
        self.fno = self.frame_ctrl.GetValue()
        self.canvas.change_motion(event)

    def frame_forward(self):
        self.fno += 1
        self.frame_ctrl.SetValue(self.fno)

    def frame_back(self):
        self.fno = max(0, self.fno - 1)
        self.frame_ctrl.SetValue(self.fno)


class TestFrame(BaseFrame):
    def __init__(self, app) -> None:
        super().__init__(
            app,
            history_keys=["model_pmx", "dress_pmx", "motion_vmd"],
            title="Mu Test Frame",
            size=wx.Size(1000, 800),
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
                if not self.file_panel.model_ctrl.data:
                    # 設定タブにうつった時に読み込む
                    self.config_panel.canvas.clear_model_set()
                    self.save_histories()

                    self.worker.start()
                else:
                    # 既に読み取りが完了していたらそのまま表示
                    self.notebook.ChangeSelection(self.config_panel.tab_idx)

    def save_histories(self):
        self.file_panel.model_ctrl.save_path()
        self.file_panel.dress_ctrl.save_path()
        self.file_panel.motion_ctrl.save_path()

        save_histories(self.histories)

    def on_result(self, result: bool, data: Optional[Any], elapsed_time: str):
        self.file_panel.console_ctrl.write(f"\n----------------\n{elapsed_time}")

        if not (result and data):
            return

        data1, data2, data3 = data
        model: PmxModel = data1
        dress: PmxModel = data2
        motion: VmdMotion = data3
        self.file_panel.model_ctrl.data = model
        self.file_panel.dress_ctrl.data = dress
        self.file_panel.motion_ctrl.data = motion

        try:
            self.config_panel.canvas.set_context()
            self.config_panel.canvas.append_model_set(self.file_panel.model_ctrl.data, self.file_panel.motion_ctrl.data)
            self.config_panel.canvas.append_model_set(self.file_panel.dress_ctrl.data, VmdMotion())
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
    MLogger.initialize(
        lang="en",
        root_dir=os.path.join(os.path.dirname(__file__), "..", "mlib"),
        level=20,
    )

    app = MuApp()
    app.MainLoop()
