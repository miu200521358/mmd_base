import argparse
import os
import sys
import multiprocessing as mp

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import wx

from mlib.base.multi import DrawProcess
from mlib.pmx.viewer import PmxCanvas
from mlib.base.logger import MLogger


class PmxPanel(wx.Panel):
    def __init__(self, parent, draw_queue: mp.Queue, compute_queue: mp.Queue):
        parser = argparse.ArgumentParser(description="MMD model viewer sample.")
        parser.add_argument("--pmx", type=str, help="MMD model file name.")
        parser.add_argument("--motion", type=str, help="MMD motion file name.")
        parser.add_argument("--level", type=int, help="MMD motion file name.")
        args = parser.parse_args()

        MLogger.initialize(lang="ja_JP", root_dir=os.path.join(os.path.dirname(__file__), "..", "mlib"), level=args.level)

        self.parent = parent
        wx.Panel.__init__(self, parent)
        self.SetBackgroundColour("#626D58")

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.canvas = PmxCanvas(self, draw_queue, compute_queue, 800, 700)

        self.sizer.Add(self.canvas, 0, wx.ALL | wx.EXPAND, 0)

        self.file_sizer = wx.BoxSizer(wx.VERTICAL)

        self.model_file_ctrl = wx.FilePickerCtrl(self, wx.ID_ANY)
        self.file_sizer.Add(self.model_file_ctrl, 1, wx.ALL | wx.EXPAND, 5)
        self.model_file_ctrl.SetPath(args.pmx)

        self.motion_file_ctrl = wx.FilePickerCtrl(self, wx.ID_ANY)
        self.file_sizer.Add(self.motion_file_ctrl, 1, wx.ALL | wx.EXPAND, 5)
        self.motion_file_ctrl.SetPath(args.motion)

        self.sizer.Add(self.file_sizer, 0, wx.EXPAND | wx.ALL, 5)

        self.btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # 読み込み
        self.load_btn = wx.Button(self, wx.ID_ANY, "Load", wx.DefaultPosition, wx.Size(100, 50))
        self.load_btn.Bind(wx.EVT_BUTTON, self.load)
        self.btn_sizer.Add(self.load_btn, 0, wx.ALL, 5)

        # 再生
        self.play_btn = wx.Button(self, wx.ID_ANY, "Play/Stop", wx.DefaultPosition, wx.Size(100, 50))
        self.play_btn.Bind(wx.EVT_BUTTON, self.play)
        self.btn_sizer.Add(self.play_btn, 0, wx.ALL, 5)

        # リセット
        self.reset_btn = wx.Button(self, wx.ID_ANY, "Reset", wx.DefaultPosition, wx.Size(100, 50))
        self.reset_btn.Bind(wx.EVT_BUTTON, self.reset)
        self.btn_sizer.Add(self.reset_btn, 0, wx.ALL, 5)

        # キャプチャー
        self.capture_btn = wx.Button(self, wx.ID_ANY, "Capture", wx.DefaultPosition, wx.Size(100, 50))
        self.capture_btn.Bind(wx.EVT_BUTTON, self.capture)
        self.btn_sizer.Add(self.capture_btn, 0, wx.ALL, 5)

        # キーフレ
        self.btn_sizer.Add(self.canvas.frame_ctrl, 0, wx.ALL, 5)

        self.sizer.Add(self.btn_sizer, 0, wx.EXPAND | wx.ALL, 5)

        self.Layout()
        self.fit()

    def fit(self):
        self.SetSizer(self.sizer)
        self.Layout()
        self.sizer.Fit(self.parent)

    def reset(self, event: wx.Event):
        self.canvas.reset(event)

    def capture(self, event: wx.Event):
        self.canvas.on_capture(event)

    def play(self, event: wx.Event):
        self.canvas.on_play(event)
        self.play_btn.SetLabelText("Stop" if self.canvas.playing else "Play")

    def load(self, event: wx.Event):
        self.canvas.on_load(event, self.model_file_ctrl.GetPath(), self.motion_file_ctrl.GetPath())


class PmxFrame(wx.Frame):
    def __init__(self, draw_queue: mp.Queue, compute_queue: mp.Queue, *args, **kw):
        self.size = (1000, 1000)
        wx.Frame.__init__(
            self,
            None,
            title="Pmx wx Frame",
            size=self.size,
            style=wx.DEFAULT_FRAME_STYLE | wx.FULL_REPAINT_ON_RESIZE,
        )

        self.Bind(wx.EVT_CLOSE, self.onClose)
        self.panel = PmxPanel(self, draw_queue, compute_queue)

    def onClose(self, event: wx.Event):
        self.Destroy()
        sys.exit(0)


class PmxApp(wx.App):
    def __init__(self, draw_queue: mp.Queue, compute_queue: mp.Queue, redirect=False, filename=None, useBestVisual=False, clearSigInt=True):
        super().__init__(redirect, filename, useBestVisual, clearSigInt)
        self.frame = PmxFrame(draw_queue, compute_queue)
        self.frame.Show()


def run_app(draw_queue: mp.Queue, compute_queue: mp.Queue):
    app = PmxApp(draw_queue, compute_queue)
    app.MainLoop()


if __name__ == "__main__":
    # 描画を担当するキュー
    draw_queue: mp.Queue = mp.Queue()
    # 計算処理を担当するキュー
    compute_queue: mp.Queue = mp.Queue()

    draw_process = DrawProcess(target=run_app, queues=[draw_queue, compute_queue])
    # compute_process = ComputeProcess(
    #     target=run_app,
    #     queues=[compute_queue],
    # )
    # gl_process = GlProcess(target=run_app, queues=[draw_queue, compute_queue])

    draw_process.start()

    draw_process.join()
