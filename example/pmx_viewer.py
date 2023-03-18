import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import wx

from mlib.pmx.viewer import PmxCanvas


class PmxPanel(wx.Panel):
    def __init__(self, parent, *args, **kw):
        parser = argparse.ArgumentParser(description="MMD model viewer sample.")
        parser.add_argument("--pmx", type=str, help="MMD model file name.")
        parser.add_argument("--motion", type=str, help="MMD motion file name.")
        args = parser.parse_args()

        self.parent = parent
        wx.Panel.__init__(self, parent)
        self.SetBackgroundColour("#626D58")

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.canvas = PmxCanvas(self, args.pmx, args.motion, 800, 800)
        self.sizer.Add(self.canvas, 0, wx.ALL | wx.EXPAND, 0)

        self.btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        # リセット
        self.reset_btn = wx.Button(self, wx.ID_ANY, "Reset", wx.DefaultPosition, wx.Size(100, 50))
        self.reset_btn.Bind(wx.EVT_BUTTON, self.reset)
        self.btn_sizer.Add(self.reset_btn, 0, wx.ALL, 5)

        # キャプチャー
        self.capture_btn = wx.Button(self, wx.ID_ANY, "Capture", wx.DefaultPosition, wx.Size(100, 50))
        self.capture_btn.Bind(wx.EVT_BUTTON, self.capture)
        self.btn_sizer.Add(self.capture_btn, 0, wx.ALL, 5)

        # キーフレ
        self.frame_slider = wx.Slider(self, wx.ID_ANY, value=0, minValue=0, maxValue=7000, pos=wx.DefaultPosition, size=wx.Size(300, 50))
        self.frame_slider.Bind(wx.EVT_SLIDER, self.on_change_frame)
        self.btn_sizer.Add(self.frame_slider, 0, wx.EXPAND | wx.ALL, 5)

        self.sizer.Add(self.btn_sizer, 0, wx.EXPAND | wx.ALL, 5)

        self.Layout()
        self.fit()

    def on_change_frame(self, event: wx.Event):
        self.canvas.frame = self.frame_slider.GetValue()
        self.canvas.change_motion(event)

    def fit(self):
        self.SetSizer(self.sizer)
        self.Layout()
        self.sizer.Fit(self.parent)

    def reset(self, event: wx.Event):
        self.canvas.reset(event)

    def capture(self, event: wx.Event):
        self.canvas.on_capture(event)


class PmxFrame(wx.Frame):
    def __init__(self, *args, **kw):
        self.size = (1000, 1000)
        wx.Frame.__init__(
            self,
            None,
            title="Pmx wx Frame",
            size=self.size,
            style=wx.DEFAULT_FRAME_STYLE | wx.FULL_REPAINT_ON_RESIZE,
        )

        self.Bind(wx.EVT_CLOSE, self.onClose)
        self.panel = PmxPanel(self)

    def onClose(self, event: wx.Event):
        self.Destroy()
        sys.exit(0)


class PmxApp(wx.App):
    def __init__(self, redirect=False, filename=None, useBestVisual=False, clearSigInt=True):
        super().__init__(redirect, filename, useBestVisual, clearSigInt)
        self.frame = PmxFrame()
        self.frame.Show()


if __name__ == "__main__":
    app = PmxApp()
    app.MainLoop()
