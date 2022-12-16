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

        self.reset_btn = wx.Button(
            self, -1, label="Reset", pos=(1130, 10), size=(100, 50)
        )
        self.reset_btn.BackgroundColour = (125, 125, 125)
        self.reset_btn.ForegroundColour = (0, 0, 0)
        self.sizer.Add(self.reset_btn, 0, wx.ALIGN_LEFT | wx.SHAPED, 5)

        self.reset_btn.Bind(wx.EVT_BUTTON, self.reset)

        self.Layout()
        self.fit()

    def fit(self):
        self.SetSizer(self.sizer)
        self.Layout()
        self.sizer.Fit(self.parent)

    def reset(self, event: wx.Event):
        self.canvas.reset()
        self.canvas.OnDraw(event)


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
    def __init__(
        self, redirect=False, filename=None, useBestVisual=False, clearSigInt=True
    ):
        super().__init__(redirect, filename, useBestVisual, clearSigInt)
        self.frame = PmxFrame()
        self.frame.Show()


if __name__ == "__main__":
    app = PmxApp()
    app.MainLoop()
