import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse

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
        self.canvas = PmxCanvas(self, args.pmx, 600, 600)
        self.sizer.Add(self.canvas, 0, wx.ALL | wx.EXPAND, 0)

        self.rot_btn = wx.Button(
            self, -1, label="Start/Stop\nrotation", pos=(1130, 10), size=(100, 50)
        )
        self.rot_btn.BackgroundColour = (125, 125, 125)
        self.rot_btn.ForegroundColour = (0, 0, 0)
        self.sizer.Add(self.rot_btn, 0, wx.ALIGN_LEFT | wx.SHAPED, 5)

        self.rot_btn.Bind(wx.EVT_BUTTON, self.rotate)

        self.Layout()
        self.fit()

    def fit(self):
        self.SetSizer(self.sizer)
        self.Layout()
        self.sizer.Fit(self.parent)

    def rotate(self, event: wx.Event):
        if not self.canvas.rotate:
            self.canvas.rotate = True
        else:
            self.canvas.rotate = False
        self.canvas.OnDraw(event)


class PmxFrame(wx.Frame):
    def __init__(self, *args, **kw):
        self.size = (600, 800)
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
