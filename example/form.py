import os
import sys

import wx

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mlib.form.base_frame import BaseFrame
from mlib.form.base_panel import BasePanel


class TestPanel(BasePanel):
    def __init__(self, parent, *args, **kw):
        super().__init__(parent, *args, **kw)


class TestFrame(BaseFrame):
    def __init__(self) -> None:
        super().__init__(title="Mu Test Frame", size=wx.Size(600, 650))

        # ファイルタブ
        self.file_panel = TestPanel(self.note_book)
        self.note_book.AddPage(self.file_panel, "ファイル", False)


class MuApp(wx.App):
    def __init__(self, redirect=False, filename=None, useBestVisual=False, clearSigInt=True):
        super().__init__(redirect, filename, useBestVisual, clearSigInt)
        self.frame = TestFrame()
        self.frame.Show()


if __name__ == "__main__":
    app = MuApp()
    app.MainLoop()
