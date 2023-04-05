import sys

import wx
from mlib.form.base_notebook import BaseNotebook
from mlib.utils.file_utils import get_root_dir


class BaseFrame(wx.Frame):
    def __init__(self, app: wx.App, title: str, size: wx.Size, *args, **kw):
        wx.Frame.__init__(
            self,
            None,
            title=title,
            size=size,
            style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL,
        )
        self.app = app
        self.root_dir = get_root_dir()
        self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNSHADOW))

        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.notebook = BaseNotebook(self)

        self.Centre(wx.BOTH)

    def on_close(self, event: wx.Event):
        self.Destroy()
        sys.exit(0)
