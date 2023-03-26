import sys

import wx


class BaseFrame(wx.Frame):
    def __init__(self, title: str, size: wx.Size, *args, **kw):
        wx.Frame.__init__(
            self,
            None,
            title=title,
            size=size,
            style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL,
        )
        self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNSHADOW))

        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.note_book = wx.Notebook(
            self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, 0
        )

        self.Centre(wx.BOTH)

    def on_close(self, event: wx.Event):
        self.Destroy()
        sys.exit(0)
