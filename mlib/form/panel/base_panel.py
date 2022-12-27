import wx


class BasePanel(wx.Panel):
    def __init__(self, parent, *args, **kw):
        self.parent = parent
        wx.Panel.__init__(self, parent)
        self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DLIGHT))

    def fit(self):
        self.SetSizer(self.sizer)
        self.Layout()
