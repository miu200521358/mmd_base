import wx

from mlib.service.form.base_frame import BaseFrame


class BasePanel(wx.Panel):
    def __init__(self, frame: BaseFrame, tab_idx: int, *args, **kw):
        self.frame = frame
        self.tab_idx = tab_idx
        super().__init__(self.frame.notebook, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DLIGHT))

        self.root_sizer = wx.BoxSizer(wx.VERTICAL)
        self.is_fix_tab = False

    def fit(self):
        self.SetSizer(self.root_sizer)
        self.Layout()
