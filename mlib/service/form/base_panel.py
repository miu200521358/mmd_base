import wx

from mlib.service.form.base_frame import BaseFrame


class BasePanel(wx.Panel):
    def __init__(self, frame: BaseFrame, tab_idx: int, *args, **kw) -> None:
        self.frame = frame
        self.tab_idx = tab_idx
        super().__init__(self.frame.notebook, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DLIGHT))

        self.active_background_color = wx.Colour("PINK")
        """ボタンが有効になっている時の背景色"""

        self.root_sizer = wx.BoxSizer(wx.VERTICAL)
        self.is_fix_tab = False

        self.SetSizer(self.root_sizer)

    def fit(self) -> None:
        self.Layout()

    def Enable(self, enable: bool) -> None:
        pass
