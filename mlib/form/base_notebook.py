import os
import wx

from mlib.base.logger import LoggingLevel, MLogger

logger = MLogger(os.path.basename(__file__))


class BaseNotebook(wx.Notebook):
    def __init__(self, frame: wx.Frame, *args, **kw):
        self.frame = frame
        super().__init__(frame, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)

        self._initialize_ui()
        self._initialize_event()

    def _initialize_ui(self):
        if logger.total_level <= LoggingLevel.DEBUG.value:
            # テスト（デバッグ版）の場合
            self.SetBackgroundColour("CORAL")
        elif not logger.saving:
            # ハイスペック版の場合、色変え
            self.SetBackgroundColour("BLUE")
        elif logger.out_log:
            # ログありの場合、色変え
            self.SetBackgroundColour("AQUAMARINE")
        else:
            self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNSHADOW))

    def _initialize_event(self):
        pass
