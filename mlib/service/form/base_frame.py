import os
import sys
from threading import Thread
from typing import List

import wx

from mlib.service.form.base_notebook import BaseNotebook
from mlib.utils.file_utils import read_histories


class BaseFrame(wx.Frame):
    def __init__(self, app: wx.App, title: str, history_keys: List[str], size: wx.Size, *args, **kw):
        wx.Frame.__init__(
            self,
            None,
            title=title,
            size=size,
            style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL | wx.FULL_REPAINT_ON_RESIZE,
        )
        self.app = app

        self.history_keys = history_keys
        self.histories = read_histories(self.history_keys)

        self._initialize_ui()
        self._initialize_event()

        self.Centre(wx.BOTH)
        self.Layout()

    def _initialize_ui(self):
        self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNSHADOW))
        self.notebook = BaseNotebook(self)

    def _initialize_event(self):
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.on_change_tab)

    def on_change_tab(self, event: wx.Event):
        pass

    def on_close(self, event: wx.Event):
        self.Destroy()
        sys.exit(0)

    def on_sound(self):
        Thread(target=self.sound_finish_thread).start()

    def sound_finish_thread(self):
        """Windowsのみ終了音を鳴らす"""
        if os.name == "nt":
            try:
                from winsound import PlaySound, SND_ALIAS

                PlaySound("SystemAsterisk", SND_ALIAS)
            except Exception:
                pass
