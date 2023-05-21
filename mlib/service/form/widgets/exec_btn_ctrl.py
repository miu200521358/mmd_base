import os
from threading import current_thread
from typing import Optional

import wx

from mlib.base.logger import MLogger
from mlib.service.base_worker import SimpleThread


logger = MLogger(os.path.basename(__file__))
__ = logger.get_text


class ExecButton(wx.Button):
    def __init__(self, parent, label: str, disable_label: str, exec_evt, width: int = 120, tooltip: Optional[str] = None, *args, **kw):
        self.parent = parent
        self.label = label
        self.disable_label = disable_label
        self.exec_evt = exec_evt

        super().__init__(
            parent,
            label=label,
            size=wx.Size(width, 50),
        )
        self.Bind(wx.EVT_BUTTON, self._exec)
        self.SetToolTip(tooltip)

    def _exec(self, event: wx.Event):
        if self.GetLabel() == self.disable_label:
            logger.info(__("*** 処理を停止します ***"))
            self.parent.Enable(False)

            # 実行停止
            thread: SimpleThread = current_thread()
            thread.killed = True
        else:
            # 実行開始
            self.SetLabel(self.disable_label)
            self.parent.Enable(False)

            self.exec_evt(event)

            self.parent.Enable(True)
            self.SetLabel(self.label)
