import os
from typing import Callable, Optional

import wx

from mlib.core.logger import MLogger
from mlib.service.base_worker import BaseWorker
from mlib.service.form.base_panel import BasePanel

logger = MLogger(os.path.basename(__file__))
__ = logger.get_text


class ExecButton(wx.Button):
    def __init__(
        self,
        panel: BasePanel,
        label: str,
        disable_label: str,
        exec_evt: Callable,
        width: int = 120,
        tooltip: Optional[str] = None,
    ):
        self.parent = panel
        self.label = label
        self.disable_label = disable_label
        self.exec_evt = exec_evt
        self.exec_worker: Optional[BaseWorker] = None

        super().__init__(
            panel,
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
            if self.exec_worker:
                self.exec_worker.killed = True

            self.parent.Enable(True)
            self.SetLabel(self.label)
        else:
            # 実行開始
            self.SetLabel(self.disable_label)
            self.parent.Enable(False)

            self.exec_evt(event)

            self.parent.Enable(True)
            self.SetLabel(self.label)
