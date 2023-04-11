import wx


class FloatSliderCtrl:
    def __init__(
        self,
        parent,
        value: float,
        min_value: float,
        max_value: float,
        increment: float,
        border: int,
        position: wx.Position = wx.DefaultPosition,
        size: wx.Size = wx.DefaultSize,
        change_event=None,
    ) -> None:
        self._min = min_value
        self._max = max_value
        self._increment = increment
        self._change_event = change_event
        i_value, i_min, i_max = [round(v / increment) for v in (value, min_value, max_value)]

        self._value_ctrl = wx.TextCtrl(parent, wx.ID_ANY, str(value), wx.DefaultPosition, wx.Size(50, -1))
        self._value_ctrl.Bind(wx.EVT_TEXT, self._on_change_value)

        self._slider = wx.Slider(parent, wx.ID_ANY, i_value, i_min, i_max, position, size, wx.SL_HORIZONTAL)
        self._slider.Bind(wx.EVT_SCROLL, self._on_scroll)
        self._slider.Bind(wx.EVT_MOUSEWHEEL, self._on_wheel_spin)

        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(self._value_ctrl, 0, wx.LEFT | wx.TOP | wx.BOTTOM, border)
        self.sizer.Add(self._slider, 0, wx.TOP | wx.RIGHT | wx.BOTTOM, border)

    def _on_scroll(self, event: wx.Event):
        v = max(self._min, min(self._max, self._slider.GetValue() * self._increment))
        self._value_ctrl.ChangeValue(f"{v:.2f}")

        if self._change_event:
            self._change_event(event)

    def _on_change_value(self, event: wx.Event):
        self._slider.SetValue(round(float(self._value_ctrl.GetValue()) / self._increment))

        if self._change_event:
            self._change_event(event)

    def _on_wheel_spin(self, event: wx.MouseEvent):
        """マウスホイールによるスピンコントロール"""
        if event.GetWheelRotation() > 0:
            self._value_ctrl.SetValue(f"{float(self._value_ctrl.GetValue()) - self._increment:.2f}")
        else:
            self._value_ctrl.SetValue(f"{float(self._value_ctrl.GetValue()) + self._increment:.2f}")

    def SetValue(self, v: float):
        self._value_ctrl.ChangeValue(f"{v:.2f}")
        self._slider.SetValue(round(v / self._increment))

    def GetValue(self):
        return self._value_ctrl.GetValue()

    def Add(
        self,
        parent_sizer: wx.Sizer,
        proportion: int,
        flag: int,
        border: int,
    ):
        self.sizer.Add(parent_sizer, proportion, flag, border)
