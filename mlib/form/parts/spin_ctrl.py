import wx


class WheelSpinCtrl(wx.SpinCtrl):
    def __init__(self, *args, **kw):
        change_event = kw.pop("change_event", None)
        super().__init__(*args, **kw)
        self.change_event = change_event

        self.Bind(wx.EVT_SPINCTRL, self.on_spin)

    def on_spin(self, event: wx.Event):
        if self.GetValue() >= 0:
            self.SetBackgroundColour("WHITE")
        else:
            self.SetBackgroundColour("TURQUOISE")

        self.change_event(event)


class WheelSpinCtrlDouble(wx.SpinCtrlDouble):
    def __init__(self, *args, **kw):
        change_event = kw.pop("change_event", None)
        super().__init__(*args, **kw)
        self.change_event = change_event

        self.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_spin)
        self.Bind(wx.EVT_MOUSEWHEEL, self.on_wheel_spin)

    def on_wheel_spin(self, event: wx.MouseEvent):
        """マウスホイールによるスピンコントロール"""
        if event.GetWheelRotation() > 0:
            self.SetValue(self.GetValue() + self.GetIncrement())
        else:
            self.SetValue(self.GetValue() - self.GetIncrement())
        self.on_spin(event)

    def on_spin(self, event: wx.Event):
        if self.GetValue() >= 0:
            self.SetBackgroundColour("WHITE")
        else:
            self.SetBackgroundColour("TURQUOISE")

        self.change_event(event)
