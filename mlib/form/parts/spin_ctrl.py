import wx


class WheelSpinCtrl(wx.SpinCtrl):
    def __init__(self, *args, **kw):
        change_event = kw.pop("change_event", None)
        inc = kw.pop("inc", 1)
        super().__init__(*args, **kw)
        self.change_event = change_event
        self.inc = inc

        self.Bind(wx.EVT_SPINCTRL, self.on_wheel_spin)

    def on_wheel_spin(self, event: wx.SpinEvent):
        if self.GetValue() >= 0:
            self.SetBackgroundColour("WHITE")
        else:
            self.SetBackgroundColour("TURQUOISE")

        self.change_event(event)


class WheelSpinCtrlDouble(wx.SpinCtrlDouble):
    def __init__(self, *args, **kw):
        change_event = kw.pop("change_event", None)
        inc = kw.pop("inc", 0.1)
        super().__init__(*args, **kw)
        self.change_event = change_event
        self.inc = inc

        self.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_wheel_spin)

    def on_wheel_spin(self, event: wx.SpinEvent):
        if self.GetValue() >= 0:
            self.SetBackgroundColour("WHITE")
        else:
            self.SetBackgroundColour("TURQUOISE")

        self.change_event(event)
