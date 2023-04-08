from typing import Callable
import wx


class WheelSpinCtrl(wx.SpinCtrl):
    def __init__(self, change_event: Callable, *args, **kw):
        super().__init__(*args, **kw)
        self.Bind(wx.EVT_SPINCTRL, change_event)
        self.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_spin_ctrl(event, 1))

    def on_wheel_spin_ctrl(self, event: wx.Event, inc=0.1):
        # スピンコントロール変更時
        if event.GetWheelRotation() > 0:
            event.GetEventObject().SetValue(event.GetEventObject().GetValue() + inc)
            if event.GetEventObject().GetValue() >= 0:
                event.GetEventObject().SetBackgroundColour("WHITE")
        else:
            event.GetEventObject().SetValue(event.GetEventObject().GetValue() - inc)
            if event.GetEventObject().GetValue() < 0:
                event.GetEventObject().SetBackgroundColour("TURQUOISE")


class WheelSpinCtrlDouble(wx.SpinCtrlDouble):
    def __init__(self, change_event: Callable, *args, **kw):
        super().__init__(*args, **kw)
        self.Bind(wx.EVT_SPINCTRL, change_event)
        self.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_spin_ctrl(event, 0.1))

    def on_wheel_spin_ctrl(self, event: wx.Event, inc=0.1):
        # スピンコントロール変更時
        if event.GetWheelRotation() > 0:
            event.GetEventObject().SetValue(event.GetEventObject().GetValue() + inc)
            if event.GetEventObject().GetValue() >= 0:
                event.GetEventObject().SetBackgroundColour("WHITE")
        else:
            event.GetEventObject().SetValue(event.GetEventObject().GetValue() - inc)
            if event.GetEventObject().GetValue() < 0:
                event.GetEventObject().SetBackgroundColour("TURQUOISE")
