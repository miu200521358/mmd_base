import OpenGL.GL as gl
import wx
from mlib.math import MQuaternion, MVector3D
from mlib.pmx.reader import PmxReader
from mlib.pmx.shader import MShader
from wx import glcanvas


class PmxCanvas(glcanvas.GLCanvas):
    def __init__(self, parent, pmx_path: str, width: int, height: int, *args, **kw):
        glcanvas.GLCanvas.__init__(self, parent, -1, size=(width, height))
        self.context = glcanvas.GLContext(self)
        self.size = wx.Size(width, height)
        self.last_pos = wx.Point(0, 0)
        self.now_pos = wx.Point(0, 0)
        self.SetCurrent(self.context)
        gl.glClearColor(0.4, 0.4, 0.4, 1)

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnResize)
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnMouseWheel)
        self.Bind(wx.EVT_MIDDLE_DOWN, self.OnMouseDown)
        self.Bind(wx.EVT_RIGHT_DOWN, self.OnMouseDown)
        self.Bind(wx.EVT_MOTION, self.OnMouseMotion)
        self.Bind(wx.EVT_MIDDLE_UP, self.OnMouseUp)
        self.Bind(wx.EVT_RIGHT_UP, self.OnMouseUp)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)

        self.shader = MShader(width, height)

        self.model = PmxReader().read_by_filepath(pmx_path)
        self.model.init_draw(self.shader)

        self.is_drag = False

    def OnEraseBackground(self, event):
        pass  # Do nothing, to avoid flashing on MSW (これがないとチラつくらしい）

    def OnResize(self, event):
        self.size: wx.Size = self.GetClientSize()
        self.shader.fit(self.size.width, self.size.height)
        event.Skip()

    def OnPaint(self, event: wx.Event):
        wx.PaintDC(self)
        self.OnDraw(event)

    def reset(self):
        self.shader.look_at_center = MVector3D()
        self.shader.camera_position = MVector3D(0, 0, -3)
        self.shader.camera_rotation = MQuaternion()

    def OnDraw(self, event: wx.Event):
        if self.model:
            self.model.update()

        for is_edge in [False, True]:
            self.shader.use(is_edge)
            self.shader.update_camera(is_edge)
            self.shader.unuse()

        if self.model:
            self.model.draw()

        gl.glFlush()
        self.SwapBuffers()
        self.Refresh(False)

    def OnKeyDown(self, event: wx.Event):
        keycode = event.GetKeyCode()
        if keycode == wx.WXK_NUMPAD0:
            # 真下から
            self.shader.look_at_center = MVector3D()
            self.shader.camera_rotation = MQuaternion()
            self.shader.camera_position = MVector3D(0, -3, -0.1)
        elif keycode == wx.WXK_NUMPAD2:
            # 真正面から
            self.shader.look_at_center = MVector3D()
            self.shader.camera_rotation = MQuaternion()
            self.shader.camera_position = MVector3D(0, 0, -3)
        elif keycode == wx.WXK_NUMPAD4:
            # 左から
            self.shader.look_at_center = MVector3D()
            self.shader.camera_rotation = MQuaternion()
            self.shader.camera_position = MVector3D(3, 0, -0.1)
        elif keycode == wx.WXK_NUMPAD6:
            # 右から
            self.shader.look_at_center = MVector3D()
            self.shader.camera_rotation = MQuaternion()
            self.shader.camera_position = MVector3D(-3, 0, -0.1)
        elif keycode == wx.WXK_NUMPAD8:
            # 真後ろから
            self.shader.look_at_center = MVector3D()
            self.shader.camera_rotation = MQuaternion()
            self.shader.camera_position = MVector3D(0, 0, 3)
        elif keycode == wx.WXK_NUMPAD5:
            # 真上から
            self.shader.look_at_center = MVector3D()
            self.shader.camera_rotation = MQuaternion()
            self.shader.camera_position = MVector3D(0, 3, -0.1)

    def OnMouseDown(self, event: wx.Event):
        self.now_pos = self.last_pos = event.GetPosition()
        self.is_drag = True
        self.CaptureMouse()

    def OnMouseUp(self, event: wx.Event):
        self.is_drag = False
        self.ReleaseMouse()

    def OnMouseMotion(self, event: wx.Event):
        if self.is_drag and event.Dragging():
            self.now_pos = event.GetPosition()
            x = (self.now_pos.x - self.last_pos.x) * 0.1
            y = (self.now_pos.y - self.last_pos.y) * 0.1
            if event.MiddleIsDown():
                self.shader.look_at_center.x += x * 0.01
                self.shader.look_at_center.y += y * 0.01

                self.shader.camera_position.x += x * 0.01
                self.shader.camera_position.y += y * 0.01
            elif event.RightIsDown():
                self.shader.camera_rotation *= MQuaternion.from_euler_degrees(y, -x, 0)
            self.last_pos = self.now_pos

    def OnMouseWheel(self, event: wx.Event):
        if event.GetWheelRotation() > 0:
            self.shader.camera_position.z += 0.1
        else:
            self.shader.camera_position.z -= 0.1
