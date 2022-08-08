import OpenGL.GL as gl
import OpenGL.GLU as glu
import wx
from wx import glcanvas


class BaseGLCanvas(glcanvas.GLCanvas):
    def __init__(self, parent, width, height, *args, **kw):
        glcanvas.GLCanvas.__init__(self, parent, -1, size=(width, height))
        self.init = False
        self.context = glcanvas.GLContext(self)
        self.size = wx.Size(width, height)
        self.SetCurrent(self.context)

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)

        # # 　いろいろなイベント
        # self.Bind(wx.EVT_SIZE, self.OnSize)
        # self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)
        # self.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)
        # self.Bind(wx.EVT_RIGHT_DOWN, self.OnMouseDown)
        # self.Bind(wx.EVT_RIGHT_UP, self.OnMouseUp)
        # self.Bind(wx.EVT_MIDDLE_DOWN, self.OnMouseDown)
        # self.Bind(wx.EVT_MIDDLE_UP, self.OnMouseUp)
        # self.Bind(wx.EVT_MOTION, self.OnMouseMotion)
        # # self.Bind(wx.EVT_CHAR, self.OnKeyboard)　　　#キーが押されたとき

    #        self.Bind(wx.EVT_IDLE, self.OnIdle)
    #        self.Bind(wx.EVT_MOUSEWHEEL, self.OnMouseWheel)

    def OnEraseBackground(self, event):
        pass  # Do nothing, to avoid flashing on MSW (これがないとチラつくらしい）

    def OnSize(self, event):
        size = self.size = self.GetClientSize()
        gl.glViewport(0, 0, size.width, size.height)
        event.Skip()

    def OnPaint(self, event: wx.Event):
        wx.PaintDC(self)
        if not self.init:
            self.InitGL()
            self.init = True
        self.OnDraw(event)

    def OnDraw(self, event: wx.Event):
        pass

    def OnMouseDown(self, event: wx.Event):
        self.CaptureMouse()
        self.x, self.y = self.lastx, self.lasty = event.GetPosition()

    def OnMouseUp(self, event: wx.Event):
        self.ReleaseMouse()

    def OnMouseMotion(self, event: wx.Event):
        if event.Dragging() and event.LeftIsDown():
            self.lastx, self.lasty = self.x, self.y
            self.x, self.y = event.GetPosition()
            self.Refresh(False)

    def OnMouseWheel(self, event: wx.Event):
        if event.GetWheelRotation() > 0:
            self.zooming *= 0.9
        else:
            self.zooming *= 1.1

    def InitGL(self):
        gl.glClearColor(0.7, 0.7, 0.7, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)  # enable shading

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()

        # set perspective
        glu.gluPerspective(
            45.0, float(self.size.width) / float(self.size.height), 0.10, 160.0
        )

        # modeling transform
        gl.glMatrixMode(gl.GL_MODELVIEW)

        # light color
        light_ambient = [0.25, 0.25, 0.25]
        light_diffuse = [1.0, 1.0, 1.0]
        light_specular = [1.0, 1.0, 1.0]

        # light position
        light_position = [0, 0, 2, 1]

        # light setting
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, light_ambient)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, light_diffuse)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, light_specular)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, light_position)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_LIGHTING)
