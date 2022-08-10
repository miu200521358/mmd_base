import OpenGL.GL as gl
import OpenGL.GLU as glu
import wx
from mlib.pmx.reader import PmxReader
from mlib.pmx.shader import MShader
from wx import glcanvas


class PmxCanvas(glcanvas.GLCanvas):
    def __init__(self, parent, pmx_path: str, width: int, height: int, *args, **kw):
        glcanvas.GLCanvas.__init__(self, parent, -1, size=(width, height))
        self.context = glcanvas.GLContext(self)
        self.size = wx.Size(width, height)
        self.SetCurrent(self.context)
        gl.glClearColor(0.1, 0.15, 0.1, 1)

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)

        self.shader = MShader(width, height)
        self.model = PmxReader().read_by_filepath(pmx_path)
        self.model.init_draw(self.shader)

    def OnEraseBackground(self, event):
        pass  # Do nothing, to avoid flashing on MSW (これがないとチラつくらしい）

    def OnSize(self, event):
        size = self.size = self.GetClientSize()
        gl.glViewport(0, 0, size.width, size.height)
        event.Skip()

    def OnPaint(self, event: wx.Event):
        wx.PaintDC(self)
        self.OnDraw(event)

    def OnDraw(self, event: wx.Event):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        # set camera
        glu.gluLookAt(0.0, 10.0, -30.0, 0.0, 10.0, 0.0, 0.0, 1.0, 0.0)

        gl.glPushMatrix()

        if self.model:
            self.model.update()
            self.model.draw()

        gl.glPopMatrix()

        gl.glFlush()  # enforce OpenGL command
        self.SwapBuffers()

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
