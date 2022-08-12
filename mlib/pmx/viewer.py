import OpenGL.GL as gl
import OpenGL.GLU as glu
import wx
from mlib.math import MMatrix4x4, MQuaternion
from mlib.pmx.reader import PmxReader
from mlib.pmx.shader import MShader
from wx import glcanvas


class PmxCanvas(glcanvas.GLCanvas):
    def __init__(self, parent, pmx_path: str, width: int, height: int, *args, **kw):
        glcanvas.GLCanvas.__init__(self, parent, -1, size=(width, height))
        self.context = glcanvas.GLContext(self)
        self.size = wx.Size(width, height)
        self.SetCurrent(self.context)
        gl.glClearColor(0.4, 0.4, 0.4, 1)

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)

        self.shader = MShader(width, height)
        self.model = PmxReader().read_by_filepath(pmx_path)
        self.model.init_draw(self.shader)

        self.rotate = False
        self.rot_mat = MMatrix4x4(identity=True)

    def OnEraseBackground(self, event):
        pass  # Do nothing, to avoid flashing on MSW (これがないとチラつくらしい）

    def OnSize(self, event):
        self.size: wx.Size = self.GetClientSize()
        gl.glViewport(0, 0, self.size.width, self.size.height)
        event.Skip()

    def OnPaint(self, event: wx.Event):
        wx.PaintDC(self)
        self.OnDraw(event)

    def OnDraw(self, event: wx.Event):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        gl.glClearColor(0.4, 0.4, 0.4, 1)

        # set camera
        glu.gluLookAt(0.0, 10.0, -30.0, 0.0, 10.0, 0.0, 0.0, 1.0, 0.0)

        gl.glPushMatrix()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        if self.rotate:
            self.rot_mat.rotate(MQuaternion.from_euler_degrees(0, 1, 0))

        self.shader.use()

        gl.glUniformMatrix4fv(
            self.shader.bone_matrix_uniform, 1, gl.GL_FALSE, self.rot_mat.vector
        )
        self.Refresh()

        if self.model:
            self.model.update()

        self.shader.unuse()
        gl.glPopMatrix()

        if self.model:
            self.model.draw()

        gl.glFlush()  # enforce OpenGL command
        self.SwapBuffers()

    # def OnMouseDown(self, event: wx.Event):
    #     self.CaptureMouse()
    #     self.x, self.y = self.lastx, self.lasty = event.GetPosition()

    # def OnMouseUp(self, event: wx.Event):
    #     self.ReleaseMouse()

    # def OnMouseMotion(self, event: wx.Event):
    #     if event.Dragging() and event.LeftIsDown():
    #         self.lastx, self.lasty = self.x, self.y
    #         self.x, self.y = event.GetPosition()
    #         self.Refresh(False)

    # def OnMouseWheel(self, event: wx.Event):
    #     if event.GetWheelRotation() > 0:
    #         self.zooming *= 0.9
    #     else:
    #         self.zooming *= 1.1
