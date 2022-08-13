import OpenGL.GL as gl
import wx
from mlib.math import MQuaternion
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
        self.Bind(wx.EVT_SIZE, self.OnResize)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)

        self.shader = MShader(width, height)

        self.model = PmxReader().read_by_filepath(pmx_path)
        self.model.init_draw(self.shader)

        self.rotate = False

    def OnEraseBackground(self, event):
        pass  # Do nothing, to avoid flashing on MSW (これがないとチラつくらしい）

    def OnResize(self, event):
        self.size: wx.Size = self.GetClientSize()
        self.shader.fit(self.size.width, self.size.height)
        event.Skip()

    def OnPaint(self, event: wx.Event):
        wx.PaintDC(self)
        self.OnDraw(event)

    def OnDraw(self, event: wx.Event):
        # # set camera
        # glu.gluLookAt(0.0, 10.0, -30.0, 0.0, 10.0, 0.0, 0.0, 1.0, 0.0)

        if self.rotate:
            # gl.glMatrixMode(gl.GL_MODELVIEW)
            # gl.glLoadIdentity()
            # gl.glRotatef(10, 0, 1, 0)
            self.shader.camera_rotation *= MQuaternion.from_euler_degrees(0, 1, 0)

        # gl.glUniformMatrix4fv(
        #     self.shader.bone_matrix_uniform, 1, gl.GL_FALSE, self.rot_mat.vector
        # )

        if self.model:
            self.model.update()

        self.shader.use()
        self.shader.update_camera()

        if self.model:
            self.model.draw()

        self.shader.unuse()

        gl.glFlush()
        self.SwapBuffers()
        self.Refresh()

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
