import OpenGL.GL as gl
import OpenGL.GLU as glu
import wx
from mlib.base.viewer import BaseGLCanvas
from mlib.pmx.reader import PmxReader


class PmxCanvas(BaseGLCanvas):
    def __init__(self, parent, pmx_path: str, width: int, height: int, *args, **kw):
        super().__init__(parent, width, height, **kw)
        self.rotate = False

        self.model = PmxReader().read_by_filepath(pmx_path)
        self.model.init_draw()

    def OnDraw(self, event: wx.Event):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        # set camera
        glu.gluLookAt(0.0, 10.0, -30.0, 0.0, 10.0, 0.0, 0.0, 1.0, 0.0)

        gl.glPushMatrix()

        if self.model:
            self.model.draw()

        gl.glPopMatrix()

        gl.glFlush()  # enforce OpenGL command
        self.SwapBuffers()
