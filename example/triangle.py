import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import wx
from mlib.math import MMatrix4x4, MQuaternion
from mlib.pmx.reader import PmxReader
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_COLOR_BUFFER_BIT,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_STATIC_DRAW,
    GL_TRIANGLES,
    GL_VERTEX_SHADER,
    ctypes,
    glBindBuffer,
    glBindVertexArray,
    glBufferData,
    glClear,
    glClearColor,
    glDrawArrays,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glGetUniformLocation,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
    shaders,
)
from wx import glcanvas

vertex_shader = """
# version 330
in layout(location = 0) vec3 positions;
in layout(location = 1) vec3 colors;

out vec3 newColor;
uniform mat4 model;

void main() {
    gl_Position = model * vec4(positions, 1.0);
    newColor = colors;
}
"""

fragments_shader = """
# version 330

in vec3 newColor;
out vec4  outColor;

void main() {
    outColor = vec4(newColor, 1.0);
}
"""


class Geometries:
    def __init__(self) -> None:
        triangle = np.array(
            [
                -0.5,
                -0.5,
                0.0,
                1.0,
                0.0,
                0.0,
                0.5,
                -0.5,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.5,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )

        self.vao_triangle = glGenVertexArrays(1)
        glBindVertexArray(self.vao_triangle)

        vbo_triangle = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_triangle)
        glBufferData(GL_ARRAY_BUFFER, len(triangle) * 4, triangle, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

        # quad ------------------

        quad = np.array(
            [
                -0.5,
                -0.5,
                0.0,
                1.0,
                0.0,
                0.0,
                0.5,
                -0.5,
                0.0,
                0.0,
                1.0,
                0.0,
                -0.5,
                0.5,
                0.0,
                0.0,
                0.0,
                1.0,
                0.5,
                0.5,
                0.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )

        self.qv_count = int(len(quad) / 6)

        self.vao_quad = glGenVertexArrays(1)
        glBindVertexArray(self.vao_quad)

        vbo_quad = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_quad)
        glBufferData(GL_ARRAY_BUFFER, len(quad) * 4, quad, GL_STATIC_DRAW)
        # vertex attribute pointer
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # color attribute pointer
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

        # 柱 ------------------

        reader = PmxReader()
        model = reader.read_by_filepath(
            "..\\test\\resources\\柱.pmx",
        )
        print(model.name)

        [v.vector for v in model.vertices]

        pillar = np.array(
            [
                -0.5,
                -0.5,
                0.0,
                1.0,
                0.0,
                0.0,
                0.5,
                -0.5,
                0.0,
                0.0,
                1.0,
                0.0,
                -0.5,
                0.5,
                0.0,
                0.0,
                0.0,
                1.0,
                0.5,
                0.5,
                0.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )

        self.qv_count = int(len(pillar) / 6)

        self.vao_pillar = glGenVertexArrays(1)
        glBindVertexArray(self.vao_pillar)

        vbo_pillar = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pillar)
        glBufferData(GL_ARRAY_BUFFER, len(pillar) * 4, pillar, GL_STATIC_DRAW)
        # vertex attribute pointer
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # color attribute pointer
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

    def bind_triangle(self):
        glBindVertexArray(self.vao_triangle)

    def bind_quad(self):
        glBindVertexArray(self.vao_quad)


class OpenGLCanvas(glcanvas.GLCanvas):
    def __init__(self, parent, *args, **kw):
        glcanvas.GLCanvas.__init__(self, parent, -1, size=(1120, 630))
        self.init = False
        self.context = glcanvas.GLContext(self)
        self.SetCurrent(self.context)
        glClearColor(0.1, 0.15, 0.1, 1)
        self.rotate = False
        self.rot_y = MMatrix4x4()
        self.rot_y.identity()

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnResize)

    def OnResize(self, event: wx.Event):
        size = self.GetClientSize()
        glViewport(0, 0, size.width, size.height)

    def OnPaint(self, event: wx.Event):
        wx.PaintDC(self)
        if not self.init:
            self.InitGL()
            self.init = True

        self.OnDraw(event)

    def InitGL(self):
        self.mesh = Geometries()
        self.mesh.bind_quad()

        shader = shaders.compileProgram(
            shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
            shaders.compileShader(fragments_shader, GL_FRAGMENT_SHADER),
        )

        glClearColor(0.1, 0.15, 0.1, 1.0)

        glUseProgram(shader)

        self.model_loc = glGetUniformLocation(shader, "model")

    def OnDraw(self, event: wx.Event):
        glClear(GL_COLOR_BUFFER_BIT)
        if self.rotate:
            self.rot_y.rotate(MQuaternion.from_euler_degrees(0, 1, 0))
            glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, self.rot_y.vector)
            self.Refresh()
        else:
            glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, self.rot_y.vector)
            self.Refresh()
        glDrawArrays(GL_TRIANGLES, 0, 3)
        self.SwapBuffers()


class MyPanel(wx.Panel):
    def __init__(self, parent, *args, **kw):
        wx.Panel.__init__(self, parent)
        self.SetBackgroundColour("#626D58")
        self.canvas = OpenGLCanvas(self)
        self.rot_btn = wx.Button(
            self, -1, label="Start/Stop\nrotation", pos=(1130, 10), size=(100, 50)
        )
        self.rot_btn.BackgroundColour = (125, 125, 125)
        self.rot_btn.ForegroundColour = (0, 0, 0)

        self.Bind(wx.EVT_BUTTON, self.rotate, self.rot_btn)

    def rotate(self, event: wx.Event):
        if not self.canvas.rotate:
            self.canvas.rotate = True
            self.canvas.Refresh()
        else:
            self.canvas.rotate = False


class MyFrame(wx.Frame):
    def __init__(self, *args, **kw):
        self.size = (1280, 720)
        wx.Frame.__init__(
            self,
            None,
            title="My wx Frame",
            size=self.size,
            style=wx.DEFAULT_FRAME_STYLE | wx.FULL_REPAINT_ON_RESIZE,
        )
        self.SetMinSize(self.size)
        self.SetMaxSize(self.size)
        self.Bind(wx.EVT_CLOSE, self.onClose)
        self.panel = MyPanel(self)

    def onClose(self, event: wx.Event):
        self.Destroy()
        sys.exit(0)


class MyApp(wx.App):
    def __init__(
        self, redirect=False, filename=None, useBestVisual=False, clearSigInt=True
    ):
        super().__init__(redirect, filename, useBestVisual, clearSigInt)
        self.frame = MyFrame()
        self.frame.Show()


if __name__ == "__main__":
    app = MyApp()
    app.MainLoop()
