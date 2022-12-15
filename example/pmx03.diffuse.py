"""
材質色の描画
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import OpenGL.GL as gl
import OpenGL.GLU as glu
import wx
from mlib.math import MMatrix4x4, MQuaternion, MVector3D
from mlib.pmx.pmx_reader import PmxReader
from OpenGL.GL import shaders
from wx import glcanvas

POSITION_ID = 0
COLOR_ID = 1

vertex_shader = """
# version 330
in layout(location = %d) vec3 positions;
in layout(location = %d) vec3 colors;

out vec3 newColor;
uniform mat4 modelMatrix;

void main() {
    vec4 pvec = modelMatrix * vec4(positions, 1.0);
    gl_Position = vec4( -pvec[ 0 ], pvec[ 1 ], pvec[ 2 ], pvec[ 3 ] );  // 座標系による反転を行う、カリングも反転
    newColor = colors;
}
""" % (
    POSITION_ID,
    COLOR_ID,
)

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
        self.model = PmxReader().read_by_filepath(
            # "C:/MMD/mmd_base/test/resources/曲げ柱tex.pmx"
            # "D:/MMD/MikuMikuDance_v926x64/UserFile/Model/VOCALOID/初音ミク/Tda式初音ミク・アペンドVer1.10/Tda式初音ミク・アペンド_Ver1.10.pmx",
            "D:/MMD/MikuMikuDance_v926x64/UserFile/Model/初音ミクVer2 準標準.pmx"
        )
        self.model.init_draw()
        if not self.model.meshs:
            return
        vertex_position_list = []
        color_list = []
        prev_face_count = 0
        for material in self.model.materials:
            face_count = material.vertices_count // 3
            for face_index in range(prev_face_count, prev_face_count + face_count):
                vertex_position_list.append(
                    np.array(
                        [
                            (
                                (
                                    self.model.vertices[vidx].position
                                    + MVector3D(0, -10, 0)
                                )
                                / 15
                            ).vector
                            for vidx in self.model.faces[face_index].vertices
                        ],
                        dtype=np.float32,
                    )
                )
                color_list.append(
                    np.array(
                        np.array(
                            [
                                [
                                    material.diffuse_color.x,
                                    material.diffuse_color.y,
                                    material.diffuse_color.z,
                                ]
                                for _ in range(
                                    len(self.model.faces[face_index].vertices)
                                )
                            ],
                            dtype=np.float32,
                        ),
                        dtype=np.float32,
                    )
                )
            prev_face_count += face_count
        self.vertices = np.array(vertex_position_list, dtype=np.float32)
        self.colors = np.array(color_list, dtype=np.float32)

        # ---------------------
        # 頂点の設定

        self.vao_vertices = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao_vertices)

        vbo_vertices = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_vertices)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            self.vertices.nbytes,
            self.vertices,
            gl.GL_STATIC_DRAW,
        )

        gl.glEnableVertexAttribArray(POSITION_ID)
        gl.glVertexAttribPointer(
            POSITION_ID, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, gl.ctypes.c_void_p(0)
        )

        # ---------------------
        # 色の描画

        vbo_colors = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_colors)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            self.colors.nbytes,
            self.colors,
            gl.GL_STATIC_DRAW,
        )

        gl.glEnableVertexAttribArray(COLOR_ID)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_colors)
        gl.glVertexAttribPointer(
            COLOR_ID, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, gl.ctypes.c_void_p(0)
        )


class OpenGLCanvas(glcanvas.GLCanvas):
    def __init__(self, parent, *args, **kw):
        self.size = wx.Size(600, 600)
        glcanvas.GLCanvas.__init__(self, parent, -1, size=self.size)
        self.init = False
        self.context = glcanvas.GLContext(self)
        self.SetCurrent(self.context)
        gl.glClearColor(0.1, 0.15, 0.1, 1)
        self.rotate = False
        self.rot_y = MMatrix4x4()
        self.rot_y.identity()

        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def OnPaint(self, event: wx.Event):
        wx.PaintDC(self)
        if not self.init:
            self.InitGL()
            self.init = True

        self.OnDraw(event)

    def InitGL(self):
        self.mesh = Geometries()
        gl.glBindVertexArray(self.mesh.vao_vertices)

        gl.glClearColor(0.1, 0.15, 0.1, 1.0)
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

        # gl.glClearColor(0.1, 0.15, 0.1, 1.0)
        # # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

        shader = shaders.compileProgram(
            shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER),
            shaders.compileShader(fragments_shader, gl.GL_FRAGMENT_SHADER),
        )

        gl.glUseProgram(shader)

        self.bone_matrix = gl.glGetUniformLocation(shader, "modelMatrix")

    def OnDraw(self, event: wx.Event):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        # set camera
        # gluLookAt(0.0, 10.0, 80.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        glu.gluLookAt(0.0, 10.0, -30.0, 0.0, 10.0, 0.0, 0.0, 1.0, 0.0)

        """  /**** **** **** **** **** **** **** ****/  """
        gl.glPushMatrix()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        if self.rotate:
            self.rot_y.rotate(MQuaternion.from_euler_degrees(0, 1, 0))
            gl.glUniformMatrix4fv(self.bone_matrix, 1, gl.GL_FALSE, self.rot_y.vector)
            self.Refresh()
        else:
            gl.glUniformMatrix4fv(self.bone_matrix, 1, gl.GL_FALSE, self.rot_y.vector)
            self.Refresh()

        gl.glPopMatrix()
        """  /**** **** **** **** **** **** **** ****/  """

        gl.glDrawArrays(
            gl.GL_TRIANGLES,
            0,
            self.mesh.vertices.shape[0] * self.mesh.vertices.shape[1],
        )
        self.SwapBuffers()


class MyPanel(wx.Panel):
    def __init__(self, parent, *args, **kw):
        wx.Panel.__init__(self, parent)
        self.SetBackgroundColour("#626D58")
        self.canvas = OpenGLCanvas(self)
        self.rot_btn = wx.Button(
            self, -1, label="Start/Stop\nrotation", pos=(620, 10), size=(100, 50)
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
        self.size = (800, 600)
        wx.Frame.__init__(
            self,
            None,
            title="My wx Frame",
            size=self.size,
            style=wx.DEFAULT_FRAME_STYLE | wx.FULL_REPAINT_ON_RESIZE,
        )
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
