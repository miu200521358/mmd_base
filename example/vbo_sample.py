# coding: utf-8
# 20190825 rewrite
import sys
import array
import ctypes
import struct
from typing import Any
from OpenGL.GL import *
from OpenGL.GLUT import *

VS = """
#version 330
in vec2 aPosition;
void main ()
{
    gl_Position = vec4(aPosition, 0.5, 1);
}
"""

FS = """
#version 330
out vec4 FragColor;
void main()
{
    FragColor = vec4(1, 1, 1, 1);
}
"""


def load_shader(src: str, shader_type: int) -> int:
    shader = glCreateShader(shader_type)
    glShaderSource(shader, src)
    glCompileShader(shader)
    error = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if error != GL_TRUE:
        info = glGetShaderInfoLog(shader)
        glDeleteShader(shader)
        raise Exception(info)
    return shader


class Shader:
    def __init__(self) -> None:
        self.program = glCreateProgram()

    def __del__(self) -> None:
        glDeleteProgram(self.program)

    def compile(self, vs_src: str, fs_src: str) -> None:
        vs = load_shader(vs_src, GL_VERTEX_SHADER)
        if not vs:
            return
        fs = load_shader(fs_src, GL_FRAGMENT_SHADER)
        if not fs:
            return
        glAttachShader(self.program, vs)
        glAttachShader(self.program, fs)
        glLinkProgram(self.program)
        error = glGetProgramiv(self.program, GL_LINK_STATUS)
        glDeleteShader(vs)
        glDeleteShader(fs)
        if error != GL_TRUE:
            info = glGetShaderInfoLog(self.program)
            raise Exception(info)

    def use(self):
        glUseProgram(self.program)

    def unuse(self):
        glUseProgram(0)


class VBO:
    def __init__(self) -> None:
        self.vbo = glGenBuffers(1)
        self.component_count = 0  # Vec2, Vec3, Vec4 などの2, 3, 4
        self.vertex_count = 0

    def __del__(self) -> None:
        glDeleteBuffers(1, [self.vbo])

    def bind(self) -> None:
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

    def unbind(self) -> None:
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def set_vertex_attribute(
        self, component_count: int, bytelength: int, data: any
    ) -> None:
        """float2, 3, 4"""
        self.component_count = component_count
        stride = 4 * self.component_count
        self.vertex_count = bytelength // stride
        self.bind()
        glBufferData(GL_ARRAY_BUFFER, bytelength, data, GL_STATIC_DRAW)

    def set_slot(self, slot: int) -> None:
        self.bind()
        glEnableVertexAttribArray(slot)
        glVertexAttribPointer(slot, self.component_count, GL_FLOAT, GL_FALSE, 0, None)

    def draw(self) -> None:
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)


class IBO:
    def __init__(self) -> None:
        self.vbo = glGenBuffers(1)
        self.index_count = 0
        self.index_type = 0

    def __del__(self) -> None:
        glDeleteBuffers(1, [self.vbo])

    def bind(self) -> None:
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo)

    def unbind(self) -> None:
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def set_indices(self, stride: int, bytelength: int, data: Any) -> None:
        self.index_count = bytelength // stride
        self.bind()
        if stride == 1:
            self.index_type = GL_UNSIGNED_BYT
        elif stride == 2:
            self.index_type = GL_UNSIGNED_SHORT
        elif stride == 4:
            self.index_type = GL_UNSIGNED_INT
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, bytelength, data, GL_STATIC_DRAW)

    def draw(self) -> None:
        glDrawElements(GL_TRIANGLES, self.index_count, self.index_type, None)


class Triangle:
    def __init__(self) -> None:
        self.vbo: VBO = None
        self.ibo: IBO = None
        self.shader: Shader = None
        self.positions = (-1.0, -1.0, 1.0, -1.0, 0.0, 1.0)
        self.indices = (0, 1, 2)

    def initialize(self):
        self.shader = Shader()
        self.shader.compile(VS, FS)
        self.vbo = VBO()
        self.ibo = IBO()
        if False:
            # Error
            self.vbo.set_vertex_attribute(
                2, 4 * 2 * 3, array.array("f", self.positions)
            )
            self.ibo.set_indices(4, 12, array.array("I", self.indices))
        elif True:
            # OK
            self.vbo.set_vertex_attribute(
                2, 4 * 2 * 3, (ctypes.c_float * 6)(*self.positions)
            )
            self.ibo.set_indices(4, 12, (ctypes.c_uint * 3)(*self.indices))
        elif False:
            # not work
            self.vbo.set_vertex_attribute(
                2, 4 * 2 * 3, memoryview(struct.pack("6f", *self.positions))
            )
            self.ibo.set_indices(4, 12, memoryview(struct.pack("3I", *self.indices)))

        else:
            # OK
            self.vbo.set_vertex_attribute(
                2, 4 * 2 * 3, struct.pack("6f", *self.positions)
            )
            self.ibo.set_indices(4, 12, struct.pack("3I", *self.indices))

    def draw(self) -> None:
        if not self.vbo:
            self.initialize()
        self.shader.use()
        self.vbo.set_slot(0)
        self.ibo.bind()
        self.ibo.draw()

        self.ibo.unbind()
        self.vbo.unbind()
        self.shader.unuse()


def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    # glutInitContextVersion(3, 3) # this call cause an error when glVertexAttribPointer
    glutInitContextFlags(GLUT_CORE_PROFILE | GLUT_DEBUG)
    glutInitWindowSize(256, 256)
    glutCreateWindow(b"vbo")

    print(glGetString(GL_VERSION))
    print(f"VERSION: {glGetInteger(GL_MAJOR_VERSION)}.{glGetInteger(GL_MINOR_VERSION)}")

    triangle = Triangle()

    def disp_func():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        triangle.draw()
        glFlush()
        glutSwapBuffers()

    glutDisplayFunc(disp_func)

    # glutIdleFunc(disp_func)
    def reshape_func(w, h):
        glViewport(0, 0, w, h)

    glutReshapeFunc(reshape_func)

    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)

    glutMainLoop()


if __name__ == "__main__":
    main()
