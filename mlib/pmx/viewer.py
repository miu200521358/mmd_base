import OpenGL.GL as gl
from OpenGL.GL import glReadPixels, GL_RGBA, GL_UNSIGNED_BYTE
import wx
from wx import glcanvas
from PIL import Image
import os
import numpy as np

from mlib.base.math import MQuaternion, MVector3D
from mlib.pmx.pmx_reader import PmxReader
from mlib.pmx.shader import MShader
from mlib.vmd.vmd_reader import VmdReader
from mlib.vmd.vmd_collection import VmdMotion


class PmxCanvas(glcanvas.GLCanvas):
    def __init__(self, parent, pmx_path: str, vmd_path: str, width: int, height: int, *args, **kw):
        glcanvas.GLCanvas.__init__(self, parent, -1, size=(width, height))
        self.context = glcanvas.GLContext(self)
        self.size = wx.Size(width, height)
        self.last_pos = wx.Point(0, 0)
        self.now_pos = wx.Point(0, 0)
        self.SetCurrent(self.context)
        gl.glClearColor(0.7, 0.7, 0.7, 1)

        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_resize)
        self.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel)
        self.Bind(wx.EVT_MIDDLE_DOWN, self.on_mouse_down)
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_mouse_down)
        self.Bind(wx.EVT_MOTION, self.on_mouse_motion)
        self.Bind(wx.EVT_MIDDLE_UP, self.on_mouse_up)
        self.Bind(wx.EVT_RIGHT_UP, self.on_mouse_up)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)
        self.Bind(wx.EVT_KEY_DOWN, self.on_key_down)

        self.model = PmxReader().read_by_filepath(pmx_path)

        self.shader = MShader(width, height, len(self.model.bones))
        self.model.init_draw(self.shader)

        self.motion = VmdReader().read_by_filepath(vmd_path) if vmd_path else VmdMotion()
        self.bone_matrixes = [np.eye(4) for _ in range(len(self.model.bones))]
        self.frame = 0

        if self.motion:
            self.bone_matrixes = self.motion.bones.get_mesh_matrixes(self.frame, self.model)

        self.is_drag = False

    def on_erase_background(self, event: wx.Event):
        # Do nothing, to avoid flashing on MSW (これがないとチラつくらしい）
        pass

    def on_resize(self, event: wx.Event):
        self.size = self.GetClientSize()
        self.shader.fit(self.size.width, self.size.height)
        event.Skip()

    def on_paint(self, event: wx.Event):
        wx.PaintDC(self)
        self.on_draw(event)

    def reset(self):
        pass

    def on_draw(self, event: wx.Event):
        if self.model:
            self.model.update()

        for is_edge in [False, True]:
            self.shader.use(is_edge)
            self.shader.update_camera(is_edge)
            self.shader.unuse()

        if self.model:
            self.model.draw(self.bone_matrixes)

        self.SwapBuffers()
        self.Refresh(False)

    def on_key_down(self, event: wx.Event):
        keycode = event.GetKeyCode()
        if keycode == wx.WXK_NUMPAD0:
            # 真下から
            self.shader.vertical_degrees = self.shader.INITIAL_VERTICAL_DEGREES
            self.shader.look_at_center = MVector3D(0, self.shader.INITIAL_LOOK_AT_CENTER_Y, 0)
            self.shader.camera_rotation = MQuaternion()
            self.shader.camera_position = MVector3D(
                0,
                -self.shader.INITIAL_CAMERA_POSITION_Y * 2,
                -0.1,
            )
        elif keycode == wx.WXK_NUMPAD2:
            # 真正面から(=リセット)
            self.shader.vertical_degrees = self.shader.INITIAL_VERTICAL_DEGREES
            self.shader.look_at_center = MVector3D(0, self.shader.INITIAL_LOOK_AT_CENTER_Y, 0)
            self.shader.camera_rotation = MQuaternion()
            self.shader.camera_position = MVector3D(
                0,
                self.shader.INITIAL_CAMERA_POSITION_Y,
                self.shader.INITIAL_CAMERA_POSITION_Z,
            )
        elif keycode == wx.WXK_NUMPAD4:
            # 左から
            self.shader.vertical_degrees = self.shader.INITIAL_VERTICAL_DEGREES
            self.shader.look_at_center = MVector3D(0, self.shader.INITIAL_LOOK_AT_CENTER_Y, 0)
            self.shader.camera_rotation = MQuaternion()
            self.shader.camera_position = MVector3D(
                self.shader.INITIAL_CAMERA_POSITION_X,
                self.shader.INITIAL_CAMERA_POSITION_Y,
                0.1,
            )
        elif keycode == wx.WXK_NUMPAD6:
            # 右から
            self.shader.vertical_degrees = self.shader.INITIAL_VERTICAL_DEGREES
            self.shader.look_at_center = MVector3D(0, self.shader.INITIAL_LOOK_AT_CENTER_Y, 0)
            self.shader.camera_rotation = MQuaternion()
            self.shader.camera_position = MVector3D(
                -self.shader.INITIAL_CAMERA_POSITION_X,
                self.shader.INITIAL_CAMERA_POSITION_Y,
                0.1,
            )
        elif keycode == wx.WXK_NUMPAD8:
            # 真後ろから
            self.shader.vertical_degrees = self.shader.INITIAL_VERTICAL_DEGREES
            self.shader.look_at_center = MVector3D(0, self.shader.INITIAL_LOOK_AT_CENTER_Y, 0)
            self.shader.camera_rotation = MQuaternion()
            self.shader.camera_position = MVector3D(
                0,
                self.shader.INITIAL_CAMERA_POSITION_Y,
                -self.shader.INITIAL_CAMERA_POSITION_Z,
            )
        elif keycode == wx.WXK_NUMPAD5:
            # 真上から
            self.shader.vertical_degrees = self.shader.INITIAL_VERTICAL_DEGREES
            self.shader.look_at_center = MVector3D(0, self.shader.INITIAL_LOOK_AT_CENTER_Y, 0)
            self.shader.camera_rotation = MQuaternion()
            self.shader.camera_position = MVector3D(
                0,
                self.shader.INITIAL_CAMERA_POSITION_Y * 3,
                -0.1,
            )
        elif keycode == wx.WXK_NUMPAD3:
            # キーフレを進める
            self.on_frame_forward(event)
        elif keycode == wx.WXK_NUMPAD1:
            # キーフレを戻す
            self.on_frame_back(event)
        elif keycode == wx.WXK_NUMPAD7:
            # キャプチャ
            self.on_capture(event)

    def on_frame_forward(self, event: wx.Event):
        self.frame += 1
        if self.motion:
            self.bone_matrixes = self.motion.bones.get_mesh_matrixes(self.frame, self.model)

    def on_frame_back(self, event: wx.Event):
        self.frame = max(0, self.frame - 1)
        if self.motion:
            self.bone_matrixes = self.motion.bones.get_mesh_matrixes(self.frame, self.model)

    def on_capture(self, event: wx.Event):
        # キャプチャ
        # OpenGLの描画バッファを読み込む
        buffer = glReadPixels(0, 0, self.size.width, self.size.height, GL_RGBA, GL_UNSIGNED_BYTE)

        # バッファをPILのImageオブジェクトに変換する
        image = Image.frombytes("RGBA", (self.size.width, self.size.height), buffer)

        # 画像を反転させる
        image = image.transpose(method=Image.FLIP_TOP_BOTTOM)

        # ImageをPNGファイルとして保存する
        file_path = os.path.join(os.path.dirname(self.model.path), "capture.png")
        image.save(file_path)

    def on_mouse_down(self, event: wx.Event):
        self.now_pos = self.last_pos = event.GetPosition()
        self.is_drag = True
        self.CaptureMouse()

    def on_mouse_up(self, event: wx.Event):
        self.is_drag = False
        self.ReleaseMouse()

    def on_mouse_motion(self, event: wx.Event):
        if self.is_drag and event.Dragging():
            self.now_pos = event.GetPosition()
            x = (self.now_pos.x - self.last_pos.x) * 0.02
            y = (self.now_pos.y - self.last_pos.y) * 0.02
            if event.MiddleIsDown():
                self.shader.look_at_center.x += x
                self.shader.look_at_center.y += y

                self.shader.camera_position.x += x
                self.shader.camera_position.y += y
            elif event.RightIsDown():
                self.shader.camera_rotation *= MQuaternion.from_euler_degrees(y * 10, -x * 10, 0)
            self.last_pos = self.now_pos

    def on_mouse_wheel(self, event: wx.Event):
        if event.GetWheelRotation() < 0:
            self.shader.vertical_degrees += 1.0
        else:
            self.shader.vertical_degrees = max(1.0, self.shader.vertical_degrees - 1.0)
