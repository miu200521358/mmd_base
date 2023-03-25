from typing import Optional
import OpenGL.GL as gl
from OpenGL.GL import glReadPixels, GL_RGBA, GL_UNSIGNED_BYTE
import wx
from wx import glcanvas
from PIL import Image
import os
import numpy as np
from multiprocessing import Process, Queue

from mlib.base.math import MQuaternion, MVector3D
from mlib.pmx.shader import MShader
from mlib.pmx.pmx_collection import PmxModel
from mlib.vmd.vmd_collection import VmdMotion


class PmxCanvas(glcanvas.GLCanvas):
    def __init__(self, parent, width: int, height: int, *args, **kw):
        attribList = (glcanvas.WX_GL_RGBA, glcanvas.WX_GL_DOUBLEBUFFER, glcanvas.WX_GL_DEPTH_SIZE, 16, 0)
        glcanvas.GLCanvas.__init__(self, parent, -1, size=(width, height), attribList=attribList)
        self.context = glcanvas.GLContext(self)
        self.size = wx.Size(width, height)
        self.last_pos = wx.Point(0, 0)
        self.now_pos = wx.Point(0, 0)

        self.SetCurrent(self.context)

        self.matrixes_queue: Queue = Queue()
        self.matrixes_process: Optional[Process] = None
        self.fps = 30

        # 毎フレーム呼ばれるメソッド
        self.on_update = self.get_mesh_matrixes_async

        self._initialize_ui(parent)
        self._initialize_ui_event()

        self.shader = MShader(width, height)
        self.model = PmxModel()
        self.motion = VmdMotion()
        self.bone_matrixes = np.array([np.eye(4) for _ in range(1)])

        # マウスドラッグフラグ
        self.is_drag = False

        # 再生中かどうかを示すフラグ
        self.playing = False

    def _initialize_ui(self, parent):
        gl.glClearColor(0.7, 0.7, 0.7, 1)

        self.frame_ctrl = wx.SpinCtrl(parent, value="0", min=0, max=10000, size=wx.Size(80, 30))
        self.frame_ctrl.Bind(wx.EVT_SPINCTRL, self.change_motion)

        # 再生タイマー
        self.play_timer = wx.Timer(self)

    def _initialize_ui_event(self):
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

        # タイマーイベントをバインド
        self.Bind(wx.EVT_TIMER, self.on_play_timer, self.play_timer)
        self.Bind(wx.EVT_TIMER, self.on_update, self.play_timer)

    def on_erase_background(self, event: wx.Event):
        # Do nothing, to avoid flashing on MSW (これがないとチラつくらしい）
        pass

    def on_resize(self, event: wx.Event):
        self.size = self.GetClientSize()
        self.shader.fit(self.size.width, self.size.height)
        event.Skip()

    def on_paint(self, event: wx.Event):
        self.draw(event)

        self.SwapBuffers()

    def on_reset(self, event: wx.Event):
        self.frame_ctrl.SetValue(0)
        self.shader.vertical_degrees = self.shader.INITIAL_VERTICAL_DEGREES
        self.shader.look_at_center = MVector3D(0, self.shader.INITIAL_LOOK_AT_CENTER_Y, 0)
        self.shader.camera_rotation = MQuaternion()
        self.shader.camera_position = MVector3D(
            0,
            self.shader.INITIAL_CAMERA_POSITION_Y,
            self.shader.INITIAL_CAMERA_POSITION_Z,
        )
        self.Refresh()

    def draw(self, event: wx.Event):
        self.SetCurrent(self.context)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        for is_edge in [False, True]:
            self.shader.use(is_edge)
            self.shader.update_camera(is_edge)
            self.shader.unuse()

        if self.model:
            self.shader.msaa.bind()
            self.model.draw(self.bone_matrixes)
            self.shader.msaa.unbind()

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
        elif keycode in [wx.WXK_NUMPAD2, wx.WXK_ESCAPE]:
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
        elif keycode in [wx.WXK_NUMPAD3, wx.WXK_RIGHT, wx.WXK_NUMPAD_RIGHT, wx.WXK_WINDOWS_RIGHT]:
            # キーフレを進める
            self.on_frame_forward(event)
        elif keycode in [wx.WXK_NUMPAD1, wx.WXK_LEFT, wx.WXK_NUMPAD_LEFT, wx.WXK_WINDOWS_LEFT]:
            # キーフレを戻す
            self.on_frame_back(event)
        elif keycode in [wx.WXK_NUMPAD7, wx.WXK_DOWN, wx.WXK_NUMPAD_DOWN]:
            # キャプチャ
            self.on_capture(event)
        elif keycode in [wx.WXK_SPACE]:
            # 再生/停止
            self.on_play(event)
        else:
            event.Skip()
        self.Refresh()

    def on_frame_forward(self, event: wx.Event):
        self.frame_ctrl.SetValue(self.frame_ctrl.GetValue() + 1)
        self.change_motion(event)

    def on_frame_back(self, event: wx.Event):
        self.frame_ctrl.SetValue(max(0, self.frame_ctrl.GetValue() - 1))
        self.change_motion(event)

    def change_motion(self, event: wx.Event):
        if self.motion:
            now_fno = self.frame_ctrl.GetValue()

            self.bone_matrixes = self.motion.bones.get_mesh_matrixes(now_fno, self.model)
            self.Refresh()

    def get_mesh_matrixes_async(self, event: wx.Event):
        if not self.matrixes_process or not self.matrixes_process.is_alive():
            self.matrixes_process = Process(
                target=self.get_mesh_matrixes, args=(self.matrixes_queue, self.motion, self.model, self.frame_ctrl.GetValue()), name="MeshMatrixProcess"
            )
            self.matrixes_process.start()

        if not self.matrixes_queue.empty():
            bone_matrixes = self.matrixes_queue.get()
            if bone_matrixes is not None:
                self.bone_matrixes = bone_matrixes
                self.frame_ctrl.SetValue(self.frame_ctrl.GetValue() + 1)

    @staticmethod
    def get_mesh_matrixes(matrixes_queue: Queue, motion: VmdMotion, model: PmxModel, now_fno: int):
        while motion and motion.max_fno > now_fno:
            now_fno += 1
            bone_matrixes = motion.bones.get_mesh_matrixes(now_fno, model)
            matrixes_queue.put(bone_matrixes)
        matrixes_queue.put(None)

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

    def on_play(self, event: wx.Event):
        self.playing = not self.playing  # フラグを反転
        if not self.playing:
            self.play_timer.Stop()  # タイマーを停止
            if self.matrixes_process:
                self.matrixes_process.terminate()
        else:
            self.play_timer.Start(int(1000 / self.fps))  # タイマーを再開

    def on_play_timer(self, event: wx.Event):
        if self.playing and self.model and self.motion:
            self.frame_ctrl.SetValue(self.frame_ctrl.GetValue() + 1)
            # self.on_frame_forward(event)

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
            self.Refresh()

    def on_mouse_wheel(self, event: wx.Event):
        if event.GetWheelRotation() < 0:
            self.shader.vertical_degrees += 1.0
        else:
            self.shader.vertical_degrees = max(1.0, self.shader.vertical_degrees - 1.0)
        self.Refresh()
