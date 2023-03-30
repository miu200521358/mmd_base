import os
from multiprocessing import Process, Queue
from typing import Optional

import numpy as np
import OpenGL.GL as gl
import wx
from PIL import Image
from wx import glcanvas

from mlib.base.logger import MLogger
from mlib.base.math import MQuaternion, MVector3D
from mlib.pmx.pmx_collection import PmxModel
from mlib.pmx.shader import MShader
from mlib.vmd.vmd_collection import VmdMotion

logger = MLogger(__name__)


def calc_bone_matrixes(queue: Queue, fno: int, motion: VmdMotion, model: PmxModel):
    while fno < motion.max_fno:
        fno += 1
        matrixes = motion.bones.get_mesh_gl_matrixes(fno, model)
        queue.put(matrixes)
    queue.put(None)


class PmxCanvas(glcanvas.GLCanvas):
    def __init__(self, parent: wx.Panel, width: int, height: int, *args, **kw):
        attribList = (
            glcanvas.WX_GL_RGBA,
            glcanvas.WX_GL_DOUBLEBUFFER,
            glcanvas.WX_GL_DEPTH_SIZE,
            16,
            0,
        )
        glcanvas.GLCanvas.__init__(self, parent, -1, size=(width, height), attribList=attribList)
        self.parent = parent
        self.context = glcanvas.GLContext(self)
        self.size = wx.Size(width, height)
        self.last_pos = wx.Point(0, 0)
        self.now_pos = wx.Point(0, 0)
        self.fps = 30

        self.SetCurrent(self.context)

        self._initialize_ui(parent)
        self._initialize_ui_event()

        self.shader = MShader(width, height)
        self.model = PmxModel()
        self.motion = VmdMotion()
        self.bone_matrixes = np.array([np.eye(4) for _ in range(1)])

        self.queue: Optional[Queue] = None
        self.process: Optional[Process] = None
        self.capture_process: Optional[Process] = None

        # マウスドラッグフラグ
        self.is_drag = False

        # 再生中かどうかを示すフラグ
        self.playing = False
        # 録画中かどうかを示すフラグ
        self.recording = False

    def _initialize_ui(self, parent):
        gl.glClearColor(0.7, 0.7, 0.7, 1)

        self.frame_ctrl = wx.SpinCtrl(parent, value="0", min=0, max=10000, size=wx.Size(80, 30))
        self.frame_ctrl.Bind(wx.EVT_SPINCTRL, self.change_motion)

        # 再生タイマー
        self.play_timer = wx.Timer(self)

    def _initialize_ui_event(self):
        # ペイントイベントをバインド
        self.Bind(wx.EVT_PAINT, self.on_paint)

        # ウィンドウサイズ変更イベントをバインド
        self.Bind(wx.EVT_SIZE, self.on_resize)

        # 背景を消すイベントをバインド
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)

        # タイマーイベントをバインド
        self.Bind(wx.EVT_TIMER, self.on_play_timer, self.play_timer)

        self.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel)
        self.Bind(wx.EVT_MIDDLE_DOWN, self.on_mouse_down)
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_mouse_down)
        self.Bind(wx.EVT_MOTION, self.on_mouse_motion)
        self.Bind(wx.EVT_MIDDLE_UP, self.on_mouse_up)
        self.Bind(wx.EVT_RIGHT_UP, self.on_mouse_up)
        self.Bind(wx.EVT_KEY_DOWN, self.on_key_down)

    def on_erase_background(self, event: wx.Event):
        # Do nothing, to avoid flashing on MSW (これがないとチラつくらしい）
        pass

    def on_resize(self, event: wx.Event):
        self.size = self.GetClientSize()
        self.shader.fit(self.size.width, self.size.height)
        event.Skip()

    def on_paint(self, event: wx.Event):
        self.draw()
        self.SwapBuffers()

    def draw(self):
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

    def on_frame_forward(self, event: wx.Event):
        self.frame_ctrl.SetValue(min(self.motion.max_fno, self.frame_ctrl.GetValue() + 1))
        self.change_motion(event)

    def on_frame_back(self, event: wx.Event):
        self.frame_ctrl.SetValue(max(0, self.frame_ctrl.GetValue() - 1))
        self.change_motion(event)

    def change_motion(self, event: wx.Event):
        if self.model and self.motion:
            now_fno = self.frame_ctrl.GetValue()
            self.bone_matrixes = self.motion.bones.get_mesh_gl_matrixes(now_fno, self.model)
            self.Refresh()

    def on_play(self, event: wx.Event, record: bool = False):
        self.playing = not self.playing
        if self.playing:
            self.recording = record
            self.queue = Queue()
            self.process = Process(
                target=calc_bone_matrixes,
                args=(self.queue, self.frame_ctrl.GetValue(), self.motion, self.model),
                name="CalcProcess",
            )
            self.process.start()
            self.play_timer.Start(1000 // self.fps)
        else:
            if self.process:
                self.process.terminate()
            self.play_timer.Stop()
            self.recording = False

    def on_play_timer(self, event: wx.Event):
        if self.queue and not self.queue.empty():
            matrixes: Optional[np.ndarray] = None
            while not self.queue.empty():
                matrixes = self.queue.get()

            if matrixes is None and self.process:
                self.on_play(event)
                return

            if matrixes is not None and matrixes.any():
                self.bone_matrixes = matrixes

            if self.recording:
                self.on_capture(event)

            self.frame_ctrl.SetValue(self.frame_ctrl.GetValue() + 1)
            self.Refresh()

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

    def on_key_down(self, event: wx.Event):
        keycode = event.GetKeyCode()
        if keycode == wx.WXK_NUMPAD1:
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
        elif keycode == wx.WXK_NUMPAD6:
            # 左から
            self.shader.vertical_degrees = self.shader.INITIAL_VERTICAL_DEGREES
            self.shader.look_at_center = MVector3D(0, self.shader.INITIAL_LOOK_AT_CENTER_Y, 0)
            self.shader.camera_rotation = MQuaternion()
            self.shader.camera_position = MVector3D(
                self.shader.INITIAL_CAMERA_POSITION_X,
                self.shader.INITIAL_CAMERA_POSITION_Y,
                0.1,
            )
        elif keycode == wx.WXK_NUMPAD4:
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
        elif keycode in [
            wx.WXK_NUMPAD9,
            wx.WXK_RIGHT,
            wx.WXK_NUMPAD_RIGHT,
            wx.WXK_WINDOWS_RIGHT,
        ]:
            # キーフレを進める
            self.on_frame_forward(event)
        elif keycode in [
            wx.WXK_NUMPAD7,
            wx.WXK_LEFT,
            wx.WXK_NUMPAD_LEFT,
            wx.WXK_WINDOWS_LEFT,
        ]:
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

    def on_capture(self, event: wx.Event):
        dc = wx.ClientDC(self)

        # キャプチャ画像のサイズを設定
        size = dc.GetSize()

        # キャプチャ用のビットマップを作成
        bitmap = wx.Bitmap(size[0], size[1])

        # キャプチャ
        memory_dc = wx.MemoryDC()
        memory_dc.SelectObject(bitmap)
        memory_dc.Blit(0, 0, size[0], size[1], dc, 0, 0)
        memory_dc.SelectObject(wx.NullBitmap)

        # PIL.Imageに変換
        pil_image = Image.new("RGB", (size[0], size[1]))
        pil_image.frombytes(bytes(bitmap.ConvertToImage().GetData()))

        # ImageをPNGファイルとして保存する
        file_path = os.path.join(os.path.dirname(self.motion.path), "capture", f"{self.frame_ctrl.GetValue():08d}.png")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 画像をファイルに保存
        pil_image.save(file_path)

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
