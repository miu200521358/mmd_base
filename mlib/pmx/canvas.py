import logging
import os
from multiprocessing import Queue

import numpy as np
import OpenGL.GL as gl
import wx
from PIL import Image
from wx import glcanvas

from mlib.base.exception import MViewerException
from mlib.base.logger import MLogger
from mlib.base.math import MQuaternion, MVector3D
from mlib.base.process import MProcess
from mlib.pmx.pmx_collection import PmxModel
from mlib.pmx.pmx_part import ShaderMaterial
from mlib.pmx.shader import MShader
from mlib.service.form.base_frame import BaseFrame
from mlib.service.form.base_panel import BasePanel
from mlib.vmd.vmd_collection import VmdMotion

logger = MLogger(os.path.basename(__file__), level=logging.DEBUG)
__ = logger.get_text


class CanvasPanel(BasePanel):
    def __init__(self, frame: BaseFrame, tab_idx: int, width: int, height: int, *args, **kw):
        super().__init__(frame, tab_idx)
        self.index = 0
        self.canvas = PmxCanvas(self, width, height)

    @property
    def fno(self) -> int:
        return self.index

    @fno.setter
    def fno(self, v: int) -> None:
        self.index = v

    def stop_play(self) -> None:
        pass

    def start_play(self) -> None:
        pass


def animate(queue: Queue, fno: int, max_fno: int, model_set: "ModelSet"):
    while fno < max_fno:
        fno += 1
        queue.put(MotionSet(model_set.model, model_set.motion, fno))
    queue.put(None)


MODEL_BONE_COLORS = [
    np.array([1, 0, 0, 1]),
    np.array([0, 0, 1, 1]),
    np.array([0, 1, 0, 1]),
]


MODEL_AXIS_COLORS = [
    np.array([0.6, 0, 1, 1]),
    np.array([0, 1, 0.6, 1]),
    np.array([1, 0, 0.6, 1]),
]


class ModelSet:
    def __init__(self, shader: MShader, model: PmxModel, motion: VmdMotion, bone_alpha: float = 1.0):
        self.model = model
        self.motion = motion
        self.bone_alpha = bone_alpha
        model.init_draw(shader)


class MotionSet:
    def __init__(self, model: PmxModel, motion: VmdMotion, fno: int) -> None:
        if motion is not None:
            (
                self.gl_matrixes,
                self.vertex_morph_poses,
                self.uv_morph_poses,
                self.uv1_morph_poses,
                self.material_morphs,
            ) = motion.animate(fno, model)
        else:
            self.gl_matrixes = np.array([np.eye(4) for _ in range(len(model.bones))])
            self.vertex_morph_poses = np.array([np.zeros(3) for _ in range(len(model.vertices))])
            self.uv_morph_poses = np.array([np.zeros(4) for _ in range(len(model.vertices))])
            self.uv1_morph_poses = np.array([np.zeros(4) for _ in range(len(model.vertices))])
            self.material_morphs = [ShaderMaterial(m, MShader.LIGHT_AMBIENT4) for m in model.materials]

    def update_morphs(self, model: PmxModel, motion: VmdMotion, fno: int):
        self.vertex_morph_poses = motion.morphs.animate_vertex_morphs(fno, model)
        self.uv_morph_poses = motion.morphs.animate_uv_morphs(fno, model, 0)
        self.uv1_morph_poses = motion.morphs.animate_uv_morphs(fno, model, 1)
        self.material_morphs = motion.morphs.animate_material_morphs(fno, model)


class PmxCanvas(glcanvas.GLCanvas):
    def __init__(self, parent: CanvasPanel, width: int, height: int, *args, **kw):
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
        self.max_fno = 0

        self.set_context()

        self._initialize_ui()
        self._initialize_event()

        self.shader = MShader(width, height)
        self.model_sets: list[ModelSet] = []
        self.animations: list[MotionSet] = []

        self.queues: list[Queue] = []
        self.processes: list[MProcess] = []

        # マウスドラッグフラグ
        self.is_drag = False

        # 再生中かどうかを示すフラグ
        self.playing = False
        # 録画中かどうかを示すフラグ
        self.recording = False

    def _initialize_ui(self) -> None:
        gl.glClearColor(0.7, 0.7, 0.7, 1)

        # 再生タイマー
        self.play_timer = wx.Timer(self)

    def _initialize_event(self) -> None:
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
        try:
            self.draw()
            self.SwapBuffers()
        except MViewerException:
            error_msg = "ビューワーの描画に失敗しました。\n一度ツールを立ち上げ直して再度実行していただき、それでも解決しなかった場合、作者にご連絡下さい。"
            logger.critical(error_msg)

            self.clear_model_set()

            dialog = wx.MessageDialog(
                self.parent,
                __(error_msg),
                style=wx.OK,
            )
            dialog.ShowModal()
            dialog.Destroy()

            self.SwapBuffers()

    def set_context(self) -> None:
        self.SetCurrent(self.context)

    def append_model_set(self, model: PmxModel, motion: VmdMotion, bone_alpha: float = 1.0):
        logger.debug("append_model_set: model_sets")
        self.model_sets.append(ModelSet(self.shader, model, motion, bone_alpha))
        logger.debug("append_model_set: animations")
        self.animations.append(MotionSet(model, motion, 0))
        logger.debug("append_model_set: max_fno")
        self.max_fno = max([model_set.motion.max_fno for model_set in self.model_sets])

    def clear_model_set(self) -> None:
        if self.model_sets:
            del self.model_sets
            del self.animations
        self.model_sets = []
        self.animations = []

    def draw(self) -> None:
        self.set_context()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        self.shader.update_camera()

        self.shader.msaa.bind()

        # 地面を描く
        self.draw_ground()

        # 透過度設定なしのメッシュを先に描画する
        for model_set, animation in zip(self.model_sets, self.animations):
            if model_set.model:
                logger.test(f"-- アニメーション描画(非透過): {model_set.model.name}")

                model_set.model.draw(
                    animation.gl_matrixes,
                    animation.vertex_morph_poses,
                    animation.uv_morph_poses,
                    animation.uv1_morph_poses,
                    animation.material_morphs,
                    False,
                )
        # その後透過度設定ありのメッシュを描画する
        for model_set, animation in zip(self.model_sets, self.animations):
            if model_set.model:
                logger.test(f"-- アニメーション描画(透過): {model_set.model.name}")

                model_set.model.draw(
                    animation.gl_matrixes,
                    animation.vertex_morph_poses,
                    animation.uv_morph_poses,
                    animation.uv1_morph_poses,
                    animation.material_morphs,
                    True,
                )
        self.shader.msaa.unbind()

        for model_set, animation, color in zip(self.model_sets, self.animations, MODEL_BONE_COLORS):
            # ボーンを表示
            if model_set.model:
                logger.test(f"-- ボーン描画(透過): {model_set.model.name}")

                model_set.model.draw_bone(
                    animation.gl_matrixes,
                    color * np.array([1, 1, 1, model_set.bone_alpha], dtype=np.float32),
                )

        # if logging.DEBUG >= logger.total_level:
        #     for model_set, animation, color in zip(self.model_sets, self.animations, MODEL_AXIS_COLORS):
        #         # ローカル軸を表示
        #         if model_set.model:
        #             logger.test(f"-- ローカル軸描画(透過): {model_set.model.name}")

        #             model_set.model.draw_axis(
        #                 animation.gl_matrixes,
        #                 color * np.fromiter([1, 1, 1, model_set.bone_alpha], count=4, dtype=np.float32),
        #             )

    def draw_ground(self) -> None:
        """平面を描画する"""
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.5, 0.5, 0.5, 0.5)
        gl.glVertex3f(-30.0, 0.0, -30.0)
        gl.glVertex3f(30.0, 0.0, -30.0)
        gl.glVertex3f(30.0, 0.0, 30.0)
        gl.glVertex3f(-30.0, 0.0, 30.0)
        gl.glEnd()

    def on_frame_forward(self, event: wx.Event):
        self.parent.fno = self.parent.fno + 1
        self.change_motion(event)

    def on_frame_back(self, event: wx.Event):
        self.parent.fno = max(0, self.parent.fno - 1)
        self.change_motion(event)

    def change_motion(self, event: wx.Event, is_bone_deform: bool = True, model_index: int = -1):
        if is_bone_deform:
            if 0 > model_index:
                animations: list[MotionSet] = []
                for model_set in self.model_sets:
                    logger.debug(f"change_motion: MotionSet: {model_set.model.name}")
                    animations.append(MotionSet(model_set.model, model_set.motion, self.parent.fno))
                self.animations = animations
            else:
                self.animations[model_index] = MotionSet(self.model_sets[model_index].model, self.model_sets[model_index].motion, self.parent.fno)
        else:
            for model_set, animation in zip(self.model_sets, self.animations):
                logger.debug(f"change_motion: update_morphs: {model_set.model.name}")
                animation.update_morphs(model_set.model, model_set.motion, self.parent.fno)

        if self.playing and self.max_fno <= self.parent.fno:
            # 最後まで行ったら止まる
            self.on_play(event)

        self.Refresh()

    def on_play(self, event: wx.Event, record: bool = False):
        self.playing = not self.playing
        if self.playing:
            logger.debug("on_play ----------------------------------------")
            self.parent.start_play()
            self.max_fno = max([model_set.motion.max_fno for model_set in self.model_sets])
            self.recording = record
            for n, model_set in enumerate(self.model_sets):
                logger.debug(f"on_play queue[{n}] append")
                self.queues.append(Queue())
                logger.debug(f"on_play process[{n}] append")
                self.processes.append(
                    MProcess(
                        target=animate,
                        args=(self.queues[-1], self.parent.fno, self.max_fno, model_set),
                        name="CalcProcess",
                    )
                )
            logger.debug("on_play process start")
            for p in self.processes:
                p.start()
            logger.debug("on_play timer start")
            self.play_timer.Start(1000 // self.fps)
        else:
            self.play_timer.Stop()
            self.recording = False
            self.clear_process()
            self.parent.stop_play()

    def clear_process(self) -> None:
        if self.processes:
            for p in self.processes:
                if p.is_alive():
                    p.terminate()
                del p
            self.processes = []
        if self.queues:
            for q in self.queues:
                del q
            self.queues = []

    def on_play_timer(self, event: wx.Event):
        if self.queues:
            # 全てのキューが終わったら受け取る
            animations: list[MotionSet] = []
            for q in self.queues:
                animation = q.get()
                animations.append(animation)

            if None in animations and self.processes:
                # アニメーションが終わったら再生をひっくり返す
                self.on_play(event)
                return

            if animations:
                self.animations = animations

            if self.recording:
                self.on_capture(event)

            self.parent.fno += 1
            self.Refresh()

    def on_reset(self, event: wx.Event):
        self.parent.fno = 0
        self.shader.vertical_degrees = self.shader.INITIAL_VERTICAL_DEGREES
        self.shader.look_at_center = self.shader.INITIAL_LOOK_AT_CENTER_POSITION.copy()
        self.shader.camera_rotation = MQuaternion()
        self.shader.camera_position = self.shader.INITIAL_CAMERA_POSITION.copy()
        self.shader.camera_offset_position = self.shader.INITIAL_CAMERA_OFFSET_POSITION.copy()
        self.Refresh()

    def on_key_down(self, event: wx.Event):
        keycode = event.GetKeyCode()
        if keycode == wx.WXK_NUMPAD1:
            # 真下から
            self.shader.vertical_degrees = self.shader.INITIAL_VERTICAL_DEGREES
            self.shader.look_at_center = self.shader.INITIAL_LOOK_AT_CENTER_POSITION.copy()
            self.shader.camera_rotation = MQuaternion.from_euler_degrees(-90, 0, 0)
            self.shader.camera_offset_position = MVector3D(0, self.shader.INITIAL_CAMERA_POSITION_Y, 0)
        elif keycode in [wx.WXK_NUMPAD2, wx.WXK_ESCAPE]:
            # 真正面から(=リセット)
            self.shader.vertical_degrees = self.shader.INITIAL_VERTICAL_DEGREES
            self.shader.look_at_center = self.shader.INITIAL_LOOK_AT_CENTER_POSITION.copy()
            self.shader.camera_rotation = MQuaternion()
            self.shader.camera_position = self.shader.INITIAL_CAMERA_POSITION.copy()
            self.shader.camera_offset_position = self.shader.INITIAL_CAMERA_OFFSET_POSITION.copy()
        elif keycode == wx.WXK_NUMPAD6:
            # 左から
            self.shader.vertical_degrees = self.shader.INITIAL_VERTICAL_DEGREES
            self.shader.look_at_center = self.shader.INITIAL_LOOK_AT_CENTER_POSITION.copy()
            self.shader.camera_rotation = MQuaternion.from_euler_degrees(0, 90, 0)
            self.shader.camera_offset_position = MVector3D()
        elif keycode == wx.WXK_NUMPAD4:
            # 右から
            self.shader.vertical_degrees = self.shader.INITIAL_VERTICAL_DEGREES
            self.shader.look_at_center = self.shader.INITIAL_LOOK_AT_CENTER_POSITION.copy()
            self.shader.camera_rotation = MQuaternion.from_euler_degrees(0, -90, 0)
            self.shader.camera_offset_position = MVector3D()
        elif keycode == wx.WXK_NUMPAD8:
            # 真後ろから
            self.shader.vertical_degrees = self.shader.INITIAL_VERTICAL_DEGREES
            self.shader.look_at_center = self.shader.INITIAL_LOOK_AT_CENTER_POSITION.copy()
            self.shader.camera_rotation = MQuaternion.from_euler_degrees(0, 180, 0)
            self.shader.camera_offset_position = MVector3D()
        elif keycode == wx.WXK_NUMPAD5:
            # 真上から
            self.shader.vertical_degrees = self.shader.INITIAL_VERTICAL_DEGREES
            self.shader.look_at_center = self.shader.INITIAL_LOOK_AT_CENTER_POSITION.copy()
            self.shader.camera_rotation = MQuaternion.from_euler_degrees(90, 180, 0)
            self.shader.camera_offset_position = MVector3D(0, self.shader.INITIAL_CAMERA_POSITION_Y, 0)
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
        file_path = os.path.join(os.path.dirname(self.motion.path), "capture", f"{self.parent.fno:06d}.png")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 画像をファイルに保存
        pil_image.save(file_path)

    def on_mouse_down(self, event: wx.Event):
        if not self.is_drag:
            self.now_pos = self.last_pos = event.GetPosition()
            self.is_drag = True
            self.CaptureMouse()

    def on_mouse_up(self, event: wx.Event):
        if self.is_drag:
            self.is_drag = False
            self.ReleaseMouse()

    def on_mouse_motion(self, event: wx.Event):
        if self.is_drag and event.Dragging():
            self.now_pos = event.GetPosition()
            x = (self.now_pos.x - self.last_pos.x) * 0.02
            y = (self.now_pos.y - self.last_pos.y) * 0.02
            if event.MiddleIsDown():
                self.shader.look_at_center += self.shader.camera_rotation * MVector3D(x, y, 0)

                self.shader.camera_offset_position.x += x
                self.shader.camera_offset_position.y += y
            elif event.RightIsDown():
                self.shader.camera_rotation *= MQuaternion.from_euler_degrees(y * 10, -x * 10, 0)
            self.last_pos = self.now_pos
            self.Refresh()

    def on_mouse_wheel(self, event: wx.Event):
        unit_degree = 5.0 if event.ShiftDown() else 1.0 if event.ControlDown() else 2.5
        if 0 > event.GetWheelRotation():
            self.shader.vertical_degrees += unit_degree
        else:
            self.shader.vertical_degrees = max(1.0, self.shader.vertical_degrees - unit_degree)
        self.Refresh()
