import os
import shutil
import sys
from datetime import datetime
from typing import Any, Optional
import numpy as np

import wx

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from mlib.base.math import MQuaternion, MVector3D

from mlib.vmd.vmd_part import VmdBoneFrame
from mlib.pmx.pmx_part import Bone, Face, Material, SphereMode, Texture, ToonSharing, Vertex
from mlib.base.exception import MApplicationException
from mlib.base.logger import MLogger
from mlib.pmx.canvas import CanvasPanel
from mlib.pmx.pmx_collection import PmxModel
from mlib.pmx.pmx_writer import PmxWriter
from mlib.service.base_worker import BaseWorker
from mlib.service.form.base_frame import BaseFrame
from mlib.service.form.base_panel import BasePanel
from mlib.service.form.widgets.console_ctrl import ConsoleCtrl
from mlib.service.form.widgets.exec_btn_ctrl import ExecButton
from mlib.service.form.widgets.file_ctrl import MPmxFilePickerCtrl, MVmdFilePickerCtrl
from mlib.service.form.widgets.float_slider_ctrl import FloatSliderCtrl
from mlib.service.form.widgets.spin_ctrl import WheelSpinCtrl, WheelSpinCtrlDouble
from mlib.utils.file_utils import save_histories, separate_path
from mlib.vmd.vmd_collection import VmdMotion

logger = MLogger(os.path.basename(__file__))
__ = logger.get_text


class FilePanel(BasePanel):
    def __init__(self, frame: BaseFrame, tab_idx: int, *args, **kw):
        super().__init__(frame, tab_idx, *args, **kw)

        self._initialize_ui()

    def _initialize_ui(self):
        self.model_ctrl = MPmxFilePickerCtrl(
            self.frame,
            self,
            key="model_pmx",
            title="表示モデル",
            is_show_name=True,
            name_spacer=20,
            is_save=False,
            tooltip="PMXモデル",
            file_change_event=self.on_change_model_pmx,
        )
        self.model_ctrl.set_parent_sizer(self.root_sizer)

        self.dress_ctrl = MPmxFilePickerCtrl(
            self.frame,
            self,
            key="dress_pmx",
            title="衣装モデル",
            is_show_name=True,
            name_spacer=20,
            is_save=False,
            tooltip="PMX衣装モデル",
            file_change_event=self.on_change_dress_pmx,
        )
        self.dress_ctrl.set_parent_sizer(self.root_sizer)

        self.motion_ctrl = MVmdFilePickerCtrl(
            self.frame,
            self,
            key="motion_vmd",
            title="表示モーション",
            is_show_name=True,
            name_spacer=20,
            is_save=False,
            tooltip="VMDモーションデータ",
            file_change_event=self.on_change_motion,
        )
        self.motion_ctrl.set_parent_sizer(self.root_sizer)

        self.output_pmx_ctrl = MPmxFilePickerCtrl(
            self.frame,
            self,
            title="出力先",
            is_show_name=False,
            is_save=True,
            tooltip="実際は動いてないよ",
        )
        self.output_pmx_ctrl.set_parent_sizer(self.root_sizer)

        self.exec_btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.exec_btn_ctrl = ExecButton(self, "実行", "実行中", self.exec, 200, "実行ボタンだよ")
        self.exec_btn_sizer.Add(self.exec_btn_ctrl, 0, wx.ALL, 3)
        self.root_sizer.Add(self.exec_btn_sizer, 0, wx.ALIGN_CENTER | wx.SHAPED, 5)

        self.console_ctrl = ConsoleCtrl(self.frame, self, rows=300)
        self.console_ctrl.set_parent_sizer(self.root_sizer)

        self.root_sizer.Add(wx.StaticLine(self, wx.ID_ANY), wx.GROW)
        self.fit()

    def exec(self, event: wx.Event):
        if not (self.model_ctrl.data and self.dress_ctrl.data):
            return

        os.makedirs(os.path.dirname(self.output_pmx_ctrl.path), exist_ok=True)
        model = self.model_ctrl.data
        dress = self.dress_ctrl.data
        dress_model = PmxModel(self.output_pmx_ctrl.path)
        dress_model.comment = model.comment + "\n------------------\n" + dress.comment
        dress_model.initialize_display_slots()

        bone_map: dict[int, dict[str, list[str]]] = {}

        # 最初にルートを追加する
        root_bone = Bone(name=Bone.SYSTEM_ROOT_NAME, index=-1)
        root_bone.parent_index = -9
        root_bone.is_system = True
        dress_model.bones.append(root_bone, is_positive_index=False)
        bone_map[-1] = {
            "parent": [Bone.SYSTEM_ROOT_NAME],
            "tail": [Bone.SYSTEM_ROOT_NAME],
            "effect": [Bone.SYSTEM_ROOT_NAME],
            "ik_target": [Bone.SYSTEM_ROOT_NAME],
            "ik_link": [],
        }

        model_bone_map: dict[int, int] = {-1: -1}
        dress_bone_map: dict[int, int] = {-1: -1}

        for bone in model.bones.writable():
            if bone.name in dress_model.bones:
                continue

            for dress_bone in dress.bones.writable():
                if 0 <= dress_bone.tail_index and dress_bone.name not in model.bones and dress.bones[dress_bone.tail_index].name == bone.name:
                    # 衣装だけのボーンが表示先が人物のボーンに繋がってる場合、その前に追加しておく
                    copy_bone = dress_bone.copy()
                    copy_bone.index = len(dress_model.bones.writable())
                    bone_map[copy_bone.index] = {
                        "parent": [dress.bones[dress_bone.parent_index].name],
                        "tail": [dress.bones[dress_bone.tail_index].name],
                        "effect": [dress.bones[dress_bone.effect_index].name],
                        "ik_target": [dress.bones[dress_bone.ik.bone_index].name if dress_bone.ik else Bone.SYSTEM_ROOT_NAME],
                        "ik_link": [dress.bones[link.bone_index].name for link in dress_bone.ik.links] if dress_bone.ik else [],
                    }
                    dress_model.bones.append(copy_bone, is_sort=False)
                    dress_bone_map[bone.index] = copy_bone.index

            copy_bone = bone.copy()
            copy_bone.index = len(dress_model.bones.writable())
            bone_map[copy_bone.index] = {
                "parent": [model.bones[bone.parent_index].name],
                "tail": [model.bones[bone.tail_index].name],
                "effect": [model.bones[bone.effect_index].name],
                "ik_target": [model.bones[bone.ik.bone_index].name if bone.ik else Bone.SYSTEM_ROOT_NAME],
                "ik_link": [model.bones[link.bone_index].name for link in bone.ik.links] if bone.ik else [],
            }
            dress_model.bones.append(copy_bone, is_sort=False)
            model_bone_map[bone.index] = copy_bone.index

        for bone in dress.bones.writable():
            if bone.name in dress_model.bones:
                # 既に登録済みのボーンは追加しない
                continue
            copy_bone = bone.copy()
            copy_bone.index = len(dress_model.bones.writable())
            bone_map[copy_bone.index] = {
                "parent": [dress.bones[bone.parent_index].name],
                "tail": [dress.bones[bone.tail_index].name],
                "effect": [dress.bones[bone.effect_index].name],
                "ik_target": [dress.bones[bone.ik.bone_index].name if bone.ik else Bone.SYSTEM_ROOT_NAME],
                "ik_link": [dress.bones[link.bone_index].name for link in bone.ik.links] if bone.ik else [],
            }
            dress_model.bones.append(copy_bone, is_sort=False)
            dress_bone_map[bone.index] = copy_bone.index

        for bone in dress_model.bones:
            bone_setting = bone_map[bone.index]
            bone.parent_index = dress_model.bones[bone_setting["parent"][0]].index
            bone.tail_index = dress_model.bones[bone_setting["tail"][0]].index
            bone.effect_index = dress_model.bones[bone_setting["effect"][0]].index
            if bone.is_ik and bone.ik:
                bone.ik.bone_index = dress_model.bones[bone_setting["ik_target"][0]].index
                for n in range(len(bone.ik.links)):
                    bone.ik.links[n].bone_index = dress_model.bones[bone_setting["ik_link"][n]].index

        model_vertex_map: dict[int, int] = {-1: -1}
        dress_vertex_map: dict[int, int] = {-1: -1}

        model_material_map: dict[int, int] = {-1: -1}
        dress_material_map: dict[int, int] = {-1: -1}

        prev_faces_count = 0
        for material in model.materials:
            if "01_Onepiece_02" in material.name:
                prev_faces_count += material.vertices_count // 3
                continue
            copy_material: Material = material.copy()
            copy_material.index = len(dress_model.materials)

            if 0 <= material.texture_index:
                copy_texture = self.copy_texture(dress_model, model.textures[material.texture_index], model.path)
                copy_material.texture_index = copy_texture.index

            if material.toon_sharing_flg == ToonSharing.INDIVIDUAL and 0 <= material.toon_texture_index:
                copy_texture = self.copy_texture(dress_model, model.textures[material.toon_texture_index], model.path)
                copy_material.toon_texture_index = copy_texture.index

            if material.sphere_mode != SphereMode.INVALID and 0 < material.sphere_texture_index:
                copy_texture = self.copy_texture(dress_model, model.textures[material.sphere_texture_index], model.path)
                copy_material.sphere_texture_index = copy_texture.index

            dress_model.materials.append(copy_material, is_sort=False)
            model_material_map[material.index] = copy_material.index

            for face_index in range(prev_faces_count, prev_faces_count + copy_material.vertices_count // 3):
                faces = []
                for vertex_index in model.faces[face_index].vertices:
                    if vertex_index not in model_vertex_map:
                        copy_vertex: Vertex = model.vertices[vertex_index].copy()
                        copy_vertex.index = -1
                        copy_vertex.deform.indexes = np.vectorize(model_bone_map.get)(copy_vertex.deform.indexes)
                        faces.append(len(dress_model.vertices))
                        model_vertex_map[vertex_index] = len(dress_model.vertices)
                        dress_model.vertices.append(copy_vertex, is_sort=False)
                    else:
                        faces.append(model_vertex_map[vertex_index])
                dress_model.faces.append(Face(vertex_index0=faces[0], vertex_index1=faces[1], vertex_index2=faces[2]), is_sort=False)

            prev_faces_count += material.vertices_count // 3

        prev_faces_count = 0
        for material in dress.materials:
            if "腕" in material.name:
                prev_faces_count += material.vertices_count // 3
                continue
            copy_material = material.copy()
            copy_material.name = f"Cos:{copy_material.name}"
            copy_material.index = len(dress_model.materials)

            if 0 <= material.texture_index:
                copy_texture = self.copy_texture(dress_model, dress.textures[material.texture_index], dress.path)
                copy_material.texture_index = copy_texture.index

            if material.toon_sharing_flg == ToonSharing.INDIVIDUAL and 0 <= material.toon_texture_index:
                copy_texture = self.copy_texture(dress_model, dress.textures[material.toon_texture_index], dress.path)
                copy_material.toon_texture_index = copy_texture.index

            if material.sphere_mode != SphereMode.INVALID and 0 < material.sphere_texture_index:
                copy_texture = self.copy_texture(dress_model, dress.textures[material.sphere_texture_index], dress.path)
                copy_material.sphere_texture_index = copy_texture.index

            dress_model.materials.append(copy_material, is_sort=False)
            dress_material_map[material.index] = copy_material.index

            for face_index in range(prev_faces_count, prev_faces_count + copy_material.vertices_count // 3):
                faces = []
                for vertex_index in dress.faces[face_index].vertices:
                    if vertex_index not in dress_vertex_map:
                        copy_vertex = dress.vertices[vertex_index].copy()
                        copy_vertex.index = -1
                        copy_vertex.deform.indexes = np.vectorize(model_bone_map.get)(copy_vertex.deform.indexes)
                        faces.append(len(dress_model.vertices))
                        dress_vertex_map[vertex_index] = len(dress_model.vertices)
                        dress_model.vertices.append(copy_vertex, is_sort=False)
                    else:
                        faces.append(dress_vertex_map[vertex_index])
                dress_model.faces.append(Face(vertex_index0=faces[0], vertex_index1=faces[1], vertex_index2=faces[2]), is_sort=False)

            prev_faces_count += material.vertices_count // 3

        PmxWriter(dress_model, self.output_pmx_ctrl.path).save()

        logger.info(self.output_pmx_ctrl.path, decoration=MLogger.Decoration.BOX)
        self.frame.on_sound()

    def copy_texture(self, dest_model: PmxModel, texture: Texture, src_model_path: str) -> Texture:
        copy_texture: Texture = texture.copy()
        copy_texture.index = len(dest_model.textures)
        texture_path = os.path.abspath(os.path.join(os.path.dirname(src_model_path), copy_texture.name))
        if copy_texture.name and os.path.exists(texture_path) and os.path.isfile(texture_path):
            new_texture_path = os.path.join(os.path.dirname(dest_model.path), copy_texture.name)
            os.makedirs(os.path.dirname(new_texture_path), exist_ok=True)
            shutil.copyfile(texture_path, new_texture_path)
        copy_texture.index = len(dest_model.textures)
        dest_model.textures.append(copy_texture, is_sort=False)

        return copy_texture

    def on_change_model_pmx(self, event: wx.Event):
        self.model_ctrl.unwrap()
        if self.model_ctrl.read_name():
            self.model_ctrl.read_digest()
            dir_path, file_name, file_ext = separate_path(self.model_ctrl.path)
            model_path = os.path.join(dir_path, f"{datetime.now():%Y%m%d_%H%M%S}", f"{file_name}_{datetime.now():%Y%m%d_%H%M%S}{file_ext}")
            self.output_pmx_ctrl.path = model_path

    def on_change_dress_pmx(self, event: wx.Event):
        self.dress_ctrl.unwrap()
        if self.dress_ctrl.read_name():
            self.dress_ctrl.read_digest()

    def on_change_motion(self, event: wx.Event):
        self.motion_ctrl.unwrap()
        if self.motion_ctrl.read_name():
            self.motion_ctrl.read_digest()

    def Enable(self, enable: bool):
        self.model_ctrl.Enable(enable)
        self.dress_ctrl.Enable(enable)
        self.motion_ctrl.Enable(enable)


class PmxLoadWorker(BaseWorker):
    def __init__(self, panel: BasePanel, result_event: wx.Event) -> None:
        super().__init__(panel, result_event)

    def thread_execute(self):
        file_panel: FilePanel = self.panel
        model: Optional[PmxModel] = None
        dress: Optional[PmxModel] = None
        motion: Optional[VmdMotion] = None

        if not file_panel.model_ctrl.data and file_panel.model_ctrl.valid():
            model = file_panel.model_ctrl.reader.read_by_filepath(file_panel.model_ctrl.path)

            if "首" not in model.bones:
                raise MApplicationException("{b}ボーン不足", b="首")
        elif file_panel.model_ctrl.data:
            model = file_panel.model_ctrl.data
        else:
            model = PmxModel()

        if not file_panel.dress_ctrl.data and file_panel.dress_ctrl.valid():
            dress = file_panel.dress_ctrl.reader.read_by_filepath(file_panel.dress_ctrl.path)
            PmxWriter(dress, "C:/MMD/mmd_base/tests/resources/result.pmx", include_system=True).save()
        elif file_panel.dress_ctrl.data:
            dress = file_panel.dress_ctrl.data
        else:
            dress = PmxModel()

        if not file_panel.motion_ctrl.data and file_panel.motion_ctrl.valid():
            motion = file_panel.motion_ctrl.reader.read_by_filepath(file_panel.motion_ctrl.path)
        elif file_panel.motion_ctrl.data:
            motion = file_panel.motion_ctrl.data
        else:
            motion = VmdMotion("empty")

        self.result_data = (model, dress, motion)

    def output_log(self):
        pass


class ConfigPanel(CanvasPanel):
    def __init__(self, frame: BaseFrame, tab_idx: int, *args, **kw):
        super().__init__(frame, tab_idx, 500, 800, *args, **kw)

        self._initialize_ui()
        self._initialize_event()

    def _initialize_ui(self):
        self.config_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.config_sizer.Add(self.canvas, 0, wx.EXPAND | wx.ALL, 0)

        self.btn_sizer = wx.BoxSizer(wx.VERTICAL)

        # キーフレ
        self.frame_ctrl = WheelSpinCtrl(self, change_event=self.on_change_frame, initial=0, min=-100, max=10000, size=wx.Size(80, -1))
        self.btn_sizer.Add(self.frame_ctrl, 0, wx.ALL, 5)

        # キーフレ
        self.double_ctrl = WheelSpinCtrlDouble(self, change_event=self.on_change_frame, initial=0, min=-100, max=10000, inc=0.01, size=wx.Size(80, -1))
        self.btn_sizer.Add(self.double_ctrl, 0, wx.ALL, 5)

        # 再生
        self.play_btn = wx.Button(self, wx.ID_ANY, "Play", wx.DefaultPosition, wx.Size(100, 50))
        self.btn_sizer.Add(self.play_btn, 0, wx.ALL, 5)

        # スライダー
        self.float_slider = FloatSliderCtrl(
            self,
            value=0.5,
            min_value=0,
            max_value=3,
            increment=0.01,
            spin_increment=0.1,
            border=3,
            position=wx.DefaultPosition,
            size=wx.Size(200, -1),
        )
        self.btn_sizer.Add(self.float_slider.sizer, 0, wx.ALL, 0)

        self.config_sizer.Add(self.btn_sizer, 0, wx.ALL, 0)
        self.root_sizer.Add(self.config_sizer, 0, wx.ALL, 0)

        self.fit()

    def _initialize_event(self):
        self.play_btn.Bind(wx.EVT_BUTTON, self.on_play)

    def on_play(self, event: wx.Event):
        self.canvas.on_play(event)
        self.play_btn.SetLabel("Stop" if self.canvas.playing else "Play")

    @property
    def fno(self):
        return self.frame_ctrl.GetValue()

    @fno.setter
    def fno(self, v: int):
        self.frame_ctrl.SetValue(v)

    def stop_play(self):
        self.play_btn.SetLabel("Play")

    def on_change_frame(self, event: wx.Event):
        self.fno = self.frame_ctrl.GetValue()
        self.canvas.change_motion(event)


class TestFrame(BaseFrame):
    def __init__(self, app) -> None:
        super().__init__(
            app,
            history_keys=["model_pmx", "dress_pmx", "motion_vmd"],
            title="Mu Test Frame",
            size=wx.Size(1000, 800),
        )
        # ファイルタブ
        self.file_panel = FilePanel(self, 0)
        self.notebook.AddPage(self.file_panel, __("ファイル"), False)

        # 設定タブ
        self.config_panel = ConfigPanel(self, 1)
        self.notebook.AddPage(self.config_panel, __("設定"), False)

        self.worker = PmxLoadWorker(self.file_panel, self.on_result)

    def on_change_tab(self, event: wx.Event):
        if self.notebook.GetSelection() == self.config_panel.tab_idx:
            self.notebook.ChangeSelection(self.file_panel.tab_idx)
            if not self.worker.started:
                if not self.file_panel.model_ctrl.valid():
                    logger.warning("モデル欄に有効なパスが設定されていない為、タブ遷移を中断します。")
                    return
                if not self.file_panel.dress_ctrl.valid():
                    logger.warning("衣装欄に有効なパスが設定されていない為、タブ遷移を中断します。")
                    return
                if not self.file_panel.model_ctrl.data or not self.file_panel.dress_ctrl.data or not self.file_panel.motion_ctrl.data:
                    # 設定タブにうつった時に読み込む
                    self.config_panel.canvas.clear_model_set()
                    self.save_histories()

                    self.worker.start()
                else:
                    # 既に読み取りが完了していたらそのまま表示
                    self.notebook.ChangeSelection(self.config_panel.tab_idx)

    def save_histories(self):
        self.file_panel.model_ctrl.save_path()
        self.file_panel.dress_ctrl.save_path()
        self.file_panel.motion_ctrl.save_path()

        save_histories(self.histories)

    def on_result(self, result: bool, data: Optional[Any], elapsed_time: str):
        self.file_panel.console_ctrl.write(f"\n----------------\n{elapsed_time}")

        if not (result and data):
            return

        data1, data2, data3 = data
        model: PmxModel = data1
        dress: PmxModel = data2
        motion: VmdMotion = data3
        self.file_panel.model_ctrl.set_data(model)
        self.file_panel.dress_ctrl.set_data(dress)
        self.file_panel.motion_ctrl.set_data(motion)

        if not (self.file_panel.motion_ctrl.data and self.file_panel.model_ctrl.data and self.file_panel.dress_ctrl.data):
            return

        dress_motion: VmdMotion = self.file_panel.motion_ctrl.data.copy()

        bf = VmdBoneFrame(0, "右腕")
        bf.local_position = MVector3D(0, 1, 0)
        bf.local_rotation = MQuaternion.from_euler_degrees(0, 20, 0)
        dress_motion.bones["右腕"].append(bf)

        bf2 = VmdBoneFrame(0, "上半身2")
        bf2.local_position = MVector3D(0, 0, 1)
        dress_motion.bones["上半身2"].append(bf2)

        bf = VmdBoneFrame(0, "左肩")
        bf.local_scale = MVector3D(0, 1.5, 1.5)
        bf.local_position = MVector3D(0, 1, 0)
        dress_motion.bones["左肩"].append(bf)

        bf = VmdBoneFrame(0, "左腕")
        bf.local_scale = MVector3D(0, -0.5, -0.5)
        dress_motion.bones["左腕"].append(bf)

        bf = VmdBoneFrame(0, "左ひじ")
        bf.local_scale = MVector3D(0, 1.2, 1.2)
        dress_motion.bones["左ひじ"].append(bf)

        try:
            self.config_panel.canvas.set_context()
            self.config_panel.canvas.append_model_set(self.file_panel.model_ctrl.data, dress_motion)
            self.config_panel.canvas.append_model_set(self.file_panel.dress_ctrl.data, dress_motion, 0.3)
            self.config_panel.canvas.Refresh()
            self.notebook.ChangeSelection(self.config_panel.tab_idx)
        except:
            logger.critical("モデル描画初期化処理失敗")


class MuApp(wx.App):
    def __init__(self, redirect=False, filename=None, useBestVisual=False, clearSigInt=True):
        super().__init__(redirect, filename, useBestVisual, clearSigInt)
        self.frame = TestFrame(self)
        self.frame.Show()


if __name__ == "__main__":
    MLogger.initialize(
        lang="en",
        root_dir=os.path.join(os.path.dirname(__file__), "..", "mlib"),
        level=10,
        is_out_log=True,
    )

    app = MuApp()
    app.MainLoop()
