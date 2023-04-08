import os
from typing import Optional

import wx

from mlib.base.collection import BaseHashModel
from mlib.base.logger import MLogger
from mlib.base.reader import BaseReader
from mlib.form.base_frame import BaseFrame
from mlib.form.base_panel import BasePanel
from mlib.utils.file_utils import get_dir_path, insert_history, validate_file, validate_save_file, unwrapped_path

logger = MLogger(os.path.basename(__file__))
__ = logger.get_text


class MFilePickerCtrl:
    def __init__(
        self,
        frame: BaseFrame,
        parent: BasePanel,
        reader: BaseReader,
        title: str,
        key: Optional[str] = None,
        is_show_name: bool = True,
        name_spacer: int = 0,
        is_save: bool = False,
        tooltip: str = "",
        file_change_event=None,
    ) -> None:
        self.frame = frame
        self.parent = parent
        self.reader = reader
        self.data: Optional[BaseHashModel] = None
        self.key = key
        self.title = __(title)
        self.is_save = is_save
        self.is_show_name = is_show_name
        self.file_change_event = file_change_event
        self.root_sizer = wx.BoxSizer(wx.VERTICAL)

        self._initialize_ui(name_spacer, tooltip)
        self._initialize_event()

    def set_parent_sizer(self, parent_sizer: wx.Sizer):
        parent_sizer.Add(self.root_sizer, 1, wx.GROW, 0)

    def _initialize_ui(self, name_spacer: int, tooltip: str):
        # ファイルタイトル
        self.title_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.title_ctrl = wx.StaticText(self.parent, wx.ID_ANY, self.title, wx.DefaultPosition, wx.DefaultSize, 0)
        self.title_ctrl.SetToolTip(__(tooltip))
        self.title_sizer.Add(self.title_ctrl, 1, wx.GROW | wx.ALL, 3)

        # モデル名等の表示
        if self.is_show_name and not self.is_save:
            self.spacer_ctrl = wx.StaticText(self.parent, wx.ID_ANY, " " * name_spacer, wx.DefaultPosition, wx.DefaultSize, 0)
            self.title_sizer.Add(self.spacer_ctrl, 0, wx.ALL, 3)

            self.name_ctrl = wx.TextCtrl(
                self.parent,
                wx.ID_ANY,
                __("(未設定)"),
                wx.DefaultPosition,
                wx.DefaultSize,
                wx.TE_READONLY | wx.BORDER_NONE | wx.WANTS_CHARS,
            )
            self.name_ctrl.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DLIGHT))
            self.name_ctrl.SetToolTip(__("{title}に記録されているモデル名です。\n文字列は選択およびコピー可能です。", title=self.title))
            self.title_sizer.Add(self.name_ctrl, 1, wx.EXPAND | wx.ALL, 3)

        self.root_sizer.Add(self.title_sizer, 0, wx.ALL, 3)

        # ------------------------------
        # ファイルコントロール
        self.file_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # 読み取りか保存かでスタイルを変える
        file_ctrl_style = wx.FLP_DEFAULT_STYLE if not self.is_save else wx.FLP_OVERWRITE_PROMPT | wx.FLP_SAVE | wx.FLP_USE_TEXTCTRL
        self.file_ctrl = wx.FilePickerCtrl(
            self.parent,
            wx.ID_ANY,
            path=wx.EmptyString,
            wildcard=self.reader.file_ext,
            style=file_ctrl_style,
        )
        self.file_ctrl.GetPickerCtrl().SetLabel(__("開く"))
        self.file_ctrl.SetToolTip(__(tooltip))
        if self.key and self.frame.histories[self.key]:
            self.file_ctrl.SetInitialDirectory(os.path.dirname(self.frame.histories[self.key][0]))

        self.file_sizer.Add(self.file_ctrl, 1, wx.GROW | wx.ALL, 3)

        if not self.is_save:
            # 保存じゃなければ履歴ボタンを表示
            self.history_ctrl = wx.Button(
                self.parent,
                wx.ID_ANY,
                label=__("履歴"),
            )
            self.history_ctrl.SetToolTip(__("これまでに指定された事のある{title}を再指定することができます。", title=self.title))
            self.file_sizer.Add(self.history_ctrl, 0, wx.ALL, 3)

        self.root_sizer.Add(self.file_sizer, 0, wx.GROW | wx.ALL, 0)

    def _initialize_event(self):
        # D&Dの実装
        self.file_ctrl.SetDropTarget(MFileDropTarget(self))

        if not self.is_save:
            self.history_ctrl.Bind(wx.EVT_BUTTON, self.on_show_histories)

        if self.file_change_event:
            self.file_ctrl.Bind(wx.EVT_FILEPICKER_CHANGED, self.file_change_event)

    def on_show_histories(self, event: wx.Event):
        """履歴一覧を表示する"""
        if not self.key:
            return

        histories = self.frame.histories[self.key] + [" " * 200]

        with wx.SingleChoiceDialog(
            self.frame,
            __("ファイルを選んでダブルクリック、またはOKボタンをクリックしてください。"),
            caption=__("ファイル履歴選択"),
            choices=histories,
            style=wx.CAPTION | wx.CLOSE_BOX | wx.SYSTEM_MENU | wx.OK | wx.CANCEL | wx.CENTRE,
        ) as choiceDialog:
            if choiceDialog.ShowModal() == wx.ID_CANCEL:
                return

            # ファイルピッカーに選択したパスを設定
            self.file_ctrl.SetPath(choiceDialog.GetStringSelection())
            self.file_change_event(wx.FileDirPickerEvent())
            self.file_ctrl.UpdatePickerFromTextCtrl()
            self.file_ctrl.SetInitialDirectory(get_dir_path(choiceDialog.GetStringSelection()))

    @property
    def path(self):
        return self.file_ctrl.GetPath()

    @path.setter
    def path(self, v: str):
        if (not self.is_save and validate_file(v, self.reader.file_type)) or (self.is_save and validate_save_file(v, self.title)):
            self.file_ctrl.SetPath(v)

    def unwrap(self):
        self.file_ctrl.SetPath(unwrapped_path(self.file_ctrl.GetPath()))

    def save_path(self):
        if not self.key:
            return
        insert_history(self.file_ctrl.GetPath(), self.frame.histories[self.key])

    def read_name(self) -> bool:
        """
        リーダー対象オブジェクトの名前を読み取る

        Returns
        -------
        bool
            読み取り出来るパスか否か
        """
        if not self.file_ctrl.GetPath():
            self.name_ctrl.SetValue(__("(未設定)"))
            return False

        if self.is_show_name and not self.is_save:
            if validate_file(self.file_ctrl.GetPath(), self.reader.file_type):
                name = self.reader.read_name_by_filepath(self.file_ctrl.GetPath())
                self.name_ctrl.SetValue(f"({name[:20]})")
                return True
        self.name_ctrl.SetValue(__("(読取失敗)"))
        return False

    def read_digest(self):
        """リーダー対象オブジェクトのハッシュを読み取る"""
        if self.is_show_name and not self.is_save and validate_file(self.file_ctrl.GetPath(), self.reader.file_type):
            digest = self.reader.read_hash_by_filepath(self.file_ctrl.GetPath())
            if self.data and self.data.digest != digest:
                # 読み取り対象データが変わっている場合、オブジェクトをクリアしておく
                self.data = None


class MFileDropTarget(wx.FileDropTarget):
    def __init__(self, parent: MFilePickerCtrl):
        self.parent = parent

        wx.FileDropTarget.__init__(self)

    def OnDropFiles(self, x, y, files):
        if validate_file(files[0], self.parent.reader.file_type):
            # 拡張子を許容してたらOK
            self.parent.file_ctrl.SetPath(files[0])

            # ファイル変更処理
            self.parent.file_change_event(wx.FileDirPickerEvent())

            return True

        # TODO アスタリスクの許容
        # # アスタリスクOKの場合、フォルダの投入を許可する
        # if os.path.isdir(files[0]) and self.is_aster:
        #     # フォルダを投入された場合、フォルダ内にvmdもしくはvpdがあれば、受け付ける
        #     child_file_name_exts = [os.path.splitext(filename) for filename in os.listdir(files[0]) if os.path.isfile(os.path.join(files[0], filename))]

        #     for ft in self.parent.file_type:
        #         # 親の許容ファイルパス
        #         for child_file_name, child_file_ext in child_file_name_exts:
        #             if child_file_ext[1:].lower() == ft:
        #                 # 子のファイル拡張子が許容拡張子である場合、アスタリスクを入れて許可する
        #                 astr_path = "{0}\\*.{1}".format(files[0], ft)
        #                 self.parent.file_ctrl.SetPath(astr_path)

        #                 # ファイル変更処理
        #                 self.parent.on_change_file(wx.FileDirPickerEvent())

        #                 return True

        logger.warning(
            "{file_title}に入力されたファイル拡張子を受け付けられませんでした。{file_type}拡張子のファイルを入力してください。\n入力ファイルパス: {file_path}",
            decoration=MLogger.Decoration.BOX,
            file_title=self.parent.title,
            file_type=self.parent.reader.file_type.name.lower(),
            file_path=files[0],
        )

        # 許容拡張子外の場合、不許可
        return False
