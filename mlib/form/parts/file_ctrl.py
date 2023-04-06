import wx
from mlib.base.logger import MLogger
from mlib.base.reader import BaseReader
from mlib.form.base_frame import BaseFrame
from mlib.form.base_panel import BasePanel
from mlib.utils.file_utils import HISTORY_MAX, get_dir_path, validate_file, validate_save_file


logger = MLogger(__name__)
__ = logger.get_text


class MFilePickerCtrl:
    def __init__(
        self,
        frame: BaseFrame,
        parent: BasePanel,
        reader: BaseReader,
        key: str,
        title: str,
        is_show_name: bool = True,
        name_spacer: int = 0,
        is_save: bool = False,
        tooltip: str = "",
        event=None,
    ) -> None:
        self.frame = frame
        self.parent = parent
        self.reader = reader
        self.key = key
        self.title = title
        self.is_save = is_save
        self.is_show_name = is_show_name
        self.root_sizer = wx.BoxSizer(wx.VERTICAL)

        self._initialize_ui(name_spacer, tooltip)
        self._initialize_event(event)

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
            self.title_sizer.Add(self.name_ctrl, 0, wx.ALL, 3)

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

    def _initialize_event(self, event):
        if not self.is_save:
            self.history_ctrl.Bind(wx.EVT_BUTTON, self.on_show_histories)

        if event:
            self.file_ctrl.Bind(wx.EVT_FILEPICKER_CHANGED, event)

    def on_show_histories(self, event: wx.Event):
        """履歴一覧を表示する"""
        histories = (self.frame.histories[self.key] + [" " * 200])[:HISTORY_MAX]

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
            self.file_ctrl.UpdatePickerFromTextCtrl()
            self.file_ctrl.SetInitialDirectory(get_dir_path(choiceDialog.GetStringSelection()))

    @property
    def path(self):
        return self.file_ctrl.GetPath()

    @path.setter
    def path(self, v: str):
        if (not self.is_save and validate_file(v, self.reader.file_type)) or (self.is_save and validate_save_file(v, self.title)):
            self.file_ctrl.SetPath(v)

    def read_name(self):
        if self.is_show_name and not self.is_save:
            if validate_file(self.file_ctrl.GetPath(), self.reader.file_type):
                name = self.reader.read_name_by_filepath(self.file_ctrl.GetPath()) or __("読取失敗")
            else:
                name = __("読取失敗")
            self.name_ctrl.SetValue(f"({name[:20]})")
