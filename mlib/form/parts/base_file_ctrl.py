import wx
from mlib.base.logger import MLogger
from mlib.base.reader import BaseReader
from mlib.form.base_frame import BaseFrame
from mlib.form.base_panel import BasePanel
from mlib.utils.file_utils import HISTORY_MAX, get_dir_path


logger = MLogger(__name__)
_ = logger.get_text


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
    ) -> None:
        self.frame = frame
        self.parent = parent
        self.reader = reader
        self.key = key
        self.root_sizer = wx.BoxSizer(wx.VERTICAL)

        self._initialize_ui(title, is_show_name, name_spacer, is_save, tooltip)
        self._initialize_event(is_save)

    def set_parent_sizer(self, parent_sizer: wx.Sizer):
        parent_sizer.Add(self.root_sizer, 1, wx.GROW | wx.TOP | wx.LEFT | wx.RIGHT, 0)

    def _initialize_ui(self, title: str, is_show_name: bool, name_spacer: int, is_save: bool, tooltip: str):
        # ファイルタイトル
        self.title_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.title_ctrl = wx.StaticText(self.parent, wx.ID_ANY, title, wx.DefaultPosition, wx.DefaultSize, 0)
        self.title_ctrl.SetToolTip(_(tooltip))
        self.title_sizer.Add(self.title_ctrl, 1, wx.GROW | wx.ALL, 3)

        # モデル名等の表示
        if is_show_name:
            self.spacer_ctrl = wx.StaticText(self.parent, wx.ID_ANY, " " * name_spacer, wx.DefaultPosition, wx.DefaultSize, 0)
            self.title_sizer.Add(self.spacer_ctrl, 0, wx.ALL, 3)

            self.name_ctrl = wx.TextCtrl(
                self.parent,
                wx.ID_ANY,
                _("未設定"),
                wx.DefaultPosition,
                wx.DefaultSize,
                wx.TE_READONLY | wx.BORDER_NONE | wx.WANTS_CHARS,
            )
            self.name_ctrl.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DLIGHT))
            self.name_ctrl.SetToolTip(_("{title}に記録されているモデル名です。\n文字列は選択およびコピー可能です。", title=title))
            self.title_sizer.Add(self.name_ctrl, 0, wx.ALL, 3)

        self.root_sizer.Add(self.title_sizer, 0, wx.ALL, 3)

        # ------------------------------
        # ファイルコントロール
        self.file_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # 読み取りか保存かでスタイルを変える
        file_ctrl_style = wx.FLP_DEFAULT_STYLE if not is_save else wx.FLP_OVERWRITE_PROMPT | wx.FLP_SAVE | wx.FLP_USE_TEXTCTRL
        self.file_ctrl = wx.FilePickerCtrl(
            self.parent,
            wx.ID_ANY,
            path=wx.EmptyString,
            wildcard=self.reader.file_type,
            style=file_ctrl_style,
        )
        self.file_ctrl.GetPickerCtrl().SetLabel(_("開く"))
        self.file_ctrl.SetToolTip(_(tooltip))

        self.file_sizer.Add(self.file_ctrl, 1, wx.GROW | wx.ALL, 3)

        if not is_save:
            # 保存じゃなければ履歴ボタンを表示
            self.history_ctrl = wx.Button(
                self.parent,
                wx.ID_ANY,
                label=_("履歴"),
            )
            self.history_ctrl.SetToolTip(_("これまでに指定された事のある{title}を再指定することができます。", title=title))
            self.file_sizer.Add(self.history_ctrl, 0, wx.ALL, 3)

        self.root_sizer.Add(self.file_sizer, 0, wx.GROW | wx.ALL, 0)

    def _initialize_event(self, is_save: bool):
        if not is_save:
            self.history_ctrl.Bind(wx.EVT_BUTTON, self.on_show_histories)

    def on_show_histories(self, event: wx.Event):
        """履歴一覧を表示する"""
        histories = (self.frame.histories[self.key] + [" " * 200])[:HISTORY_MAX]

        with wx.SingleChoiceDialog(
            self.frame,
            _("ファイルを選んでダブルクリック、またはOKボタンをクリックしてください。"),
            caption=_("ファイル履歴選択"),
            choices=histories,
            style=wx.CAPTION | wx.CLOSE_BOX | wx.SYSTEM_MENU | wx.OK | wx.CANCEL | wx.CENTRE,
        ) as choiceDialog:
            if choiceDialog.ShowModal() == wx.ID_CANCEL:
                return

            # ファイルピッカーに選択したパスを設定
            self.file_ctrl.SetPath(choiceDialog.GetStringSelection())
            self.file_ctrl.UpdatePickerFromTextCtrl()
            self.file_ctrl.SetInitialDirectory(get_dir_path(choiceDialog.GetStringSelection()))

            # ファイル変更処理
            self.on_change_file(wx.FileDirPickerEvent())

    def on_change_file(self, event: wx.Event):
        pass
