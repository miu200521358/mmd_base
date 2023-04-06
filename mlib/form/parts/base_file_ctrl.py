import wx
from mlib.base.logger import MLogger
from mlib.base.reader import BaseReader

logger = MLogger(__name__)
_ = logger.get_text


class BaseFilePickerCtrl:
    def __init__(self, parent, reader: BaseReader, title: str, is_show_name: bool, name_spacer: int, is_save: bool, tooltip: str) -> None:
        self.parent = parent
        self.reader = reader
        self.root_sizer = wx.BoxSizer(wx.VERTICAL)

        self._initialize_ui(title, is_show_name, name_spacer, is_save, tooltip)

    def set_parent_sizer(self, parent_sizer: wx.Sizer):
        parent_sizer.Add(self.root_sizer, 1, wx.EXPAND | wx.ALL, 0)

    def _initialize_ui(self, title: str, is_show_name: bool, name_spacer: int, is_save: bool, tooltip: str):
        # ファイルタイトル
        self.title_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.title_ctrl = wx.StaticText(self.parent, wx.ID_ANY, title, wx.DefaultPosition, wx.DefaultSize, 0)
        self.title_sizer.Add(self.title_ctrl, 1, wx.EXPAND | wx.ALL, 3)

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

        self.file_sizer.Add(self.file_ctrl, 1, wx.EXPAND | wx.ALL, 3)

        if not is_save:
            # 保存じゃなければ履歴ボタンを表示
            self.history_ctrl = wx.Button(
                self.parent,
                wx.ID_ANY,
                label=_("履歴"),
            )
            self.file_sizer.Add(self.history_ctrl, 0, wx.ALL, 3)

        self.root_sizer.Add(self.file_sizer, 0, wx.EXPAND | wx.ALL, 0)
