import numpy as np


def parse2str(obj: object) -> str:
    """オブジェクトの変数の名前と値の一覧を文字列で返す

    Parameters
    ----------
    obj : object

    Returns
    -------
    str
        変数リスト文字列
        Sample[x=2, a=sss, child=ChildSample[y=4.5, b=xyz]]
    """
    return f"{obj.__class__.__name__}[{', '.join([f'{k}={v}' for k, v in vars(obj).items()])}]"


def parse_str(v: object) -> str:
    """
    丸め処理付き文字列変換処理

    小数だったら丸めて一定桁数までしか出力しない
    """
    decimals = 5
    if isinstance(v, float):
        return f"{round(v, decimals)}"
    elif isinstance(v, np.ndarray):
        return f"{np.round(v, decimals)}"
    elif hasattr(v, "data"):
        return f"{np.round(v.__getattribute__('data'), decimals)}"
    else:
        return f"{v}"
