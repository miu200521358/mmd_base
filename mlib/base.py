import _pickle as cPickle  # type: ignore
from mlib.logger import parse2str


class BaseModel:
    """基底クラス"""

    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return parse2str(self)

    def copy(self):
        return cPickle.loads(cPickle.dumps(self, -1))
