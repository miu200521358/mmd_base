import multiprocessing as mp
from typing import Callable


class DrawProcess(mp.Process):
    """wxPythonによる画面描画を担当するプロセス"""

    def __init__(self, target: Callable[..., object], queues: list[mp.Queue]) -> None:
        super().__init__(target=target, args=queues)

    pass


class GlProcess(mp.Process):
    """OpenGLを担当するプロセス"""

    def __init__(self, target: Callable[..., object], queues: list[mp.Queue]) -> None:
        super().__init__(target=target, args=queues)

    pass


class ComputeProcess(mp.Process):
    """計算処理を担当するプロセス"""

    def __init__(self, target: Callable[..., object], queues: list[mp.Queue]) -> None:
        super().__init__(target=target, args=queues)

    pass
