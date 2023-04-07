import os
from threading import Thread, enumerate, current_thread
from functools import wraps
from time import sleep, time
from typing import Any, Callable, Optional
import wx
from mlib.base.exception import MLibException
from mlib.base.logger import MLogger

from mlib.form.base_panel import BasePanel

logger = MLogger(os.path.basename(__file__))
__ = logger.get_text


# https://doloopwhile.hatenablog.com/entry/20090627/1275175850
class SimpleThread(Thread):
    """呼び出し可能オブジェクト（関数など）を実行するだけのスレッド"""

    def __init__(self, base_thread, callable):
        self.base_thread = base_thread
        self.callable = callable
        self._result = None
        self.killed = False
        super(SimpleThread, self).__init__(name="simple_thread")

    def run(self):
        self._result = self.callable(self.base_thread)

    def result(self):
        return self._result


def task_takes_time(callable: Callable):
    """
    callable本来の処理は別スレッドで実行しながら、
    ウィンドウを更新するwx.YieldIfNeededを呼び出し続けるようにする
    """

    @wraps(callable)
    def f(worker: "BaseWorker"):
        thread = SimpleThread(worker, callable)
        thread.daemon = True
        thread.start()
        while thread.is_alive():
            wx.YieldIfNeeded()
            sleep(0.01)

            if worker.killed:
                # 呼び出し元から停止命令が出ている場合、自分以外の全部のスレッドに終了命令
                for th in enumerate():
                    if isinstance(th, SimpleThread) and th.ident != current_thread().ident:
                        th.killed = True
                break

        return thread.result()

    return f


def show_worked_time(elapsed_time: float):
    """経過秒数を時分秒に変換"""
    td_m, td_s = divmod(elapsed_time, 60)
    td_h, td_m = divmod(td_m, 60)

    if td_m == 0:
        worked_time = "00:00:{0:02d}".format(int(td_s))
    elif td_h == 0:
        worked_time = "00:{0:02d}:{1:02d}".format(int(td_m), int(td_s))
    else:
        worked_time = "{0:02d}:{1:02d}:{2:02d}".format(int(td_h), int(td_m), int(td_s))

    return worked_time


class BaseWorker:
    def __init__(self, panel: BasePanel, result_func: Callable) -> None:
        self.start_time = 0.0
        self.panel = panel
        self.killed = False
        self.result: bool = True
        self.result_data: Optional[Any] = None
        self.result_func = result_func

    def start(self):
        self.start_time = time()

        self.run()

    def stop(self):
        self.killed = True

    @task_takes_time
    def run(self):
        try:
            self.thread_execute()
        except MLibException:
            logger.error(__("処理を中断しました"))
            self.result = False
        except Exception:
            logger.critical(__("予期せぬエラーが発生しました"))
            self.result = False
        finally:
            self.result_func(result=self.result, data=self.result_data, elapsed_time=show_worked_time(time() - self.start_time))

    def thread_execute(self):
        raise NotImplementedError
