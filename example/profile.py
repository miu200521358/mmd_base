import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mlib.pmx.collection import PmxModel
from mlib.pmx.reader import PmxReader
from mlib.vmd.collection import VmdMotion
from mlib.vmd.reader import VmdReader

vmd_reader = VmdReader()
motion: VmdMotion = vmd_reader.read_by_filepath(
    os.path.join("tests", "resources", "サンプルモーション.vmd")
)

pmx_reader = PmxReader()
model: PmxModel = pmx_reader.read_by_filepath(
    os.path.join("tests", "resources", "サンプルモデル.pmx")
)

# 時間計測開始
stime = time.perf_counter()

# キーフレ
bone_trees = model.bone_trees.gets(["左手首", "右手首"])
bone_matrixes = motion.bones.get_matrix_by_indexes(
    list(range(0, 100)), bone_trees, model
)

# 時間計測終了
etime = time.perf_counter()

print(etime - stime)
