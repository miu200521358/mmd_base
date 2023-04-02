import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mlib.pmx.pmx_reader import PmxReader
from mlib.vmd.vmd_reader import VmdReader

model = PmxReader().read_by_filepath("D:/MMD/MikuMikuDance_v926x64/UserFile/Model/VOCALOID/初音ミク/Lat式ミクVer2.31/Lat式ミクVer2.31_Normal_準標準.pmx")
motion = VmdReader().read_by_filepath("D:/MMD/MikuMikuDance_v926x64/UserFile/Motion/ダンス_1人/好き雪本気マジック_モーション hino/好き雪本気マジック_Lat式.vmd")

# 時間計測開始
start_time = time.perf_counter()

for fno in range(1000, 1010):
    results = motion.animate(fno, model)

# # キーフレ
# bone_trees = model.bone_trees.gets(["左手首", "右手首"])
# bone_matrixes = motion.bones.get_matrix_by_indexes(list(range(0, 300)), bone_trees, model)

# 時間計測終了
end_time = time.perf_counter()

print(end_time - start_time)
