import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mlib.pmx.pmx_reader import PmxReader
from mlib.vmd.vmd_collection import VmdMotion
from mlib.vmd.vmd_part import VmdMorphFrame

# 全体プロファイル
# python -m cProfile -s cumtime example\profile3.py
# 行プロファイル
# kernprof -l example\profile3.py
# python -m line_profiler profile3.py.lprof

# model = PmxReader().read_by_filepath("D:/MMD/MikuMikuDance_v926x64/UserFile/Model/VOCALOID/初音ミク/Lat式ミクVer2.31/Lat式ミクVer2.31_Normal_準標準.pmx")
# motion = VmdReader().read_by_filepath("D:/MMD/MikuMikuDance_v926x64/UserFile/Motion/ダンス_1人/好き雪本気マジック_モーション hino/好き雪本気マジック_Lat式.vmd")

# 時間計測開始
start_time = time.perf_counter()

model = PmxReader().read_by_filepath("E:/MMD/less検証/Test-Dressup/wa_20220424_high-peti3mod_cloth/和洋折衷_20230422_075212.pmx")
motion = VmdMotion()

# フィッティングモーフは常に適用
bmf = VmdMorphFrame(0, "BoneFitting")
bmf.ratio = 1
motion.morphs[bmf.name].append(bmf)

vmf = VmdMorphFrame(0, "VertexFitting")
vmf.ratio = 1
motion.morphs[vmf.name].append(bmf)

results = motion.animate(0, model)

# # キーフレ
# bone_trees = model.bone_trees.gets(["左手首", "右手首"])
# bone_matrixes = motion.bones.get_matrix_by_indexes(list(range(0, 300)), bone_trees, model)

# 時間計測終了
end_time = time.perf_counter()

print(end_time - start_time)
