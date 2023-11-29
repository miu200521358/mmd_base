import os
import sys
import time
from multiprocessing import freeze_support

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mlib.pmx.pmx_reader import PmxReader  # noqa: E402
from mlib.vmd.vmd_reader import VmdReader  # noqa: E402

# 全体プロファイル
# python -m cProfile -s cumtime crumb\profile_animate.py
# 行プロファイル
# kernprof -l crumb\profile_animate.py
# python -m line_profiler profile_animate.py.lprof

# model = PmxReader().read_by_filepath("D:/MMD/MikuMikuDance_v926x64/UserFile/Model/VOCALOID/初音ミク/Lat式ミクVer2.31/Lat式ミクVer2.31_Normal_準標準.pmx")
# motion = VmdReader().read_by_filepath("D:/MMD/MikuMikuDance_v926x64/UserFile/Motion/ダンス_1人/好き雪本気マジック_モーション hino/好き雪本気マジック_Lat式.vmd")


def main() -> None:
    # model = PmxReader().read_by_filepath("D:/MMD/MikuMikuDance_v926x64/UserFile/Model/VOCALOID/初音ミク/ISAO式ミク/I_ミクv4/Miku_V4.pmx")
    # motion = VmdReader().read_by_filepath("D:/MMD/MikuMikuDance_v926x64/UserFile/Motion/ダンス_1人/テレキャスタービーボーイ 粉ふきスティック/TeBeboy.vmd")
    model = PmxReader().read_by_filepath(
        "D:/MMD/MikuMikuDance_v926x64/UserFile/Model/VOCALOID/初音ミク/どっと式初音ミク_ハニーウィップ_ver.2.01/どっと式初音ミク_ハニーウィップ.pmx"
    )
    motion = VmdReader().read_by_filepath(
        "D:/MMD/MikuMikuDance_v926x64/UserFile/Motion/ダンス_1人/ドクヘビ mobiusP/ドクヘビLight.vmd"
    )

    # # モーフ追加
    # morph = Morph(name="上半身")
    # morph.morph_type = MorphType.BONE
    # offset = BoneMorphOffset(model.bones["上半身"].index, MVector3D(), MQuaternion())
    # offset.local_position = MVector3D(0, 1, 0)
    # offset.local_rotation.qq = MQuaternion.from_euler_degrees(10, 0, 0)
    # offset.local_scale = MVector3D(0, 1.5, 1.5)
    # morph.offsets.append(offset)
    # model.morphs.append(morph)

    # motion.morphs["上半身"].append(VmdMorphFrame(0, "上半身", 1))

    # 時間計測開始
    start_time = time.perf_counter()

    # max_worker = 1
    max_worker = 2
    # max_worker = 4
    # max_worker = max(1, int(min(32, (os.cpu_count() or 0) + 4) / 4))
    print(f"max_worker: {max_worker}")

    motion.animate_bone(
        list(range(1000, 1500)),
        model,
        out_fno_log=True,
        max_worker=max_worker,
    )

    # # キーフレ
    # bone_trees = model.bone_trees.gets(["左手首", "右手首"])
    # bone_matrixes = motion.bones.get_matrix_by_indexes(list(range(0, 300)), bone_trees, model)

    # 時間計測終了
    end_time = time.perf_counter()

    print(end_time - start_time)


if __name__ == "__main__":
    freeze_support()

    main()
