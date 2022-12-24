def read_model():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from mlib.pmx.pmx_collection import PmxModel
    from mlib.pmx.pmx_reader import PmxReader
    from mlib.pmx.pmx_writer import PmxWriter

    reader = PmxReader()
    model: PmxModel = reader.read_by_filepath(
        # "D:\\MMD\\MikuMikuDance_v926x64\\UserFile\\Model\\刀剣乱舞\\025_一期一振\\一期一振 ちゃむ式 20211211\\01_10_極_一期_ちゃむ20211211.pmx",
        "E:/MMD/MikuMikuDance_v926x64/Work/202101_vroid/_報告/ラワイル1223/APmiku_nakedhair_IKx_onlyee.pmx",
    )
    print(model.name)

    PmxWriter().write(
        model,
        "E:/MMD/MikuMikuDance_v926x64/Work/202101_vroid/_報告/ラワイル1223/APmiku_nakedhair_IKx_onlyee_output.pmx",
    )


def read_motion():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from mlib.vmd.vmd_collection import VmdMotion
    from mlib.vmd.vmd_reader import VmdReader

    reader = VmdReader()
    motion: VmdMotion = reader.read_by_filepath(
        # "D:\\MMD\\MikuMikuDance_v926x64\\UserFile\\Motion\\ダンス_1人\\ジャンキーナイトタウンオーケストラ 粉ふきスティック\\janky_N_O.vmd"  # noqa
        "D:\\MMD\\MikuMikuDance_v926x64\\UserFile\\Motion\\ダンス_1人\\ヒアソビ mobiusP\\HIASOBI_motion.vmd"  # noqa
    )
    print(motion.model_name)


if __name__ == "__main__":
    read_model()
