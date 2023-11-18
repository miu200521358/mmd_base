import glob
import os

root_py_dir = "C:/MMD/mmd_base/mlib/**/*.py"
root_vert_dir = "C:/MMD/mmd_base/mlib/**/*.vert"
root_frag_dir = "C:/MMD/mmd_base/mlib/**/*.frag"

with open("C:/MMD/mmd_base/chatgpt/3dcg_mentor/mlib.txt", "w", encoding="utf-8") as f:
    f.write("このファイルには mlib ライブラリのPythonコードをまとめて記載しています\n\n")

    for file_path in (
        glob.glob(root_py_dir, recursive=True)
        + glob.glob(root_vert_dir, recursive=True)
        + glob.glob(root_frag_dir, recursive=True)
    ):
        if "__init__" in file_path:
            continue

        rel_file_path = os.path.relpath(file_path, ".")
        f.write("----------------------------- \n")
        f.write(rel_file_path)
        f.write("\n\n")

        with open(file_path, "r", encoding="utf-8") as pf:
            f.write(pf.read())
            f.write("\n\n")

        print(rel_file_path)

    print("■ FINISH ■")
