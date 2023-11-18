import glob
import os

root_dir = "C:/development/anaconda3/envs/mtool/Lib/site-packages"

with open(
    "C:/MMD/mmd_base/chatgpt/3dcg_mentor/library.txt", "w", encoding="utf-8"
) as f:
    f.write("このファイルには、主に使われるライブラリのPythonコードをまとめて記載しています\n\n")

    for library_dir in ["numpy", "OpenGL", "PIL", "wx", "bezier", "quaternion"]:
        root_py_dir = f"{root_dir}/{library_dir}/*.py"
        root_pyx_dir = f"{root_dir}/{library_dir}/*.pyx"
        root_pyd_dir = f"{root_dir}/{library_dir}/*.pyd"

        for file_path in (
            glob.glob(root_py_dir, recursive=True)
            + glob.glob(root_pyd_dir, recursive=True)
            + glob.glob(root_pyx_dir, recursive=True)
        ):
            if "__init__" in file_path:
                continue

            rel_file_path = os.path.relpath(file_path, ".")
            print(rel_file_path)

            f.write("----------------------------- \n")
            f.write(rel_file_path)
            f.write("\n\n")

            try:
                with open(file_path, "r", encoding="utf-8") as pf:
                    f.write(pf.read())
                    f.write("\n\n")
            except Exception:
                continue

print("■ FINISH ■")
