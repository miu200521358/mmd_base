import glob
import os

cwd = os.getcwd()

py_list = glob.glob("mlib/**/*.py", recursive=True)
import_list = [f.replace(".py", "") for f in py_list if "__init__.py" not in f]


__all__ = import_list
