import glob

import_list = [
    f.replace(".py", "")
    for f in glob.glob("mlib/**/*.py", recursive=True)
    if "__init__.py" not in f
]

print(import_list)

__all__ = import_list
