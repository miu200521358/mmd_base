import os
import shutil
from glob import glob

shutil.rmtree("build")

for source in glob("mlib\\**\\*.pyd", recursive=True):
    print(f"remove {source}")
    os.remove(source)
