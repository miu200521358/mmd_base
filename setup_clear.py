from glob import glob
import os
import shutil

shutil.rmtree("build")

for source in glob("mlib\\**\\*.pyd", recursive=True):
    print(f"remove {source}")
    os.remove(source)
