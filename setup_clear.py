from glob import glob
import os

for source in glob("mlib\\**\\*.pyd", recursive=True):
    print(f"remove {source}")
    os.remove(source)
