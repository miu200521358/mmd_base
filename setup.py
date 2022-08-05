import os
from glob import glob

from Cython.Build import cythonize
from Cython.Distutils import build_ext

# fmt: off
# cimport numpy を使うため
from numpy import get_include  # type: ignore
from setuptools import Extension, setup

# fmt: on


bezier_path = "C:/Development/Anaconda3/envs/mmd_tool/Lib/site-packages/bezier/include"


def get_ext():
    ext = []

    for source in glob("mlib\\**\\*.py", recursive=True):
        if "__init__" in source:
            continue
        path = source.replace("\\", ".").replace(".py", "")
        print("%s -> %s" % (source, path))
        ext.append(
            Extension(
                path,
                sources=[source],
                include_dirs=[".", bezier_path, get_include()],
                # define_macros=[("NPY_NO_DEPRECATED_API")],
            )
        )

    return ext


for source in glob("mlib\\**\\*.pyd", recursive=True):
    print(f"remove {source}")
    os.remove(source)

print("----------------")

setup(
    name="*",
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(
        get_ext(),
        compiler_directives={"language_level": "3"},
        **{"output_dir": "./build/output", "build_dir": "./build/"},
    ),
)
