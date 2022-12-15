import os
from glob import glob

from Cython.Build import cythonize
from Cython.Distutils import build_ext
from numpy import get_include
from setuptools import Extension, setup

bezier_path = "C:/Development/Anaconda3/envs/mmd_base/Lib/site-packages/bezier/include"

kwargs = {"output_dir": "./build/output", "build_dir": "./build/"}


def get_ext():
    ext = []
    sources = [
        path
        for path in glob("mlib/**/*.py", recursive=True)
        if "__init__" not in path and os.path.isfile(path)
    ]

    for source in sources:
        path = (
            source.replace("/", ".")
            .replace("\\", ".")
            .replace(".pyx", "")
            .replace(".py", "")
        )
        print("%s -> %s" % (source, path))
        ext.append(
            Extension(
                path,
                sources=[source],
                include_dirs=[".", bezier_path, get_include()],
                define_macros=[],
            )
        )

    return ext


setup(
    name="*",
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(get_ext(), **kwargs),
)
