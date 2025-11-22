from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

kernel_sources = (
    glob.glob("src/*.cpp") +
    glob.glob("src/**/*.cpp") +
    glob.glob("src/*.cu") +
    glob.glob("src/**/*.cu")
)

setup(
    name="my_kernels",
    ext_modules=[
        CUDAExtension(
            "my_kernels",
            kernel_sources,
            include_dirs=["include"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
