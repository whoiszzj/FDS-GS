from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

os.path.dirname(os.path.abspath(__file__))

setup(
    name="confidence_filter",
    packages=['confidence_filter'],
    ext_modules=[
        CUDAExtension(
            name="confidence_filter._C",
            sources=[
                "compute_confidence_impl.cu",
                "heap_sort_impl.cu",
                "confidence_filter.cpp",
                "ext.cpp"],
            extra_compile_args={
                "nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]
            })
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
