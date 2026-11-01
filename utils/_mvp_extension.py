"""只加载/显式编译原CenterPoint IoU/NMS扩展；无几何近似fallback。"""
from functools import lru_cache
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'model/MVP/artifacts/extensions/mvp_iou3d_nms'
@lru_cache(None)
def load_extension():
    paths=list(BUILD.glob('mvp_iou3d_nms*.so'))
    path=paths[0] if len(paths)==1 else BUILD/'mvp_iou3d_nms.so'
    if not path.is_file():raise FileNotFoundError('缺少原MVP CUDA扩展：%s；执行test_MVP.py --mode prepare --prepare-extension'%path)
    spec=importlib.util.spec_from_file_location('mvp_iou3d_nms',str(path));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def build_extension():
    """显式构建原源代码，产物仅在当前项目；需nvcc及当前torch CUDA工具链。"""
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension,CUDAExtension
    src=ROOT/'utils/_mvp_iou3d_nms_src'
    BUILD.mkdir(parents=True,exist_ok=True)
    setup(name='mvp_iou3d_nms',script_args=['build_ext','--build-lib',str(BUILD),'--build-temp',str(BUILD/'objects')],ext_modules=[CUDAExtension('mvp_iou3d_nms',[str(src/f) for f in ['iou3d_cpu.cpp','iou3d_nms_api.cpp','iou3d_nms.cpp','iou3d_nms_kernel.cu']],extra_compile_args={'cxx':['-O2'],'nvcc':['-O2']})],cmdclass={'build_ext':BuildExtension.with_options(use_ninja=False)})
    load_extension.cache_clear()
    return load_extension()
