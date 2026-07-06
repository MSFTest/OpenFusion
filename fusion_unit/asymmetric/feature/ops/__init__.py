import importlib
import pkgutil
import pathlib

_pkg_dir = pathlib.Path(__file__).parent

for module in pkgutil.iter_modules([str(_pkg_dir)]):
    name = module.name
    if name.startswith("_"):
        continue
    importlib.import_module(f"{__name__}.{name}")