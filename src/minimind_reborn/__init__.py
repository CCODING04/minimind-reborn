"""minimind_reborn：MiniMind 的工程化重构训练项目。"""

from importlib.metadata import PackageNotFoundError, version

# 版本单一来源是 pyproject.toml（python-style §1.3）；此处仅派生读取
__version__ = version("minimind-reborn")

__all__ = ["__version__"]
