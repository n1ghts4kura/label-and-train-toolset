"""
config_manager.py
────────────────────────────────────────────────────────────────
项目配置管理模块

职责：
- 加载/切换项目配置 (configs/projects/*.yaml)
- 生成 dataset.yaml（供 YOLOv5 使用）
- 提供全局配置访问接口

使用方式:
    from core.config_manager import ConfigManager, get_config

    # 获取单例配置管理器
    cfg = get_config()

    # 切换项目配置
    cfg.switch_project("my_project")

    # 获取当前项目信息
    print(cfg.project_name)
    print(cfg.classes)
"""

import os
import re
from pathlib import Path
from typing import Optional

import yaml


class ConfigManager:
    """
    配置管理器，支持多项目切换

    project.yaml 结构:
        project_name: "my_project"
        classes:
          - name: "black_bold_box"
            id: 0
        data_root: "./train_data"
        origin_dir: "./origin_pics"
        isat_yaml: "./origin_pics/isat.yaml"
        train_val_ratio: 10
    """

    _instance: Optional["ConfigManager"] = None

    def __init__(self, project_file: Optional[Path] = None):
        self._project_file: Optional[Path] = None
        self._config: dict = {}
        self._project_root: Path = Path(__file__).parent.parent

        if project_file:
            self.load(project_file)

    @classmethod
    def get_instance(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self, project_file: Path) -> dict:
        """
        加载指定的 project.yaml 文件
        """
        if not project_file.is_absolute():
            project_file = self._project_root / project_file

        if not project_file.exists():
            raise FileNotFoundError(f"项目配置文件不存在: {project_file}")

        with open(project_file, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}

        self._project_file = project_file
        return self._config

    def switch_project(self, project_name: str) -> dict:
        """
        通过项目名称切换配置（查找 configs/projects/{project_name}.yaml）
        """
        project_file = self._project_root / "configs" / "projects" / f"{project_name}.yaml"
        if not project_file.exists():
            # 尝试直接作为完整路径
            project_file = Path(project_name)
        return self.load(project_file)

    def switch_project_by_path(self, project_file: Path) -> dict:
        """
        通过文件路径切换项目配置
        """
        return self.load(project_file)

    @property
    def project_name(self) -> str:
        return self._config.get("project_name", "default")

    @property
    def classes(self) -> list[dict]:
        """返回类别列表 [{'name': 'xxx', 'id': 0}, ...]"""
        return self._config.get("classes", [])

    @property
    def class_map(self) -> dict[str, int]:
        """返回类别名称到 ID 的映射 {'name': id}"""
        return {c["name"]: c["id"] for c in self.classes}

    @property
    def data_root(self) -> Path:
        root = self._config.get("data_root", "./train_data")
        if not Path(root).is_absolute():
            root = self._project_root / root
        return Path(root)

    @property
    def origin_dir(self) -> Path:
        origin = self._config.get("origin_dir", "./origin_pics")
        if not Path(origin).is_absolute():
            origin = self._project_root / origin
        return Path(origin)

    @property
    def isat_yaml(self) -> Path:
        isat = self._config.get("isat_yaml", "./origin_pics/isat.yaml")
        if not Path(isat).is_absolute():
            isat = self._project_root / isat
        return Path(isat)

    @property
    def train_val_ratio(self) -> int:
        return self._config.get("train_val_ratio", 10)

    @property
    def random_seed(self) -> int:
        return self._config.get("random_seed", 42)

    @property
    def config(self) -> dict:
        """返回完整配置字典"""
        return self._config.copy()

    @property
    def project_file(self) -> Optional[Path]:
        return self._project_file

    def get_dataset_yaml_content(self) -> str:
        """
        生成 YOLOv5 使用的 dataset.yaml 内容
        """
        data_root_str = str(self.data_root.resolve()).replace("\\", "/")

        # 生成 names 字典
        names_lines = []
        for c in self.classes:
            names_lines.append(f"  {c['id']}: {c['name']}")

        names_dict = "\n".join(names_lines) if names_lines else "  0: unknown"

        return f"""# ============================================================
# YOLOv5-seg 数据集配置文件
# 由 ConfigManager 自动生成，请勿手动修改
# ============================================================

path: {data_root_str}

train: images/train
val: images/val

nc: {len(self.classes)}

names:
{names_dict}
"""

    def save_dataset_yaml(self) -> Path:
        """
        将 dataset.yaml 保存到项目根目录
        """
        dataset_yaml_path = self._project_root / "dataset.yaml"
        content = self.get_dataset_yaml_content()

        with open(dataset_yaml_path, "w", encoding="utf-8") as f:
            f.write(content)

        return dataset_yaml_path

    def resolve_path(self, path: str) -> Path:
        """将相对路径解析为绝对路径（相对于项目根目录）"""
        p = Path(path)
        if p.is_absolute():
            return p
        return self._project_root / p


def get_config() -> ConfigManager:
    """
    获取全局配置管理器单例

    首次调用时尝试加载默认项目配置:
    - configs/projects/<当前目录名>.yaml
    - configs/projects/default.yaml
    """
    manager = ConfigManager.get_instance()

    if not manager._project_file:
        # 尝试自动查找配置
        project_root = manager._project_root

        # 1. 尝试使用目录名作为项目名
        project_name = project_root.name
        default_project = project_root / "configs" / "projects" / f"{project_name}.yaml"

        if default_project.exists():
            manager.load(default_project)
        else:
            # 2. 尝试 default.yaml
            default_yaml = project_root / "configs" / "projects" / "default.yaml"
            if default_yaml.exists():
                manager.load(default_yaml)
            else:
                # 创建默认配置
                manager._config = {
                    "project_name": "default",
                    "classes": [],
                    "data_root": "./train_data",
                    "origin_dir": "./origin_pics",
                    "isat_yaml": "./origin_pics/isat.yaml",
                    "train_val_ratio": 10,
                    "random_seed": 42,
                }

    return manager
