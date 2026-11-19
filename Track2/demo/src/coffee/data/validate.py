"""Check a dataset tree and report every problem, not just the first."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from coffee.config import AppConfig


class DatasetError(ValueError):
    """The dataset tree cannot be used."""


@dataclass
class DatasetReport:
    root: Path
    train_images: dict[str, list[Path]]
    test_images: dict[str, list[Path]]
    test_flat: list[Path]
    day3_train: list[Path]
    day3_test: list[Path]
    problems: list[str] = field(default_factory=list)

    def raise_if_problems(self) -> None:
        if self.problems:
            details = "\n".join(f"- {problem}" for problem in self.problems)
            raise DatasetError(f"Dataset at {self.root} is not usable:\n{details}")


def validate_dataset(root: Path, cfg: AppConfig) -> DatasetReport:
    root = root.resolve()
    report = DatasetReport(
        root=root,
        train_images={},
        test_images={},
        test_flat=[],
        day3_train=[],
        day3_test=[],
    )
    train_root = root / cfg.train_dirname
    if not train_root.is_dir():
        report.problems.append(f"Missing train folder: {train_root}")
    else:
        _validate_class_folders(train_root, cfg, report.train_images, report.problems, "train")
    test_root = root / cfg.test_dirname
    if not test_root.is_dir():
        report.problems.append(f"Missing test folder: {test_root}")
    else:
        _validate_test(test_root, cfg, report)
    day3_root = root / cfg.day3_dirname
    if day3_root.is_dir():
        _validate_day3(day3_root, cfg, report)
    return report


def _validate_class_folders(
    folder: Path,
    cfg: AppConfig,
    dest: dict[str, list[Path]],
    problems: list[str],
    label: str,
) -> None:
    found = {path.name for path in folder.iterdir() if path.is_dir()}
    unknown = sorted(found - set(cfg.classes))
    for name in unknown:
        problems.append(f"Unknown class folder in {label}: {folder / name}")
    for name in cfg.classes:
        class_dir = folder / name
        if not class_dir.is_dir():
            problems.append(f"Missing {label} class folder: {class_dir}")
            continue
        images = _images(class_dir, cfg)
        if not images:
            problems.append(f"Empty {label} class folder: {class_dir}")
        dest[name] = images


def _validate_test(folder: Path, cfg: AppConfig, report: DatasetReport) -> None:
    subdirs = [path for path in folder.iterdir() if path.is_dir()]
    images = _images(folder, cfg)
    if subdirs and images:
        report.problems.append(
            f"Test folder mixes class subfolders and loose images: {folder}. Use one layout."
        )
        return
    if subdirs:
        _validate_class_folders(folder, cfg, report.test_images, report.problems, "test")
        return
    if not images:
        report.problems.append(f"Empty test folder: {folder}")
        return
    report.test_flat = images


def _validate_day3(folder: Path, cfg: AppConfig, report: DatasetReport) -> None:
    for split, attr in (("train", "day3_train"), ("test", "day3_test")):
        split_dir = folder / split
        if not split_dir.is_dir():
            report.problems.append(f"Missing day 3 {split} folder: {split_dir}")
            continue
        images = _images(split_dir, cfg)
        if not images:
            report.problems.append(f"Empty day 3 {split} folder: {split_dir}")
        setattr(report, attr, images)
        if split != "train":
            continue
        for image_path in images:
            layout_path = image_path.with_name(image_path.stem + cfg.layout_suffix)
            if not layout_path.is_file():
                report.problems.append(f"Missing day 3 layout file: {layout_path}")
                continue
            names = [line.strip() for line in layout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not names:
                report.problems.append(f"Empty day 3 layout file: {layout_path}")
                continue
            unknown = [name for name in names if name not in cfg.classes]
            if unknown:
                report.problems.append(
                    f"Unknown class names in {layout_path}: {', '.join(unknown)}"
                )


def _images(folder: Path, cfg: AppConfig) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in cfg.image_extensions
    )
