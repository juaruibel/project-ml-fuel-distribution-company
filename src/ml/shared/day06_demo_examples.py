from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.ml.shared.project_paths import ASSETS_DIR


@dataclass(frozen=True)
class Day06DemoWorkbook:
    """Descriptor for one workbook exposed as a first-class demo entrypoint."""

    key: str
    label: str
    button_label: str
    asset_filename: str
    source_name: str
    note: str

    @property
    def asset_path(self) -> Path:
        """Resolve the workbook inside the shared assets directory."""
        return ASSETS_DIR / self.asset_filename


DAY06_DEMO_WORKBOOKS: tuple[Day06DemoWorkbook, ...] = (
    Day06DemoWorkbook(
        key="example_1",
        label="Ejemplo 1",
        button_label="Cargar ejemplo 1",
        asset_filename="day06_demo_comparativa.xlsx",
        source_name="Comparativa de precios 21-10-2015.xlsx",
        note="Caso limpio para enseñar el flujo completo.",
    ),
    Day06DemoWorkbook(
        key="example_2",
        label="Ejemplo 2",
        button_label="Cargar ejemplo 2",
        asset_filename="day06_demo_comparativa_2.xlsx",
        source_name="Comparativa de precios 13-11-2015.xlsx",
        note="Caso algo más denso para enseñar revisión.",
    ),
)


def get_day06_demo_workbooks() -> tuple[Day06DemoWorkbook, ...]:
    """Return the ordered workbook descriptors used by UI and deploy exports."""
    return DAY06_DEMO_WORKBOOKS
