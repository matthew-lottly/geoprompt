from __future__ import annotations

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

from nbclient import NotebookClient


SECTION_D_NOTEBOOKS = [
    Path("examples/notebooks/section_d/d1_utilities_workflow.ipynb"),
    Path("examples/notebooks/section_d/d2_forestry_management_workflow.ipynb"),
    Path("examples/notebooks/section_d/d3_flood_analysis_workflow.ipynb"),
    Path("examples/notebooks/section_d/d4_transportation_workflow.ipynb"),
    Path("examples/notebooks/section_d/d5_climate_workflow.ipynb"),
]

GEOPROMPT_NOTEBOOKS = [
    Path("examples/notebooks/geoprompt/d1_utilities_workflow.ipynb"),
    Path("examples/notebooks/geoprompt/d2_forestry_management_workflow.ipynb"),
    Path("examples/notebooks/geoprompt/d3_flood_analysis_workflow.ipynb"),
    Path("examples/notebooks/geoprompt/d4_transportation_workflow.ipynb"),
    Path("examples/notebooks/geoprompt/d5_climate_workflow.ipynb"),
]

GEOPANDAS_NOTEBOOKS = [
    Path("examples/notebooks/geopandas/d1_utilities_workflow.ipynb"),
    Path("examples/notebooks/geopandas/d2_forestry_management_workflow.ipynb"),
    Path("examples/notebooks/geopandas/d3_flood_analysis_workflow.ipynb"),
    Path("examples/notebooks/geopandas/d4_transportation_workflow.ipynb"),
    Path("examples/notebooks/geopandas/d5_climate_workflow.ipynb"),
]


@pytest.mark.parametrize("path", SECTION_D_NOTEBOOKS)
def test_section_d_notebook_structure(path: Path) -> None:
    assert path.exists(), path
    notebook = nbformat.read(path.open("r", encoding="utf-8"), as_version=4)

    assert notebook["cells"], f"Notebook has no cells: {path}"
    markdown_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "markdown"]
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]

    assert len(markdown_cells) >= 3
    assert len(code_cells) >= 4

    text = "\n".join("".join(cell.get("source", "")) for cell in markdown_cells)
    assert "Section A" in text
    assert "Section B" in text
    assert "Section C" in text


@pytest.mark.parametrize("path", SECTION_D_NOTEBOOKS)
def test_section_d_notebook_executes(path: Path) -> None:
    notebook = nbformat.read(path.open("r", encoding="utf-8"), as_version=4)
    client = NotebookClient(notebook, timeout=40, kernel_name="python3")
    client.execute(cwd=str(Path.cwd()))


@pytest.mark.parametrize("path", GEOPROMPT_NOTEBOOKS)
def test_geoprompt_notebook_structure(path: Path) -> None:
    assert path.exists(), path
    notebook = nbformat.read(path.open("r", encoding="utf-8"), as_version=4)
    assert notebook["cells"], f"Notebook has no cells: {path}"
    markdown_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "markdown"]
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    assert len(markdown_cells) >= 3
    assert len(code_cells) >= 4
    text = "\n".join("".join(cell.get("source", "")) for cell in markdown_cells)
    assert "Section A" in text
    assert "Section B" in text
    assert "Section C" in text


@pytest.mark.parametrize("path", GEOPROMPT_NOTEBOOKS)
def test_geoprompt_notebook_executes(path: Path) -> None:
    notebook = nbformat.read(path.open("r", encoding="utf-8"), as_version=4)
    client = NotebookClient(notebook, timeout=60, kernel_name="python3")
    client.execute(cwd=str(Path.cwd()))


@pytest.mark.parametrize("path", GEOPANDAS_NOTEBOOKS)
def test_geopandas_notebook_structure(path: Path) -> None:
    assert path.exists(), path
    notebook = nbformat.read(path.open("r", encoding="utf-8"), as_version=4)
    assert notebook["cells"], f"Notebook has no cells: {path}"
    markdown_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "markdown"]
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    assert len(markdown_cells) >= 3
    assert len(code_cells) >= 4
    text = "\n".join("".join(cell.get("source", "")) for cell in markdown_cells)
    assert "Section A" in text
    assert "Section B" in text
    assert "Section C" in text


@pytest.mark.parametrize("path", GEOPANDAS_NOTEBOOKS)
def test_geopandas_notebook_executes(path: Path) -> None:
    notebook = nbformat.read(path.open("r", encoding="utf-8"), as_version=4)
    client = NotebookClient(notebook, timeout=60, kernel_name="python3")
    client.execute(cwd=str(Path.cwd()))
