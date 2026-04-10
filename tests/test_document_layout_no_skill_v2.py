import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest


def _module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def service_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "azure", _module("azure"))
    monkeypatch.setitem(sys.modules, "azure.ai", _module("azure.ai"))
    monkeypatch.setitem(
        sys.modules,
        "azure.ai.documentintelligence",
        _module("azure.ai.documentintelligence"),
    )
    monkeypatch.setitem(
        sys.modules,
        "azure.ai.documentintelligence.models",
        _module(
            "azure.ai.documentintelligence.models",
            DocumentContentFormat=type("DocumentContentFormat", (), {"TEXT": "text"}),
        ),
    )

    monkeypatch.setitem(
        sys.modules,
        "src.conf.conf",
        _module("src.conf.conf", get_config=lambda: {}),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.services.ai_search.service",
        _module(
            "src.services.ai_search.service",
            AISearchService=type("AISearchService", (), {}),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.services.document_intelligence.service",
        _module(
            "src.services.document_intelligence.service",
            DocumentIntelligenceService=type("DocumentIntelligenceService", (), {}),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.services.shared",
        _module(
            "src.services.shared",
            DEFAULT_CHUNK_CONTAINER="chunk-container",
            DEFAULT_TARGET_INDEX_NAME="rag-index",
            build_shared_index=lambda *args, **kwargs: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.services.storage_account",
        _module(
            "src.services.storage_account",
            AzureStorageAccountService=type("AzureStorageAccountService", (), {}),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.storage",
        _module("src.storage", LocalOutputStore=type("LocalOutputStore", (), {})),
    )

    module_path = (
        Path(__file__).resolve().parents[1]
        / "src/services/document_layout_no_skill_v2/service.py"
    )
    spec = importlib.util.spec_from_file_location(
        "document_layout_no_skill_v2_service_test", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def service(service_module):
    return service_module.DocumentLayoutNoSkillV2Service.__new__(
        service_module.DocumentLayoutNoSkillV2Service
    )


def _region(page_number: int, polygon: list[float]) -> dict[str, Any]:
    return {
        "page_number": page_number,
        "polygon": polygon,
    }


def _paragraph(
    *,
    index: int,
    text: str,
    left: float,
    top: float,
    right: float,
    bottom: float,
    page_number: int = 1,
) -> dict[str, Any]:
    return {
        "index": index,
        "text": text,
        "page_number": page_number,
        "bounding_regions": [
            _region(
                page_number,
                [left, top, right, top, right, bottom, left, bottom],
            )
        ],
        "bbox": {
            "page_number": float(page_number),
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "center_x": (left + right) / 2,
            "center_y": (top + bottom) / 2,
        },
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "center_x": (left + right) / 2,
        "center_y": (top + bottom) / 2,
    }


def test_bbox_from_bounding_regions_returns_normalized_metrics(service) -> None:
    bbox = service._bbox_from_bounding_regions(
        bounding_regions=[_region(1, [20, 10, 60, 10, 60, 30, 20, 30])],
        page_dimensions={1: {"width": 100.0, "height": 50.0}},
    )

    assert bbox is not None
    assert bbox["left"] == 0.2
    assert bbox["right"] == 0.6
    assert bbox["top"] == 0.2
    assert bbox["bottom"] == 0.6
    assert bbox["center_x"] == 0.4
    assert bbox["center_y"] == 0.4


def test_bbox_from_bounding_regions_returns_none_without_polygon(service) -> None:
    bbox = service._bbox_from_bounding_regions(
        bounding_regions=[{"page_number": 1, "polygon": []}],
        page_dimensions={1: {"width": 100.0, "height": 50.0}},
    )
    assert bbox is None


def test_extract_figure_reference_ids_normalizes_variants(service) -> None:
    refs = service._extract_figure_reference_ids(
        "See Figure 1.2, Fig. 1-3, and the comparison in (2.1)."
    )
    assert refs == {"1.2", "1.3", "2.1"}


def test_is_spatial_candidate_rejects_cross_column_paragraph(service) -> None:
    figure_bbox = {
        "left": 0.1,
        "right": 0.4,
        "top": 0.2,
        "bottom": 0.5,
        "center_x": 0.25,
        "center_y": 0.35,
    }
    paragraph_bbox = {
        "left": 0.6,
        "right": 0.9,
        "top": 0.25,
        "bottom": 0.35,
        "center_x": 0.75,
        "center_y": 0.3,
    }
    assert not service._is_spatial_candidate(
        figure_bbox=figure_bbox,
        paragraph_bbox=paragraph_bbox,
    )


def test_is_spatial_candidate_accepts_caption_band_paragraph(service) -> None:
    figure_bbox = {
        "left": 0.1,
        "right": 0.4,
        "top": 0.2,
        "bottom": 0.5,
        "center_x": 0.25,
        "center_y": 0.35,
    }
    paragraph_bbox = {
        "left": 0.15,
        "right": 0.38,
        "top": 0.52,
        "bottom": 0.58,
        "center_x": 0.265,
        "center_y": 0.55,
    }
    assert service._is_spatial_candidate(
        figure_bbox=figure_bbox,
        paragraph_bbox=paragraph_bbox,
    )


def test_select_relevant_text_prevents_cross_figure_contamination(service) -> None:
    figure_bbox = {
        "page_number": 1.0,
        "left": 0.1,
        "right": 0.45,
        "top": 0.18,
        "bottom": 0.5,
        "center_x": 0.275,
        "center_y": 0.34,
    }
    page_paragraphs = {
        1: [
            _paragraph(
                index=0,
                text="Figure 1.2 shows the separate workflow for the later stage.",
                left=0.12,
                top=0.12,
                right=0.42,
                bottom=0.17,
            ),
            _paragraph(
                index=1,
                text="Figure 1.1 compares the multi-panel baseline with the tuned variants.",
                left=0.12,
                top=0.51,
                right=0.42,
                bottom=0.57,
            ),
        ]
    }

    relevant, surrounding = service._select_relevant_text(
        figure_id="1.1",
        caption="Figure 1.1 multi-panel comparison",
        figure_bbox=figure_bbox,
        page_number=1,
        page_paragraphs=page_paragraphs,
    )

    assert "Figure 1.1" in relevant
    assert "Figure 1.2" not in relevant
    assert surrounding == ""


def test_select_relevant_text_limits_context_deterministically(
    service, service_module
) -> None:
    figure_bbox = {
        "page_number": 1.0,
        "left": 0.1,
        "right": 0.45,
        "top": 0.18,
        "bottom": 0.5,
        "center_x": 0.275,
        "center_y": 0.34,
    }
    page_paragraphs = {
        1: [
            _paragraph(
                index=0,
                text="Figure 2.1 overview.",
                left=0.12,
                top=0.51,
                right=0.42,
                bottom=0.56,
            ),
            _paragraph(
                index=1,
                text="Figure 2.1 trend details.",
                left=0.11,
                top=0.57,
                right=0.43,
                bottom=0.62,
            ),
            _paragraph(
                index=2,
                text="Figure 2.1 caption-adjacent interpretation.",
                left=0.13,
                top=0.63,
                right=0.44,
                bottom=0.68,
            ),
            _paragraph(
                index=3,
                text="Figure 2.1 remaining nearby note.",
                left=0.12,
                top=0.69,
                right=0.41,
                bottom=0.74,
            ),
            _paragraph(
                index=4,
                text="Figure 2.1 distant nearby note.",
                left=0.12,
                top=0.75,
                right=0.41,
                bottom=0.8,
            ),
        ]
    }

    relevant, surrounding = service._select_relevant_text(
        figure_id="2.1",
        caption="Figure 2.1 overview",
        figure_bbox=figure_bbox,
        page_number=1,
        page_paragraphs=page_paragraphs,
    )

    assert relevant.count("Figure 2.1") <= service_module.MAX_RELEVANT_PARAGRAPHS
    assert surrounding.count("Figure 2.1") <= service_module.MAX_SURROUNDING_PARAGRAPHS


def test_select_relevant_text_returns_empty_when_no_spatial_candidates(service) -> None:
    figure_bbox = {
        "page_number": 1.0,
        "left": 0.1,
        "right": 0.3,
        "top": 0.1,
        "bottom": 0.2,
        "center_x": 0.2,
        "center_y": 0.15,
    }
    page_paragraphs = {
        1: [
            _paragraph(
                index=0,
                text="Figure 3.1 appears elsewhere on the page.",
                left=0.7,
                top=0.7,
                right=0.95,
                bottom=0.8,
            )
        ]
    }

    relevant, surrounding = service._select_relevant_text(
        figure_id="3.1",
        caption="Figure 3.1",
        figure_bbox=figure_bbox,
        page_number=1,
        page_paragraphs=page_paragraphs,
    )

    assert relevant == ""
    assert surrounding == ""
