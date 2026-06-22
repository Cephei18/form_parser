from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import logging

logging.disable(logging.CRITICAL)

from src.document_routing import (
    LARGE,
    MEDIUM,
    ROUTE_ASYNC,
    ROUTE_JSON_REPLAY,
    ROUTE_LEGACY_SYNC,
    ROUTE_PARALLEL_SYNC,
    ROUTE_SYNC,
    SMALL,
    DocumentRoutingConfig,
    choose_routing,
)
from src.pipeline_config import PipelineConfig
from src.pipelines import textract_pipeline
from src.pipelines.textract_pipeline import _analyze_pages_with_routing, run_textract_pipeline


def _page_files(tmp_path: Path, count: int, *, suffix: str = ".png") -> list[tuple[int, Path]]:
    pages = []
    for page in range(1, count + 1):
        path = tmp_path / f"page_{page}{suffix}"
        path.write_bytes(f"page-{page}".encode("ascii"))
        pages.append((page, path))
    return pages


def _page_images(tmp_path: Path, count: int) -> list[tuple[int, Path]]:
    pages = []
    for page in range(1, count + 1):
        path = tmp_path / f"page_{page}.png"
        cv2.imwrite(str(path), np.full((320, 240, 3), 255, np.uint8))
        pages.append((page, path))
    return pages


def _bbox(left, top, w=0.2, h=0.02):
    return {"BoundingBox": {"Left": left, "Top": top, "Width": w, "Height": h}}


def _multipage_response(n_pages: int, fields_per_page: int = 1) -> dict:
    blocks: list[dict] = []
    for page in range(1, n_pages + 1):
        ids: list[str] = []
        for index in range(fields_per_page):
            top = 0.20 + index * 0.12
            kid, kw, vid, vw = f"k{page}_{index}", f"kw{page}_{index}", f"v{page}_{index}", f"vw{page}_{index}"
            ids.extend([kid, kw, vid, vw])
            blocks.extend(
                [
                    {
                        "Id": kid,
                        "BlockType": "KEY_VALUE_SET",
                        "EntityTypes": ["KEY"],
                        "Confidence": 95.0,
                        "Page": page,
                        "Geometry": _bbox(0.10, top, 0.12),
                        "Relationships": [{"Type": "CHILD", "Ids": [kw]}, {"Type": "VALUE", "Ids": [vid]}],
                    },
                    {
                        "Id": kw,
                        "BlockType": "WORD",
                        "Text": f"Field{index}",
                        "Confidence": 99.0,
                        "Page": page,
                        "Geometry": _bbox(0.10, top, 0.10),
                    },
                    {
                        "Id": vid,
                        "BlockType": "KEY_VALUE_SET",
                        "EntityTypes": ["VALUE"],
                        "Confidence": 95.0,
                        "Page": page,
                        "Geometry": _bbox(0.32, top, 0.20),
                        "Relationships": [{"Type": "CHILD", "Ids": [vw]}],
                    },
                    {
                        "Id": vw,
                        "BlockType": "WORD",
                        "Text": f"value{page}_{index}",
                        "Confidence": 99.0,
                        "Page": page,
                        "Geometry": _bbox(0.32, top, 0.16),
                    },
                ]
            )
        blocks.insert(
            0 if page == 1 else len(blocks) - len(ids),
            {
                "Id": f"page{page}",
                "BlockType": "PAGE",
                "Page": page,
                "Geometry": _bbox(0, 0, 1, 1),
                "Relationships": [{"Type": "CHILD", "Ids": ids}],
            },
        )
    return {"Blocks": blocks, "DocumentMetadata": {"Pages": n_pages}}


def _config(**overrides) -> DocumentRoutingConfig:
    payload = {
        "enabled": True,
        "async_textract_enabled": True,
        "parallel_sync_enabled": True,
        "small_max_pages": 5,
        "medium_max_pages": 20,
        "max_pages": 150,
        "max_textract_page_units": 150,
        "max_raster_bytes": 900 * 1024 * 1024,
        "max_source_bytes": 500 * 1024 * 1024,
        "parallel_max_workers": 4,
        "estimated_sync_seconds_per_page": 3.0,
        "timeout_budget_seconds": 90.0,
        "textract_cost_per_page_usd": 0.0,
    }
    payload.update(overrides)
    return DocumentRoutingConfig(**payload)


@pytest.mark.parametrize(
    ("page_count", "classification", "route"),
    [
        (1, SMALL, ROUTE_SYNC),
        (5, SMALL, ROUTE_SYNC),
        (20, MEDIUM, ROUTE_PARALLEL_SYNC),
        (50, LARGE, ROUTE_ASYNC),
        (100, LARGE, ROUTE_ASYNC),
    ],
)
def test_document_routing_policy_by_size(tmp_path, page_count, classification, route):
    decision = choose_routing(
        _page_files(tmp_path, page_count),
        source_bytes=page_count * 10,
        has_document_location=True,
        config=_config(),
    )

    assert decision.classification == classification
    assert decision.route == route
    assert decision.document_size.page_count == page_count
    assert decision.document_size.estimated_textract_page_units == page_count


@pytest.mark.parametrize(
    ("page_count", "classification", "route"),
    [
        (5, SMALL, ROUTE_SYNC),
        (6, MEDIUM, ROUTE_PARALLEL_SYNC),
        (20, MEDIUM, ROUTE_PARALLEL_SYNC),
        (21, LARGE, ROUTE_ASYNC),
    ],
)
def test_threshold_boundaries(tmp_path, page_count, classification, route):
    decision = choose_routing(
        _page_files(tmp_path, page_count),
        has_document_location=True,
        config=_config(),
    )

    assert decision.classification == classification
    assert decision.route == route


def test_rollback_disabled_uses_legacy_sync(tmp_path):
    decision = choose_routing(
        _page_files(tmp_path, 100),
        has_document_location=True,
        config=_config(enabled=False),
    )

    assert decision.route == ROUTE_LEGACY_SYNC
    assert decision.sync_pages == 100
    assert decision.async_pages == 0


def test_large_document_degrades_when_async_unavailable(tmp_path):
    decision = choose_routing(
        _page_files(tmp_path, 50),
        has_document_location=False,
        config=_config(async_textract_enabled=True, parallel_sync_enabled=True),
    )

    assert decision.route == ROUTE_PARALLEL_SYNC
    assert decision.degrade_reason == "async_unavailable"
    assert "async_textract_unavailable" in decision.guardrail_warnings


def test_guardrails_surface_limits(tmp_path):
    decision = choose_routing(
        _page_files(tmp_path, 151),
        source_bytes=10,
        has_document_location=True,
        config=_config(max_pages=150, max_textract_page_units=150),
    )

    assert decision.classification == LARGE
    assert "page_count_exceeds_configured_limit" in decision.guardrail_warnings
    assert "estimated_textract_page_units_exceeds_limit" in decision.guardrail_warnings


def test_config_flags_default_off(monkeypatch):
    monkeypatch.delenv("FORM_PARSER_LARGE_DOC_ROUTING_ENABLED", raising=False)
    monkeypatch.delenv("FORM_PARSER_ASYNC_TEXTRACT_ENABLED", raising=False)
    monkeypatch.delenv("FORM_PARSER_TEXTRACT_ASYNC", raising=False)
    monkeypatch.delenv("FORM_PARSER_PARALLEL_SYNC_ENABLED", raising=False)

    config = PipelineConfig.from_env()

    assert config.large_doc_routing_enabled is False
    assert config.async_textract_enabled is False
    assert config.parallel_sync_enabled is False


def test_config_flags_and_thresholds(monkeypatch):
    monkeypatch.setenv("FORM_PARSER_LARGE_DOC_ROUTING_ENABLED", "1")
    monkeypatch.setenv("FORM_PARSER_ASYNC_TEXTRACT_ENABLED", "1")
    monkeypatch.setenv("FORM_PARSER_PARALLEL_SYNC_ENABLED", "1")
    monkeypatch.setenv("FORM_PARSER_SMALL_DOC_MAX_PAGES", "4")
    monkeypatch.setenv("FORM_PARSER_MEDIUM_DOC_MAX_PAGES", "12")
    monkeypatch.setenv("FORM_PARSER_MAX_DOCUMENT_PAGES", "120")
    monkeypatch.setenv("FORM_PARSER_PARALLEL_SYNC_MAX_WORKERS", "3")

    config = PipelineConfig.from_env()

    assert config.large_doc_routing_enabled is True
    assert config.async_textract_enabled is True
    assert config.parallel_sync_enabled is True
    assert config.small_doc_max_pages == 4
    assert config.medium_doc_max_pages == 12
    assert config.max_document_pages == 120
    assert config.parallel_sync_max_workers == 3


def test_parallel_sync_preserves_page_order(monkeypatch, tmp_path):
    monkeypatch.setenv("FORM_PARSER_LARGE_DOC_ROUTING_ENABLED", "1")
    monkeypatch.setenv("FORM_PARSER_PARALLEL_SYNC_ENABLED", "1")
    monkeypatch.delenv("FORM_PARSER_ASYNC_TEXTRACT_ENABLED", raising=False)

    def fake_analyze(path: Path):
        page = int(path.stem.split("_")[-1])
        return {"Blocks": [{"Id": f"page-{page}", "BlockType": "WORD"}], "DocumentMetadata": {"Pages": 1}}

    monkeypatch.setattr(textract_pipeline, "_analyze_with_textract", fake_analyze)
    raw, diagnostics = _analyze_pages_with_routing(_page_files(tmp_path, 20), source_path=tmp_path / "doc.pdf")

    assert diagnostics["route"] == ROUTE_PARALLEL_SYNC
    assert diagnostics["parallel_workers"] == 4
    assert [block["Id"] for block in raw["Blocks"]] == [f"page-{page}" for page in range(1, 21)]
    assert [block["Page"] for block in raw["Blocks"]] == list(range(1, 21))


def test_async_route_reuses_existing_textract_analysis(monkeypatch, tmp_path):
    monkeypatch.setenv("FORM_PARSER_LARGE_DOC_ROUTING_ENABLED", "1")
    monkeypatch.setenv("FORM_PARSER_ASYNC_TEXTRACT_ENABLED", "1")
    monkeypatch.setenv("FORM_PARSER_PARALLEL_SYNC_ENABLED", "1")
    called = {}

    def fake_async(bucket, key, **kwargs):
        called["bucket"] = bucket
        called["key"] = key
        return {
            "DocumentMetadata": {"Pages": 50},
            "Blocks": [{"Id": "async-page", "BlockType": "WORD", "Page": 1}],
            "JobStatus": "SUCCEEDED",
        }

    import src.textract_analysis as textract_analysis

    monkeypatch.setattr(textract_analysis, "analyze_document_async", fake_async)
    raw, diagnostics = _analyze_pages_with_routing(
        _page_files(tmp_path, 50),
        document_location={"bucket": "raw", "key": "uploads/form.pdf"},
        source_path=tmp_path / "form.pdf",
    )

    assert diagnostics["route"] == ROUTE_ASYNC
    assert diagnostics["async_pages"] == 50
    assert called == {"bucket": "raw", "key": "uploads/form.pdf"}
    assert raw["JobStatus"] == "SUCCEEDED"


def test_pipeline_writes_routing_diagnostics_for_json_replay(monkeypatch, tmp_path):
    monkeypatch.setenv("FORM_PARSER_LARGE_DOC_ROUTING_ENABLED", "1")
    monkeypatch.setenv("FORM_PARSER_PARALLEL_SYNC_ENABLED", "1")
    monkeypatch.setenv("FORM_PARSER_CONFIDENCE_PIPELINE_ENABLED", "1")
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(_multipage_response(5)), encoding="utf-8")
    pages = _page_images(tmp_path, 5)

    result = run_textract_pipeline(
        str(raw),
        str(tmp_path / "run"),
        reference_image_path=str(pages[0][1]),
        page_images=[(page, str(path)) for page, path in pages],
    )

    routing = json.loads(Path(result["routing_diagnostics_path"]).read_text(encoding="utf-8"))
    diagnostics = json.loads(Path(result["diagnostics_path"]).read_text(encoding="utf-8"))
    assert routing["actual_route"] == ROUTE_JSON_REPLAY
    assert routing["classification"] == SMALL
    assert routing["document_size"]["page_count"] == 5
    assert diagnostics["document_routing"]["actual_route"] == ROUTE_JSON_REPLAY
    assert diagnostics["confidence"]["enabled"] is True
    assert result["page_observability"]["pages_rendered"] == [1, 2, 3, 4, 5]
