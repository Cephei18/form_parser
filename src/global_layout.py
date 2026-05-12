from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, median
from typing import Any


EXCLUDED_REGION_TYPES = {
    "photo_region",
    "signature_area",
    "stamp_area",
    "logo_region",
    "decorative_box",
    "non_text_sparse_region",
    "non_text_candidate",
    "table_like_region",
    "checkbox_region",
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _median(values, default: float) -> float:
    clean = [float(value) for value in values if isinstance(value, (int, float)) and value > 0]
    if not clean:
        return default
    return float(median(clean))


def _ocr_bounds(item):
    bbox = item.get("bbox")
    if not bbox:
        return None
    try:
        xs = [float(point[0]) for point in bbox]
        ys = [float(point[1]) for point in bbox]
    except Exception:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _line_bounds(line):
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _line_center(line):
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _line_length(line):
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _box_center(bounds):
    x1, y1, x2, y2 = bounds
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _box_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    overlap_x = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    overlap_y = max(0.0, min(ay2, by2) - max(ay1, by1))
    if overlap_x <= 0 or overlap_y <= 0:
        return 0.0
    return (overlap_x * overlap_y) / max((ax2 - ax1) * (ay2 - ay1), 1.0)


def _page_bounds(ocr_data, field_lines, semantic_regions):
    xs = []
    ys = []
    for item in ocr_data or []:
        bounds = _ocr_bounds(item)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    for line in field_lines or []:
        x1, y1, x2, y2 = _line_bounds(line)
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    for region in semantic_regions or []:
        xs.extend([float(region.get("x", 0)), float(region.get("x", 0)) + float(region.get("width", 0))])
        ys.extend([float(region.get("y", 0)), float(region.get("y", 0)) + float(region.get("height", 0))])

    return {
        "width": max(xs) if xs else 0.0,
        "height": max(ys) if ys else 0.0,
    }


def _group_by_axis(items, getter, tolerance):
    groups = []
    for item in sorted(items, key=getter):
        value = getter(item)
        if not groups or abs(groups[-1]["center"] - value) > tolerance:
            groups.append({"center": value, "items": [item]})
            continue
        group = groups[-1]
        group["items"].append(item)
        group["center"] = mean(getter(entry) for entry in group["items"])
    return groups


def _build_entities(ocr_data, field_lines, semantic_regions):
    ocr_entities = []
    for index, item in enumerate(ocr_data or []):
        bounds = _ocr_bounds(item)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        ocr_entities.append(
            {
                "id": f"ocr-{index}",
                "kind": "ocr",
                "text": item.get("text", ""),
                "bounds": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "center": [round((x1 + x2) / 2.0, 2), round((y1 + y2) / 2.0, 2)],
                "confidence": float(item.get("confidence") or 0.0),
                "source_index": index,
            }
        )

    field_entities = []
    for index, line in enumerate(field_lines or []):
        x1, y1, x2, y2 = _line_bounds(line)
        field_entities.append(
            {
                "id": f"field-{index}",
                "kind": "field_line",
                "bounds": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "center": [round((x1 + x2) / 2.0, 2), round((y1 + y2) / 2.0, 2)],
                "length": round(_line_length(line), 2),
                "field_type": line.get("field_type", "line"),
                "source_index": index,
            }
        )

    region_entities = []
    for index, region in enumerate(semantic_regions or []):
        x = float(region.get("x", 0))
        y = float(region.get("y", 0))
        width = float(region.get("width", 0))
        height = float(region.get("height", 0))
        region_entities.append(
            {
                "id": f"region-{index}",
                "kind": "semantic_region",
                "region_type": region.get("type", "region"),
                "bounds": [round(x, 2), round(y, 2), round(x + width, 2), round(y + height, 2)],
                "center": [round(x + width / 2.0, 2), round(y + height / 2.0, 2)],
                "confidence": float(region.get("confidence") or 0.0),
                "reasons": region.get("reasons", []),
                "source_index": index,
            }
        )

    return {
        "ocr": ocr_entities,
        "fields": field_entities,
        "regions": region_entities,
    }


def _region_bounds(region):
    return [
        float(region.get("x", 0)),
        float(region.get("y", 0)),
        float(region.get("x", 0)) + float(region.get("width", 0)),
        float(region.get("y", 0)) + float(region.get("height", 0)),
    ]


def _point_in_bounds(point, bounds, padding: float = 0.0):
    x, y = point
    x1, y1, x2, y2 = bounds
    return x1 - padding <= x <= x2 + padding and y1 - padding <= y <= y2 + padding


def _bounds_center_from_items(items):
    xs1 = [item["bounds"][0] for item in items]
    ys1 = [item["bounds"][1] for item in items]
    xs2 = [item["bounds"][2] for item in items]
    ys2 = [item["bounds"][3] for item in items]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def _cluster_entities(entities, axis: str, tolerance: float):
    getter = (lambda entity: entity["center"][0]) if axis == "x" else (lambda entity: entity["center"][1])
    groups = _group_by_axis(entities, getter, tolerance)
    clusters = []
    for index, group in enumerate(groups):
        items = group["items"]
        bounds = _bounds_center_from_items(items)
        centers = [getter(item) for item in items]
        clusters.append(
            {
                "id": f"{axis}-cluster-{index}",
                "axis": axis,
                "center": round(group["center"], 2),
                "item_ids": [item["id"] for item in items],
                "items": items,
                "item_count": len(items),
                "bounds": [round(value, 2) for value in bounds],
                "span": round(max(centers) - min(centers) if len(centers) > 1 else 0.0, 2),
            }
        )
    return clusters


def _band_type(label_count: int, field_count: int, height: float, metrics: dict[str, Any]) -> str:
    if height >= metrics["multiline_band_gap"] and label_count > 1:
        return "multiline_band"
    if field_count >= 4:
        return "repeated_field_band"
    if label_count >= 2 and field_count >= 1:
        return "section_band"
    if field_count >= 2:
        return "field_band"
    return "label_band"


def _zone_support(items, total_count: int) -> float:
    if not total_count:
        return 0.0
    return _clamp01(len(items) / float(total_count))


def _band_from_group(group, metrics):
    items = group["items"]
    bounds = _bounds_center_from_items(items)
    label_items = [item for item in items if item["kind"] == "ocr"]
    field_items = [item for item in items if item["kind"] == "field_line"]
    return {
        "id": f"band-{len(items)}-{round(group['center'], 2)}",
        "type": _band_type(len(label_items), len(field_items), bounds[3] - bounds[1], metrics),
        "center_y": round(group["center"], 2),
        "bounds": [round(value, 2) for value in bounds],
        "item_ids": [item["id"] for item in items],
        "label_item_ids": [item["id"] for item in label_items],
        "field_item_ids": [item["id"] for item in field_items],
        "label_count": len(label_items),
        "field_count": len(field_items),
        "confidence": round(_clamp01(0.35 + 0.45 * _zone_support(items, len(items))), 4),
        "repetition_score": round(_clamp01(len(field_items) / max(1, len(items))), 4),
    }


def _cluster_membership(items, cluster_item_ids):
    return [item for item in items if item["id"] in cluster_item_ids]


def _build_column_records(clusters, column_type: str, metrics: dict[str, Any]):
    records = []
    page_width = max(metrics.get("page_width", 1.0), 1.0)
    for index, cluster in enumerate(clusters):
        items = cluster["items"]
        bounds = cluster["bounds"]
        x1, y1, x2, y2 = bounds
        average_width = mean(item["bounds"][2] - item["bounds"][0] for item in items)
        record = {
            "id": f"{column_type}-column-{index}",
            "type": f"{column_type}_column",
            "column_type": column_type,
            "index": index,
            "center_x": round(cluster["center"], 2),
            "bounds": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "item_ids": list(cluster["item_ids"]),
            "item_count": len(items),
            "support": round(_clamp01(len(items) / max(1.0, len(clusters))), 4),
            "average_item_width": round(average_width, 2),
            "confidence": round(_clamp01(0.55 + 0.25 * len(items) / max(1, len(clusters))), 4),
            "is_primary": False,
        }
        if column_type == "field":
            record["right_lean"] = round(_clamp01(cluster["center"] / page_width), 4)
        else:
            record["left_lean"] = round(_clamp01(1.0 - cluster["center"] / page_width), 4)
        records.append(record)
    if records:
        primary_index = max(range(len(records)), key=lambda idx: records[idx]["item_count"])
        records[primary_index]["is_primary"] = True
    return records


def _build_zones(bands, columns, entities, zone_type: str, metrics: dict[str, Any]):
    item_lookup = {item["id"]: item for item in entities}
    zones = []
    for band in bands:
        for column in columns:
            overlap_ids = [item_id for item_id in band["item_ids"] if item_id in column["item_ids"]]
            if not overlap_ids:
                continue
            items = [item_lookup[item_id] for item_id in overlap_ids]
            bounds = _bounds_center_from_items(items)
            band_bounds = band["bounds"]
            bounds[0] = min(bounds[0], band_bounds[0])
            bounds[1] = min(bounds[1], band_bounds[1])
            bounds[2] = max(bounds[2], band_bounds[2])
            bounds[3] = max(bounds[3], band_bounds[3])
            support = _zone_support(items, len(band["item_ids"]))
            zone = {
                "id": f"{zone_type}-zone-{len(zones)}",
                "type": f"{zone_type}_zone",
                "zone_type": zone_type,
                "band_id": band["id"],
                "column_id": column["id"],
                "bounds": [round(value, 2) for value in bounds],
                "center": [round((bounds[0] + bounds[2]) / 2.0, 2), round((bounds[1] + bounds[3]) / 2.0, 2)],
                "item_ids": overlap_ids,
                "item_count": len(items),
                "support": round(support, 4),
                "confidence": round(_clamp01(0.45 + 0.35 * support + 0.10 * band["confidence"]), 4),
                "line_count": len([item for item in items if item["kind"] == "field_line"]),
                "text_count": len([item for item in items if item["kind"] == "ocr"]),
                "band_type": band["type"],
                "column_type": column["column_type"],
                "multiline": bool(zone_type == "field" and (band["type"] == "multiline_band" or len(items) > 2)),
            }
            zones.append(zone)
    return zones


def _best_match(bounds, candidates, prefer_right_of: float | None = None):
    if not bounds:
        return None
    best = None
    best_score = -1.0
    center = _box_center(bounds)
    for candidate in candidates or []:
        candidate_bounds = candidate.get("bounds")
        if not candidate_bounds:
            continue
        overlap = _box_overlap(bounds, candidate_bounds)
        candidate_center = _box_center(candidate_bounds)
        horizontal_bonus = 0.0
        if prefer_right_of is not None:
            if candidate_center[0] >= prefer_right_of:
                horizontal_bonus = 0.15
            else:
                horizontal_bonus = -0.1
        score = (
            overlap * 0.55
            + candidate.get("confidence", 0.0) * 0.2
            + candidate.get("support", 0.0) * 0.15
            + (1.0 - min(abs(candidate_center[1] - center[1]) / max(candidate_bounds[3] - candidate_bounds[1], 1.0), 1.0)) * 0.1
            + horizontal_bonus
        )
        if score > best_score:
            best_score = score
            best = candidate
    return best


def _region_payload(zone, kind: str, confidence: float, bounds, label_zone=None, field_zone=None, band=None):
    payload = {
        "id": f"{kind}-region-{zone['id']}",
        "type": kind,
        "bounds": [round(value, 2) for value in bounds],
        "center": [round((bounds[0] + bounds[2]) / 2.0, 2), round((bounds[1] + bounds[3]) / 2.0, 2)],
        "confidence": round(_clamp01(confidence), 4),
        "label_zone_id": label_zone["id"] if label_zone else None,
        "field_zone_id": field_zone["id"] if field_zone else None,
        "band_id": band["id"] if band else None,
        "reasons": [],
    }
    if label_zone:
        payload["label_column_id"] = label_zone.get("column_id")
    if field_zone:
        payload["field_column_id"] = field_zone.get("column_id")
    return payload


def _build_ownership_regions(label_zones, field_zones, bands, metrics):
    ownership_regions = []
    for label_zone in label_zones:
        band = next((candidate for candidate in bands if candidate["id"] == label_zone["band_id"]), None)
        if band is None:
            continue
        same_band_fields = [zone for zone in field_zones if zone["band_id"] == band["id"]]
        field_zone = _best_match(label_zone["bounds"], same_band_fields, prefer_right_of=label_zone["bounds"][2])
        if field_zone is None:
            nearby_fields = [
                zone
                for zone in field_zones
                if abs((next((candidate for candidate in bands if candidate["id"] == zone["band_id"]), band)["center_y"]) - band["center_y"]) <= metrics["band_tolerance"] * 1.5
            ]
            field_zone = _best_match(label_zone["bounds"], nearby_fields, prefer_right_of=label_zone["bounds"][2])
        if field_zone is None:
            continue
        bounds = [
            min(label_zone["bounds"][0], field_zone["bounds"][0]),
            min(label_zone["bounds"][1], field_zone["bounds"][1]),
            max(label_zone["bounds"][2], field_zone["bounds"][2]),
            max(label_zone["bounds"][3], field_zone["bounds"][3]),
        ]
        multiline = bool(label_zone.get("text_count", 0) > 1 or field_zone.get("multiline") or band.get("type") == "multiline_band")
        confidence = _clamp01(
            0.34
            + 0.28 * label_zone.get("confidence", 0.0)
            + 0.28 * field_zone.get("confidence", 0.0)
            + 0.10 * band.get("confidence", 0.0)
        )
        ownership_regions.append(
            {
                **_region_payload(
                    label_zone,
                    "multiline_ownership_region" if multiline else "ownership_region",
                    confidence,
                    bounds,
                    label_zone=label_zone,
                    field_zone=field_zone,
                    band=band,
                ),
                "multiline": multiline,
                "coverage": round(_clamp01(field_zone.get("support", 0.0) + label_zone.get("support", 0.0)), 4),
            }
        )
    return ownership_regions


def _build_excluded_regions(semantic_regions):
    excluded_regions = []
    for region in semantic_regions or []:
        region_type = region.get("type", "region")
        if region_type not in EXCLUDED_REGION_TYPES:
            continue
        excluded_regions.append(
            {
                "id": f"excluded-{region.get('type', 'region')}-{region.get('x', 0)}-{region.get('y', 0)}",
                "type": region_type,
                "bounds": _region_bounds(region),
                "confidence": float(region.get("confidence", 0.0)),
                "reasons": region.get("reasons", []),
            }
        )
    return excluded_regions


def _best_zone_for_item(bounds, zones):
    if not bounds:
        return None
    best = None
    best_score = -1.0
    for zone in zones or []:
        zone_bounds = zone.get("bounds")
        if not zone_bounds:
            continue
        overlap = _box_overlap(bounds, zone_bounds)
        if overlap <= 0 and not _point_in_bounds(_box_center(bounds), zone_bounds, padding=6.0):
            continue
        score = overlap * 0.7 + zone.get("confidence", 0.0) * 0.2 + zone.get("support", 0.0) * 0.1
        if score > best_score:
            best = zone
            best_score = score
    return best


def _best_band_for_item(bounds, bands):
    return _best_zone_for_item(bounds, bands)


def _excluded_penalty(bounds, excluded_regions):
    for region in excluded_regions or []:
        if _box_overlap(bounds, region["bounds"]) > 0:
            return 1.0
        if _point_in_bounds(_box_center(bounds), region["bounds"], padding=2.0):
            return 1.0
    return 0.0


def build_layout_metrics(ocr_data, field_lines, semantic_regions=None):
    text_heights = []
    text_widths = []
    for item in ocr_data or []:
        bounds = _ocr_bounds(item)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        text_heights.append(y2 - y1)
        text_widths.append(x2 - x1)

    line_lengths = [_line_length(line) for line in field_lines or []]
    avg_text_height = _median(text_heights, 18.0)
    avg_text_width = _median(text_widths, 80.0)
    avg_line_length = _median(line_lengths, 220.0)
    page_bounds = _page_bounds(ocr_data, field_lines, semantic_regions)
    page_width = max(page_bounds["width"], 1.0)
    page_height = max(page_bounds["height"], 1.0)
    page_area = page_width * page_height
    ocr_density = len(ocr_data or []) / max(page_area / 100000.0, 1.0)

    return {
        "page_width": round(page_width, 2),
        "page_height": round(page_height, 2),
        "page_area": round(page_area, 2),
        "page_aspect_ratio": round(page_width / page_height, 4),
        "page_mid_x": round(page_width / 2.0, 2),
        "page_mid_y": round(page_height / 2.0, 2),
        "avg_text_height": round(avg_text_height, 2),
        "avg_text_width": round(avg_text_width, 2),
        "avg_line_length": round(avg_line_length, 2),
        "row_tolerance": round(max(18.0, avg_text_height * 1.5, page_height * 0.008), 2),
        "band_tolerance": round(max(28.0, avg_text_height * 2.3, page_height * 0.015), 2),
        "column_tolerance": round(max(28.0, avg_text_width * 0.8, page_width * 0.04), 2),
        "multiline_band_gap": round(max(34.0, avg_text_height * 2.6, page_height * 0.03), 2),
        "indent_tolerance": round(max(42.0, avg_text_height * 3.0, page_width * 0.04), 2),
        "region_gap": round(max(24.0, avg_text_height * 1.9), 2),
        "distance_limit": round(max(420.0, avg_line_length * 3.2, page_width * 0.55), 2),
        "ocr_density": round(ocr_density, 4),
        "ocr_count": len(ocr_data or []),
        "field_line_count": len(field_lines or []),
        "semantic_region_count": len(semantic_regions or []),
        "structure_weight": 0.52,
    }


def infer_page_structure(ocr_data, field_lines, semantic_regions=None):
    semantic_regions = semantic_regions or []
    metrics = build_layout_metrics(ocr_data, field_lines, semantic_regions)
    entities = _build_entities(ocr_data, field_lines, semantic_regions)
    excluded_regions = _build_excluded_regions(semantic_regions)

    page_mid_x = metrics["page_mid_x"]
    label_entities = [entity for entity in entities["ocr"] if entity["center"][0] <= page_mid_x * 1.12]
    field_entities = list(entities["fields"])

    label_clusters_x = _cluster_entities(label_entities, "x", metrics["column_tolerance"])
    field_clusters_x = _cluster_entities(field_entities, "x", metrics["column_tolerance"])
    band_groups = _group_by_axis(
        entities["ocr"] + entities["fields"],
        lambda entity: entity["center"][1],
        metrics["band_tolerance"],
    )
    bands = []
    for index, group in enumerate(band_groups):
        band = _band_from_group(group, metrics)
        band["id"] = f"band-{index}"
        band["index"] = index
        bands.append(band)

    label_columns = _build_column_records(label_clusters_x, "label", metrics)
    field_columns = _build_column_records(field_clusters_x, "field", metrics)

    label_zones = _build_zones(bands, label_columns, entities["ocr"], "label", metrics)
    field_zones = _build_zones(bands, field_columns, entities["fields"], "field", metrics)
    ownership_regions = _build_ownership_regions(label_zones, field_zones, bands, metrics)
    band_lookup = {band["id"]: band for band in bands}

    for zone in field_zones:
        bounds = zone.get("bounds")
        if not bounds or len(bounds) != 4:
            continue
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        area = width * height
        band = band_lookup.get(zone.get("band_id"), {})
        label_supported_field_band = int(band.get("label_count", 0)) > 0 and int(band.get("field_count", 0)) > 0
        sparse_zone = zone.get("text_count", 0) == 0 and zone.get("line_count", 0) >= 1 and zone.get("support", 0.0) < 0.55
        if label_supported_field_band and sparse_zone:
            continue
        large_sparse_zone = sparse_zone and height >= metrics["avg_text_height"] * 2.0 and area >= max(metrics["avg_text_height"] * metrics["avg_text_width"] * 0.5, 1200.0)
        square_sparse_zone = sparse_zone and 0.7 <= width / max(height, 1.0) <= 1.4 and area >= 700.0
        if large_sparse_zone or square_sparse_zone:
            excluded_regions.append(
                {
                    "id": f"structural-excluded-{zone['id']}",
                    "type": "photo_region" if square_sparse_zone else "non_text_sparse_region",
                    "bounds": bounds,
                    "confidence": round(_clamp01(0.62 + zone.get("support", 0.0) * 0.2), 4),
                    "reasons": ["structural_sparse_zone", "no_ocr_ownership"],
                }
            )

    label_zone_lookup = {zone["id"]: zone for zone in label_zones}
    field_zone_lookup = {zone["id"]: zone for zone in field_zones}
    column_lookup = {column["id"]: column for column in (label_columns + field_columns)}
    region_lookup = {region["id"]: region for region in excluded_regions + ownership_regions}

    graph_nodes = entities["ocr"] + entities["fields"] + entities["regions"] + bands + label_columns + field_columns + label_zones + field_zones + excluded_regions + ownership_regions
    graph_edges = []
    for entity in entities["ocr"] + entities["fields"]:
        entity_bounds = entity["bounds"]
        band = _best_band_for_item(entity_bounds, bands)
        if band is not None:
            graph_edges.append(
                {
                    "source": entity["id"],
                    "target": band["id"],
                    "relationship": f"{entity['kind']}_band_membership",
                    "score": round(band.get("confidence", 0.0), 4),
                    "reasons": [band.get("type", "band")],
                }
            )
        for zone in (label_zones if entity["kind"] == "ocr" else field_zones):
            if _box_overlap(entity_bounds, zone["bounds"]) <= 0:
                continue
            graph_edges.append(
                {
                    "source": entity["id"],
                    "target": zone["id"],
                    "relationship": f"{entity['kind']}_zone_membership",
                    "score": round(zone.get("confidence", 0.0), 4),
                    "reasons": [zone.get("zone_type", "zone")],
                }
            )
        for region in excluded_regions:
            if _box_overlap(entity_bounds, region["bounds"]) <= 0:
                continue
            graph_edges.append(
                {
                    "source": entity["id"],
                    "target": region["id"],
                    "relationship": "excluded_region",
                    "score": 1.0,
                    "reasons": [region["type"]],
                }
            )

    for zone in label_zones + field_zones:
        column = column_lookup.get(zone.get("column_id"))
        band = band_lookup.get(zone.get("band_id"))
        if column is not None:
            graph_edges.append(
                {
                    "source": zone["id"],
                    "target": column["id"],
                    "relationship": "structural_alignment",
                    "score": round(column.get("confidence", 0.0), 4),
                    "reasons": [column.get("column_type", "column")],
                }
            )
        if band is not None:
            graph_edges.append(
                {
                    "source": zone["id"],
                    "target": band["id"],
                    "relationship": "band_membership",
                    "score": round(band.get("confidence", 0.0), 4),
                    "reasons": [band.get("type", "band")],
                }
            )

    for region in ownership_regions:
        label_zone = label_zone_lookup.get(region.get("label_zone_id"))
        field_zone = field_zone_lookup.get(region.get("field_zone_id"))
        band = band_lookup.get(region.get("band_id"))
        if label_zone is not None and field_zone is not None:
            graph_edges.append(
                {
                    "source": label_zone["id"],
                    "target": field_zone["id"],
                    "relationship": "ownership",
                    "score": round(region.get("confidence", 0.0), 4),
                    "reasons": [region["type"]],
                }
            )
        if band is not None:
            graph_edges.append(
                {
                    "source": region["id"],
                    "target": band["id"],
                    "relationship": "structural_band",
                    "score": round(region.get("confidence", 0.0), 4),
                    "reasons": [region["type"]],
                }
            )

    text_rows = _group_by_axis(entities["ocr"], lambda entity: entity["center"][1], metrics["row_tolerance"])
    field_rows = _group_by_axis(entities["fields"], lambda entity: entity["center"][1], metrics["row_tolerance"])
    field_clusters = []
    for group in field_rows:
        items = group["items"]
        xs = [item["center"][0] for item in items]
        ys = [item["center"][1] for item in items]
        field_clusters.append(
            {
                "center_y": round(group["center"], 2),
                "line_count": len(items),
                "min_x": round(min(xs), 2) if xs else None,
                "max_x": round(max(xs), 2) if xs else None,
                "min_y": round(min(ys), 2) if ys else None,
                "max_y": round(max(ys), 2) if ys else None,
                "avg_line_length": round(_median([item["length"] for item in items], metrics["avg_line_length"]), 2),
                "type": "repeated_row" if len(items) >= 3 else "field_row",
            }
        )

    structural_zones = label_zones + field_zones

    adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in graph_edges:
        adjacency[edge["source"]].append(edge)

    return {
        "metrics": metrics,
        "text_rows": [
            {
                "center_y": round(group["center"], 2),
                "item_count": len(group["items"]),
                "texts": [item.get("text", "") for item in group["items"]],
            }
            for group in text_rows
        ],
        "field_clusters": field_clusters,
        "semantic_regions": semantic_regions,
        "entities": entities,
        "excluded_regions": excluded_regions,
        "label_columns": label_columns,
        "field_columns": field_columns,
        "layout_bands": bands,
        "label_zones": label_zones,
        "field_zones": field_zones,
        "structural_zones": structural_zones,
        "ownership_regions": ownership_regions,
        "ownership_groups": [
            {
                "id": region["id"],
                "type": region["type"],
                "item_ids": [region.get("label_zone_id"), region.get("field_zone_id")],
                "texts": [],
                "bounds": region["bounds"],
                "center": region["center"],
                "group_type": "multiline_label" if region.get("multiline") else "single_label",
                "confidence": region.get("confidence", 0.0),
            }
            for region in ownership_regions
        ],
        "graph": {
            "nodes": graph_nodes,
            "edges": graph_edges,
            "adjacency": {key: value for key, value in adjacency.items()},
        },
        "indexes": {
            "band_by_id": band_lookup,
            "label_zone_by_id": label_zone_lookup,
            "field_zone_by_id": field_zone_lookup,
            "column_by_id": column_lookup,
            "region_by_id": region_lookup,
        },
        "page_priors": {
            "primary_label_column_id": next((column["id"] for column in label_columns if column.get("is_primary")), None),
            "primary_field_column_id": next((column["id"] for column in field_columns if column.get("is_primary")), None),
            "band_count": len(bands),
            "label_column_count": len(label_columns),
            "field_column_count": len(field_columns),
            "zone_count": len(structural_zones),
            "excluded_region_count": len(excluded_regions),
        },
    }


def candidate_lines_for_label(label_item, field_lines, layout_structure):
    if not layout_structure:
        return [line for line in field_lines or []]

    label_bounds = _ocr_bounds(label_item)
    if label_bounds is None:
        return [line for line in field_lines or []]

    label_zone = _best_zone_for_item(label_bounds, layout_structure.get("label_zones", []))
    label_band = _best_band_for_item(label_bounds, layout_structure.get("layout_bands", []))
    excluded_regions = layout_structure.get("excluded_regions", []) or []
    candidate_zones = []

    if label_band is not None:
        candidate_zones = [zone for zone in layout_structure.get("field_zones", []) if zone.get("band_id") == label_band["id"]]
        if not candidate_zones:
            candidate_zones = [
                zone
                for zone in layout_structure.get("field_zones", [])
                if abs((layout_structure.get("indexes", {}).get("band_by_id", {}).get(zone.get("band_id"), {}).get("center_y", label_band["center_y"])) - label_band["center_y"]) <= layout_structure.get("metrics", {}).get("band_tolerance", 36.0) * 1.5
            ]
    if not candidate_zones:
        candidate_zones = list(layout_structure.get("field_zones", []))

    if label_zone is not None:
        same_column = [zone for zone in candidate_zones if zone.get("column_id") == label_zone.get("column_id")]
        right_side = [zone for zone in candidate_zones if zone["bounds"][0] >= label_zone["bounds"][2] - layout_structure.get("metrics", {}).get("column_tolerance", 40.0) * 0.2]
        candidate_zones = same_column or right_side or candidate_zones

    selected_lines = []
    for line in field_lines or []:
        bounds = _line_bounds(line)
        if _excluded_penalty(bounds, excluded_regions) > 0:
            continue
        line_center = _line_center(line)
        if candidate_zones:
            if any(_point_in_bounds(line_center, zone["bounds"], padding=4.0) for zone in candidate_zones):
                selected_lines.append(line)
                continue
            if label_zone is not None and line_center[0] >= label_zone["bounds"][2] - layout_structure.get("metrics", {}).get("column_tolerance", 40.0):
                selected_lines.append(line)
                continue
        else:
            selected_lines.append(line)

    if not selected_lines:
        selected_lines = [line for line in field_lines or [] if _excluded_penalty(_line_bounds(line), excluded_regions) == 0]

    selected_lines.sort(key=lambda line: (_line_center(line)[1], _line_center(line)[0]))
    return selected_lines


def structural_context_for(label_item, field_line, layout_structure):
    label_bounds = _ocr_bounds(label_item)
    line_bounds = _line_bounds(field_line)
    metrics = (layout_structure or {}).get("metrics", {})
    if label_bounds is None:
        return {
            "label_zone": None,
            "field_zone": None,
            "label_band": None,
            "field_band": None,
            "label_column": None,
            "field_column": None,
            "same_band": 0.0,
            "column_consistency": 0.0,
            "ownership_confidence": 0.0,
            "global_structure_score": 0.0,
            "excluded_penalty": 0.0,
            "structural_band_support": 0.0,
        }

    label_zone = _best_zone_for_item(label_bounds, layout_structure.get("label_zones", []))
    field_zone = _best_zone_for_item(line_bounds, layout_structure.get("field_zones", []))
    label_band = _best_band_for_item(label_bounds, layout_structure.get("layout_bands", []))
    field_band = _best_band_for_item(line_bounds, layout_structure.get("layout_bands", []))
    label_column = next((column for column in layout_structure.get("label_columns", []) if column.get("id") == (label_zone or {}).get("column_id")), None)
    field_column = next((column for column in layout_structure.get("field_columns", []) if column.get("id") == (field_zone or {}).get("column_id")), None)

    same_band = 1.0 if label_band is not None and field_band is not None and label_band["id"] == field_band["id"] else 0.0
    band_alignment = 0.0
    if label_band is not None and field_band is not None:
        band_gap = abs(float(field_band.get("center_y", 0.0)) - float(label_band.get("center_y", 0.0)))
        band_alignment = _clamp01(1.0 - (band_gap / max(float(metrics.get("band_tolerance", 36.0)) * 2.2, 1.0)))
    label_zone_support = float(label_zone.get("support", 0.0)) if label_zone else 0.0
    field_zone_support = float(field_zone.get("support", 0.0)) if field_zone else 0.0
    band_support = float((field_band or label_band or {}).get("confidence", 0.0))
    ownership_confidence = 0.0

    for region in layout_structure.get("ownership_regions", []) or []:
        if label_zone is not None and region.get("label_zone_id") != label_zone.get("id"):
            continue
        if field_zone is not None and region.get("field_zone_id") != field_zone.get("id"):
            continue
        ownership_confidence = max(ownership_confidence, float(region.get("confidence", 0.0)))

    if ownership_confidence == 0.0:
        ownership_confidence = _clamp01((label_zone_support * 0.35) + (field_zone_support * 0.35) + (band_support * 0.3))

    column_consistency = 0.0
    if label_column is not None and field_column is not None:
        right_of_label = field_column.get("center_x", 0.0) >= label_column.get("center_x", 0.0)
        column_gap = abs(field_column.get("center_x", 0.0) - label_column.get("center_x", 0.0))
        column_consistency = _clamp01(
            (0.6 if right_of_label else 0.15)
            + (0.25 * _clamp01(column_gap / max(metrics.get("page_width", 1.0) * 0.35, 1.0)))
            + (0.15 if field_column.get("is_primary") else 0.0)
        )
    elif field_column is not None:
        column_consistency = 0.55 if field_column.get("is_primary") else 0.35

    excluded_penalty = _excluded_penalty(line_bounds, layout_structure.get("excluded_regions", []))
    structural_band_support = max(same_band, band_support)
    global_structure_score = _clamp01(
        (same_band * 0.22)
        + (band_alignment * 0.18)
        + (column_consistency * 0.32)
        + (ownership_confidence * 0.26)
        + (field_zone_support * 0.10)
        + (label_zone_support * 0.06)
        + (structural_band_support * 0.04)
        - (excluded_penalty * 0.55)
    )

    return {
        "label_zone": label_zone,
        "field_zone": field_zone,
        "label_band": label_band,
        "field_band": field_band,
        "label_column": label_column,
        "field_column": field_column,
        "same_band": round(same_band, 4),
        "band_alignment": round(band_alignment, 4),
        "column_consistency": round(column_consistency, 4),
        "ownership_confidence": round(ownership_confidence, 4),
        "global_structure_score": round(global_structure_score, 4),
        "excluded_penalty": round(excluded_penalty, 4),
        "structural_band_support": round(structural_band_support, 4),
        "label_zone_support": round(label_zone_support, 4),
        "field_zone_support": round(field_zone_support, 4),
    }


def relationship_features_for(label_item, field_line, layout_structure):
    metrics = (layout_structure or {}).get("metrics", {})
    label_bounds = _ocr_bounds(label_item)
    if label_bounds is None:
        return {
            "same_row": 0.0,
            "same_column": 0.0,
            "vertical_continuation": 0.0,
            "x_overlap_ratio": 0.0,
            "spacing_similarity": 0.0,
            "size_consistency": 0.0,
            "ocr_confidence": float(label_item.get("confidence") or 0.0),
            "ownership_strength": 0.0,
            "region_support": 0.0,
            "alignment_score": 0.0,
            "column_consistency": 0.0,
            "ownership_confidence": 0.0,
            "global_structure_score": 0.0,
            "band_alignment": 0.0,
            "band_alignment": 0.0,
            "excluded_penalty": 0.0,
            "structural_band_support": 0.0,
            "label_zone_support": 0.0,
            "field_zone_support": 0.0,
        }

    line_bounds = _line_bounds(field_line)
    label_center = _box_center(label_bounds)
    line_center = _box_center(line_bounds)
    label_width = max(label_bounds[2] - label_bounds[0], 1.0)
    line_width = max(line_bounds[2] - line_bounds[0], 1.0)
    label_height = max(label_bounds[3] - label_bounds[1], 1.0)

    row_tolerance = max(float(metrics.get("row_tolerance", 24.0)), label_height * 1.25)
    multiline_gap = max(float(metrics.get("multiline_band_gap", 50.0)), label_height * 2.4)
    column_tolerance = max(float(metrics.get("column_tolerance", 40.0)), label_width * 0.8)

    vertical_gap = line_center[1] - label_center[1]
    horizontal_gap = line_center[0] - label_center[0]
    same_row = _clamp01(1.0 - (abs(vertical_gap) / max(row_tolerance, 1.0)))
    same_column = _clamp01(1.0 - (abs(horizontal_gap) / max(column_tolerance, 1.0)))
    continuation = _clamp01(1.0 - (max(0.0, vertical_gap) / max(multiline_gap, 1.0))) if vertical_gap >= 0 else 0.0

    span_overlap = max(0.0, min(label_bounds[2], line_bounds[2]) - max(label_bounds[0], line_bounds[0]))
    x_overlap_ratio = _clamp01(span_overlap / max(min(label_width, line_width), 1.0))
    spacing_similarity = _clamp01(1.0 - (abs(vertical_gap) / max(multiline_gap, 1.0)))
    size_consistency = _clamp01(1.0 - (abs(line_width - float(metrics.get("avg_line_length", line_width))) / max(float(metrics.get("avg_line_length", line_width)) * 1.4, 1.0)))
    alignment_score = _clamp01((same_row * 0.35) + (same_column * 0.2) + (continuation * 0.25) + (spacing_similarity * 0.2))

    structure_context = structural_context_for(label_item, field_line, layout_structure or {})
    ownership_strength = _clamp01((same_row * 0.24) + (continuation * 0.30) + (alignment_score * 0.18) + (structure_context["ownership_confidence"] * 0.28))
    region_support = max(float(structure_context.get("field_zone_support", 0.0)), float(structure_context.get("label_zone_support", 0.0)))

    return {
        "same_row": round(same_row, 4),
        "same_column": round(same_column, 4),
        "vertical_continuation": round(continuation, 4),
        "x_overlap_ratio": round(x_overlap_ratio, 4),
        "spacing_similarity": round(spacing_similarity, 4),
        "size_consistency": round(size_consistency, 4),
        "ocr_confidence": round(float(label_item.get("confidence") or 0.0), 4),
        "ownership_strength": round(_clamp01(ownership_strength), 4),
        "region_support": round(_clamp01(region_support), 4),
        "alignment_score": round(alignment_score, 4),
        "column_consistency": round(structure_context["column_consistency"], 4),
        "ownership_confidence": round(structure_context["ownership_confidence"], 4),
        "global_structure_score": round(structure_context["global_structure_score"], 4),
        "band_alignment": round(structure_context["band_alignment"], 4),
        "excluded_penalty": round(structure_context["excluded_penalty"], 4),
        "structural_band_support": round(structure_context["structural_band_support"], 4),
        "label_zone_support": round(structure_context["label_zone_support"], 4),
        "field_zone_support": round(structure_context["field_zone_support"], 4),
        "label_zone": structure_context["label_zone"],
        "field_zone": structure_context["field_zone"],
        "label_band": structure_context["label_band"],
        "field_band": structure_context["field_band"],
        "label_column": structure_context["label_column"],
        "field_column": structure_context["field_column"],
    }
