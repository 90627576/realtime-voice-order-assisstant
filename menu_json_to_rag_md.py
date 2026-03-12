"""
Convert a menu JSON export into a RAG-friendly Markdown file.

Designed for JSON shaped like:
[
  {
    "id": ...,
    "name": ...,
    "categories": [
      {"name": "...", "products": [ { "id":..., "name":..., "description":..., "attrs":[...] }, ... ] },
      ...
    ],
    "menu_times": [ {"week": 1..7, "start_at": "HH:MM", "end_at": "HH:MM"}, ... ]
  }
]

Why:
- RAGFlow chunking works best when each "document" is semantically coherent.
- The original JSON repeats a lot of attrs; this script extracts a common "options template"
  and only records per-product differences to avoid vector-db bloat.

Additional features:
- Automatically extracts ATTR_GROUP_* and ATTR_MAP_BY_ID from menu JSON
- Outputs `店家設定.yaml` for order_manager.py to consume
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


WEEK_LABEL = {
    1: "週一",
    2: "週二",
    3: "週三",
    4: "週四",
    5: "週五",
    6: "週六",
    7: "週日",
}


def _norm_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _week_range_label(weeks: Sequence[int]) -> str:
    ws = sorted(set(int(w) for w in weeks if w is not None))
    if not ws:
        return ""
    # Common cases
    if ws == [1, 2, 3, 4, 5, 6, 7]:
        return "週一～週日"
    if ws == [1, 2, 3, 4, 5]:
        return "週一～週五"
    if ws == [6, 7]:
        return "週六～週日"
    return "、".join(WEEK_LABEL.get(w, f"週{w}") for w in ws)


def _summarize_menu_times(menu_times: Any) -> str:
    """
    Returns a short human-readable summary.
    Example: "週一～週日 09:00–21:00"
    """
    if not isinstance(menu_times, list) or not menu_times:
        return ""
    week_to_time: Dict[int, Tuple[str, str]] = {}
    for t in menu_times:
        if not isinstance(t, dict):
            continue
        w = t.get("week")
        try:
            w = int(w)
        except Exception:
            continue
        start = str(t.get("start_at") or "").strip()
        end = str(t.get("end_at") or "").strip()
        if start and end:
            week_to_time[w] = (start, end)

    if not week_to_time:
        return ""

    # Group weeks by identical (start,end)
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for w, se in week_to_time.items():
        groups[se].append(w)

    parts: List[str] = []
    for (start, end), weeks in sorted(groups.items(), key=lambda kv: (min(kv[1]), kv[0][0], kv[0][1])):
        parts.append(f"{_week_range_label(weeks)} {start}–{end}")
    return "；".join(p for p in parts if p)


def _guess_attr_group_label(option_names: Sequence[str]) -> str:
    joined = " ".join(option_names)

    # Ice / temperature
    ice_temp_markers = [
        "正常冰",
        "少冰",
        "微冰",
        "去冰",
        "熱",
        "溫",
        "常溫",
        "去冰",
        "完全去冰",
    ]
    if any(m in joined for m in ice_temp_markers):
        return "冰量/溫度"

    # Sugar
    sugar_markers = [
        "全糖",
        "半糖",
        "微糖",
        "少糖",
        "無糖",
        "正常甜",
        "少甜",
        "微甜",
        "不另外加糖",
    ]
    if any(m in joined for m in sugar_markers):
        return "甜度"

    # Toppings
    topping_markers = [
        "珍珠",
        "椰果",
        "仙草",
        "布丁",
        "粉粿",
        "芋圓",
        "啵啵",
        "奶蓋",
        "奶霜",
        "咖啡凍",
        "茶凍",
        "燕麥",
        "紅豆",
    ]
    if any(m in joined for m in topping_markers):
        return "加料"

    return "選項"


@dataclass(frozen=True)
class OptionItem:
    name: str
    attr_item_id: Optional[int] = None
    attr_id: Optional[int] = None


def _format_option_with_price(
    opt: OptionItem,
    *,
    group_label: str,
    addon_price_default: Optional[int],
    addon_price_map: Dict[str, int],
) -> str:
    """
    Render option name with optional addon price label like "珍珠(+10)".
    Pricing is not present in this menu JSON, so we rely on user-provided defaults/maps.
    """
    name = opt.name
    if group_label != "加料":
        return name

    price: Optional[int] = None
    # Priority 1: attr_item_id
    if opt.attr_item_id is not None:
        price = addon_price_map.get(str(opt.attr_item_id))
    # Priority 2: exact name
    if price is None:
        price = addon_price_map.get(name)
    # Priority 3: group default
    if price is None:
        price = addon_price_default

    if price is None:
        return name
    sign = "+" if price >= 0 else ""
    return f"{name}({sign}{price})"


def _format_option_for_rag(
    opt: OptionItem,
    *,
    group_label: str,
    addon_price_default: Optional[int],
    addon_price_map: Dict[str, int],
) -> str:
    """Like _format_option_with_price but appends attr_id for vendor order format (RAG)."""
    base = _format_option_with_price(
        opt,
        group_label=group_label,
        addon_price_default=addon_price_default,
        addon_price_map=addon_price_map,
    )
    if opt.attr_id is not None:
        return f"{base}(attr_id:`{opt.attr_id}`)"
    return base


@dataclass
class ProductDoc:
    product_id: int
    name: str
    short_name: str = ""
    unit: str = ""
    status: Optional[int] = None
    description: str = ""
    image_url: str = ""
    categories: List[str] = field(default_factory=list)
    # list of (size/name, price, is_default, price_id)
    prices: List[Tuple[str, int, bool, Optional[int]]] = field(default_factory=list)
    # group_id -> ordered option names
    options_by_group: Dict[int, Tuple[OptionItem, ...]] = field(default_factory=dict)


def _extract_products(menu: Dict[str, Any]) -> Dict[int, ProductDoc]:
    products: Dict[int, ProductDoc] = {}

    categories = menu.get("categories")
    if not isinstance(categories, list):
        return products

    for cat in categories:
        if not isinstance(cat, dict):
            continue
        cat_name = str(cat.get("name") or "").strip()
        plist = cat.get("products")
        if not isinstance(plist, list):
            continue

        for p in plist:
            if not isinstance(p, dict):
                continue
            pid = p.get("id")
            try:
                pid = int(pid)
            except Exception:
                continue

            name = str(p.get("name") or "").strip()
            short_name = str(p.get("short_name") or "").strip()
            unit = str(p.get("unit") or "").strip()
            status = p.get("status")
            try:
                status = int(status) if status is not None else None
            except Exception:
                status = None
            description = _norm_text(p.get("description"))
            image_url = str(p.get("image_url") or "").strip()

            doc = products.get(pid)
            if not doc:
                doc = ProductDoc(
                    product_id=pid,
                    name=name,
                    short_name=short_name,
                    unit=unit,
                    status=status,
                    description=description,
                    image_url=image_url,
                    categories=[],
                    prices=[],
                    options_by_group={},
                )
                products[pid] = doc
            else:
                # Prefer keeping the richest description if duplicates differ
                if description and (not doc.description or len(description) > len(doc.description)):
                    doc.description = description
                if image_url and not doc.image_url:
                    doc.image_url = image_url
                if unit and not doc.unit:
                    doc.unit = unit
                if name and not doc.name:
                    doc.name = name
                if short_name and not doc.short_name:
                    doc.short_name = short_name

            if cat_name and cat_name not in doc.categories:
                doc.categories.append(cat_name)

            # prices -> list of sizes / amounts + price_id (for vendor order format)
            prices = p.get("prices")
            if isinstance(prices, list) and prices:
                parsed: List[Tuple[str, int, bool, Optional[int]]] = []
                for pr in prices:
                    if not isinstance(pr, dict):
                        continue
                    nm = str(pr.get("name") or "").strip()
                    if not nm:
                        nm = "價格"
                    price_val = pr.get("price")
                    try:
                        price_int = int(price_val)
                    except Exception:
                        continue
                    is_default = pr.get("is_default")
                    try:
                        is_default_bool = bool(int(is_default)) if is_default is not None else False
                    except Exception:
                        is_default_bool = False
                    # price_id: from price.id or pivot (menuable_type=product_price -> menuable_id)
                    price_id: Optional[int] = pr.get("id")
                    if price_id is None and isinstance(pr.get("pivot"), dict):
                        piv = pr["pivot"]
                        if str(piv.get("menuable_type") or "").strip() == "product_price":
                            price_id = piv.get("menuable_id")
                    try:
                        price_id = int(price_id) if price_id is not None else None
                    except Exception:
                        price_id = None
                    parsed.append((nm, price_int, is_default_bool, price_id))

                if parsed:
                    # De-dup & keep stable ordering: default first, then name, then price
                    parsed = list(dict.fromkeys(parsed))
                    parsed.sort(key=lambda x: (not x[2], x[0], x[1]))
                    # Keep the richer list if duplicates differ
                    if len(parsed) > len(doc.prices):
                        doc.prices = parsed
            else:
                # 單一價格（無 prices 陣列）：先喝道等店家僅有 price 欄位
                price_val = p.get("price")
                try:
                    price_int = int(price_val)
                except Exception:
                    price_int = 0
                if price_int >= 0 and not doc.prices:
                    doc.prices = [("單一規格", price_int, True, None)]

            # attrs -> grouped options
            attrs = p.get("attrs")
            if isinstance(attrs, list) and attrs:
                group_to_items: Dict[int, List[Tuple[int, OptionItem]]] = defaultdict(list)
                for a in attrs:
                    if not isinstance(a, dict):
                        continue
                    gid = a.get("attr_group_id")
                    try:
                        gid = int(gid)
                    except Exception:
                        continue
                    nm = str(a.get("name") or "").strip()
                    if not nm:
                        continue
                    attr_item_id = a.get("attr_item_id")
                    try:
                        attr_item_id_int = int(attr_item_id) if attr_item_id is not None else None
                    except Exception:
                        attr_item_id_int = None
                    attr_id = a.get("id")
                    try:
                        attr_id_int = int(attr_id) if attr_id is not None else None
                    except Exception:
                        attr_id_int = None
                    try:
                        srt = int(a.get("sort")) if a.get("sort") is not None else 10_000
                    except Exception:
                        srt = 10_000
                    group_to_items[gid].append(
                        (
                            srt,
                            OptionItem(
                                name=nm,
                                attr_item_id=attr_item_id_int,
                                attr_id=attr_id_int,
                            ),
                        )
                    )

                for gid, items in group_to_items.items():
                    ordered = tuple(opt for _, opt in sorted(items, key=lambda x: (x[0], x[1].name)))
                    # Keep the longer one if duplicates differ (rare)
                    if gid not in doc.options_by_group or len(ordered) > len(doc.options_by_group[gid]):
                        doc.options_by_group[gid] = ordered

    return products


def _build_group_templates(products: Iterable[ProductDoc]) -> Dict[int, Tuple[OptionItem, ...]]:
    """
    For each group_id, find the most common option list and treat it as "template".
    """
    group_counters: Dict[int, Counter[Tuple[OptionItem, ...]]] = defaultdict(Counter)
    for p in products:
        for gid, opt_tuple in p.options_by_group.items():
            if opt_tuple:
                group_counters[gid][opt_tuple] += 1

    templates: Dict[int, Tuple[OptionItem, ...]] = {}
    for gid, c in group_counters.items():
        if not c:
            continue
        templates[gid] = c.most_common(1)[0][0]
    return templates


def _render_markdown(
    *,
    menu: Dict[str, Any],
    products: Dict[int, ProductDoc],
    templates: Dict[int, Tuple[OptionItem, ...]],
    addon_price_default: Optional[int],
    addon_price_map: Dict[str, int],
) -> str:
    menu_id = menu.get("id")
    menu_name = str(menu.get("name") or "菜單").strip()
    menu_times = _summarize_menu_times(menu.get("menu_times"))

    lines: List[str] = []
    lines.append(f"# {menu_name}".strip())
    if menu_id is not None:
        lines.append(f"- menu_id: `{menu_id}`")
    if menu_times:
        lines.append(f"- 可點餐時段：{menu_times}")
    lines.append("")

    # Templates
    if templates:
        lines.append("## 常見可選項模板（用來去重）")
        # Sort groups by id for stability
        for gid in sorted(templates.keys()):
            opt = templates[gid]
            label = _guess_attr_group_label([o.name for o in opt])
            rendered = "、".join(
                _format_option_for_rag(
                    o,
                    group_label=label,
                    addon_price_default=addon_price_default,
                    addon_price_map=addon_price_map,
                )
                for o in opt
            )
            lines.append(f"- **{label}**（group_id: `{gid}`）：{rendered}")
        lines.append("")

    # Products by category for readability, but product sections are self-contained.
    # Build category -> product_ids
    cat_to_pids: Dict[str, List[int]] = defaultdict(list)
    for pid, p in products.items():
        for cn in p.categories or ["未分類"]:
            cat_to_pids[cn].append(pid)
    for cn in cat_to_pids:
        cat_to_pids[cn] = sorted(set(cat_to_pids[cn]), key=lambda x: (x))

    for cat_name in sorted(cat_to_pids.keys()):
        lines.append(f"## 分類：{cat_name}")
        lines.append("")
        for pid in cat_to_pids[cat_name]:
            p = products[pid]
            title = p.name or p.short_name or f"品項 {pid}"
            lines.append(f"### 品項：{title}")
            lines.append(f"- product_id: `{p.product_id}`")
            if p.short_name and p.short_name != p.name:
                lines.append(f"- short_name: {p.short_name}")
            if p.unit:
                lines.append(f"- 單位：{p.unit}")
            if p.prices:
                parts = []
                for item in p.prices:
                    nm, price_int, is_default, price_id = (
                        (item[0], item[1], item[2], item[3])
                        if len(item) >= 4
                        else (item[0], item[1], item[2], None)
                    )
                    part = f"{nm} {price_int}" + ("（預設）" if is_default else "")
                    if price_id is not None:
                        part += f"（price_id: `{price_id}`）"
                    parts.append(part)
                lines.append(f"- 價格：{'；'.join(parts)}")
            if p.categories:
                lines.append(f"- 所屬分類：{'、'.join(sorted(set(p.categories)))}")
            if p.image_url:
                lines.append(f"- 圖片：{p.image_url}")
            if p.description:
                lines.append("")
                lines.append("**描述**")
                lines.append(p.description)
                lines.append("")

            # Options: show diffs vs templates
            if not p.options_by_group:
                lines.append("- 可選項：無資料")
                lines.append("")
                continue

            diffs: List[str] = []
            for gid, opt in sorted(p.options_by_group.items(), key=lambda kv: kv[0]):
                if not opt:
                    continue
                tmpl = templates.get(gid)
                if tmpl and opt == tmpl:
                    continue
                label = _guess_attr_group_label([o.name for o in opt])
                rendered = "、".join(
                    _format_option_for_rag(
                        o,
                        group_label=label,
                        addon_price_default=addon_price_default,
                        addon_price_map=addon_price_map,
                    )
                    for o in opt
                )
                diffs.append(f"{label}（group_id:{gid}）：{rendered}")

            missing_groups = []
            for gid in sorted(templates.keys()):
                if gid not in p.options_by_group:
                    missing_groups.append(gid)

            if not diffs and not missing_groups:
                lines.append("- 可選項：同「常見可選項模板」")
                lines.append("")
                continue

            if diffs:
                lines.append("- **可選項（此品項差異）**：")
                for d in diffs:
                    lines.append(f"  - {d}")
            if missing_groups:
                # Only label which template groups are missing, without guessing too much.
                lines.append("- **不提供的模板群組**： " + "、".join(f"`{gid}`" for gid in missing_groups))

            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _sanitize_filename(name: str, *, fallback: str) -> str:
    """
    Windows-safe filename. Keep CJK, remove reserved characters.
    """
    s = (name or "").strip() or fallback
    s = re.sub(r'[<>:"/\\\\|?*]', " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Avoid trailing dots/spaces on Windows
    s = s.rstrip(" .")
    return s or fallback


def _extract_store_config(menu: Dict[str, Any]) -> Dict[str, Any]:
    """
    從菜單 JSON 中自動提取店家設定：
    - attr_groups: { ice: group_id, sugar: group_id, topping: group_id }
    - attr_map: { group_id: { attr_name: attr_id } }
    
    根據 attr name 自動判斷 group 類型：
    - 含「冰」「常溫」「熱」「溫」→ 冰度
    - 含「糖」→ 甜度
    - 其他 → 加料
    """
    # 收集所有 attrs
    all_attrs: List[Dict] = []
    categories = menu.get("categories")
    if isinstance(categories, list):
        for cat in categories:
            if not isinstance(cat, dict):
                continue
            products = cat.get("products")
            if not isinstance(products, list):
                continue
            for p in products:
                if not isinstance(p, dict):
                    continue
                attrs = p.get("attrs")
                if isinstance(attrs, list):
                    all_attrs.extend(attrs)
    
    # 分析每個 group_id 的屬性
    group_attrs: Dict[int, List[Dict]] = defaultdict(list)
    for attr in all_attrs:
        if not isinstance(attr, dict):
            continue
        gid = attr.get("attr_group_id")
        if gid is None:
            continue
        try:
            gid = int(gid)
        except (ValueError, TypeError):
            continue
        group_attrs[gid].append(attr)
    
    # 判斷每個 group 的類型
    attr_groups: Dict[str, int] = {}
    attr_map: Dict[str, Dict[str, str]] = {}
    
    ice_markers = {"冰", "常溫", "熱", "溫"}
    sugar_markers = {"糖"}
    
    for gid, attrs in group_attrs.items():
        # 收集該 group 的所有 name
        names = [str(a.get("name", "")).strip() for a in attrs if a.get("name")]
        joined = "".join(names)
        
        # 判斷類型
        group_type = None
        if any(m in joined for m in ice_markers):
            group_type = "ice"
        elif any(m in joined for m in sugar_markers):
            group_type = "sugar"
        else:
            # 預設為 topping（可能有多個 topping group，只取第一個）
            if "topping" not in attr_groups:
                group_type = "topping"
        
        if group_type and group_type not in attr_groups:
            attr_groups[group_type] = gid
        
        # 建立 attr_map: name → id
        gid_str = str(gid)
        if gid_str not in attr_map:
            attr_map[gid_str] = {}
        
        for a in attrs:
            name = str(a.get("name", "")).strip()
            aid = a.get("id")
            if name and aid is not None:
                try:
                    attr_map[gid_str][name] = str(int(aid))
                except (ValueError, TypeError):
                    pass
    
    return {
        "attr_groups": attr_groups,
        "attr_map": attr_map,
    }


def _write_store_config(config: Dict[str, Any], output_path: Path) -> None:
    """將店家設定寫入 YAML 檔案"""
    popular = config.get("popular_items", [])
    popular_yaml = "\n".join(f"  - {repr(x)}" for x in popular[:20]) if popular else "  # 換店時填入該店熱門品項，供促單使用\n  # - 珍珠奶茶\n  # - 大正紅茶拿鐵"
    content = f"""# 店家設定（由 menu_json_to_rag_md.py 自動生成）
# 請勿手動編輯，除非 JSON 格式不同需要手動調整

# 屬性群組 ID：用於 convert_to_vendor_format
attr_groups:
  ice: {config['attr_groups'].get('ice', 'null')}
  sugar: {config['attr_groups'].get('sugar', 'null')}
  topping: {config['attr_groups'].get('topping', 'null')}

# 屬性對照表：口頭值 → attr_id（用於 fallback，當 LLM 沒傳 attr_id 時）
attr_map:
"""
    for gid, mapping in sorted(config["attr_map"].items(), key=lambda x: int(x[0])):
        content += f"  '{gid}':\n"
        for name, aid in sorted(mapping.items()):
            content += f"    '{name}': '{aid}'\n"
    
    content += f"""
# 熱門品項（供促單用，結帳前/猶豫時可推薦。換店時手動填入）
popular_items:
{popular_yaml}
"""
    output_path.write_text(content, encoding="utf-8")
    print(f"OK: wrote {output_path} (attr_groups: {config['attr_groups']})")


def _render_product_doc(
    *,
    menu: Dict[str, Any],
    product: ProductDoc,
    addon_price_default: Optional[int],
    addon_price_map: Dict[str, int],
) -> str:
    """
    Render a single product as a self-contained document.
    This avoids reliance on RagFlow delimiter splitting.
    """
    menu_name = str(menu.get("name") or "菜單").strip()
    menu_times = _summarize_menu_times(menu.get("menu_times"))

    title = product.name or product.short_name or f"品項 {product.product_id}"
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append(f"- menu: {menu_name}")
    if menu_times:
        lines.append(f"- 可點餐時段：{menu_times}")
    lines.append(f"- product_id: `{product.product_id}`")
    if product.unit:
        lines.append(f"- 單位：{product.unit}")
    if product.prices:
        parts = []
        has_any_price_id = False
        for item in product.prices:
            nm, price_int, is_default, price_id = (
                (item[0], item[1], item[2], item[3])
                if len(item) >= 4
                else (item[0], item[1], item[2], None)
            )
            if price_id is not None:
                has_any_price_id = True
            part = f"{nm} {price_int}" + ("（預設）" if is_default else "")
            if price_id is not None:
                part += f"（price_id: `{price_id}`）"
            parts.append(part)
        lines.append(f"- 價格：{'；'.join(parts)}")
        if len(product.prices) == 1:
            lines.append("- 尺寸：無（單一選項，不須追問；呼叫時傳 size=M）")
            if has_any_price_id:
                lines.append("- price_id：見上方價格列，有則填、無則傳 null（禁止捏造）")
            else:
                lines.append("- price_id：無（傳 null，禁止捏造）")
        else:
            lines.append("- 尺寸：有（需追問中杯/大杯，依所選填 price_id）")
    if product.categories:
        lines.append(f"- 所屬分類：{'、'.join(sorted(set(product.categories)))}")
    if product.image_url:
        lines.append(f"- 圖片：{product.image_url}")
    if product.description:
        lines.append("")
        lines.append("## 描述")
        lines.append(product.description)
    lines.append("")

    if product.options_by_group:
        lines.append("## 可選項")
        for gid, opt in sorted(product.options_by_group.items(), key=lambda kv: kv[0]):
            if not opt:
                continue
            label = _guess_attr_group_label([o.name for o in opt])
            rendered = "、".join(
                _format_option_for_rag(
                    o,
                    group_label=label,
                    addon_price_default=addon_price_default,
                    addon_price_map=addon_price_map,
                )
                for o in opt
            )
            lines.append(f"- {label}（group_id:{gid}）：{rendered}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert menu JSON to RAG-friendly Markdown.")
    ap.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to menu JSON file (UTF-8).",
    )
    ap.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output markdown path.",
    )
    ap.add_argument(
        "--menu-index",
        type=int,
        default=0,
        help="If input is a list of menus, pick which one (default: 0).",
    )
    ap.add_argument(
        "--addon-price",
        type=int,
        default=None,
        help="Default addon surcharge for topping options (加料), e.g. 10 means (+10).",
    )
    ap.add_argument(
        "--addon-map",
        default=None,
        help="JSON file path mapping addon option to surcharge. Keys can be option name (e.g. 珍珠) or attr_item_id (e.g. 33).",
    )
    ap.add_argument(
        "--output-products-dir",
        default=None,
        help="If set, also write one markdown file per product into this directory (most robust for RagFlow).",
    )
    ap.add_argument(
        "--output-store-config",
        default=None,
        help="If set, output store config YAML (attr_groups, attr_map) to this path. Default: data/店家設定.yaml",
    )
    ap.add_argument(
        "--no-store-config",
        action="store_true",
        help="Disable automatic store config generation.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    addon_price_default: Optional[int] = args.addon_price
    addon_price_map: Dict[str, int] = {}
    if args.addon_map:
        mp = Path(str(args.addon_map))
        with mp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise SystemExit("--addon-map must be a JSON object (dict).")
        for k, v in data.items():
            try:
                addon_price_map[str(k)] = int(v)
            except Exception:
                continue

    with in_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    menu: Dict[str, Any]
    if isinstance(payload, list):
        if not payload:
            raise SystemExit("Input JSON list is empty.")
        try:
            menu = payload[int(args.menu_index)]
        except Exception as e:
            raise SystemExit(f"Invalid --menu-index: {e}")
        if not isinstance(menu, dict):
            raise SystemExit("Selected menu is not a JSON object.")
    elif isinstance(payload, dict):
        menu = payload
    else:
        raise SystemExit("Unsupported JSON top-level type (expect list or object).")

    products = _extract_products(menu)
    templates = _build_group_templates(products.values())
    md = _render_markdown(
        menu=menu,
        products=products,
        templates=templates,
        addon_price_default=addon_price_default,
        addon_price_map=addon_price_map,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")

    if args.output_products_dir:
        out_dir = Path(str(args.output_products_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        # Write one file per product to bypass delimiter chunking issues.
        for pid in sorted(products.keys()):
            p = products[pid]
            safe = _sanitize_filename(p.name or p.short_name, fallback=f"product_{pid}")
            # Keep it deterministic and unique
            filename = f"{pid:04d}_{safe}.md"
            doc = _render_product_doc(
                menu=menu,
                product=p,
                addon_price_default=addon_price_default,
                addon_price_map=addon_price_map,
            )
            (out_dir / filename).write_text(doc, encoding="utf-8")

    print(f"OK: wrote {out_path} (products: {len(products)})")
    
    # 自動生成店家設定（除非 --no-store-config）
    if not args.no_store_config:
        store_config = _extract_store_config(menu)
        if store_config["attr_groups"]:
            # 預設輸出路徑
            if args.output_store_config:
                config_path = Path(args.output_store_config)
            else:
                config_path = out_path.parent / "店家設定.yaml"
            _write_store_config(store_config, config_path)
        else:
            print("WARN: 未能從 JSON 提取 attr_groups，跳過店家設定生成")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

