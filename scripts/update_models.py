#!/usr/bin/env python3
"""
scripts/update_models.py
Discover and verify model entries in models.js using ollama.com only.
No local tooling, no model downloads, no HuggingFace dependency.

DATA FLOW
  libraries.json  — input: list of ollama library slugs to track
  models.js       — output: MODELS array, one entry per canonical size tag

  Each MODELS entry has:
    - Architecture fields (block_count, heads, etc.) from the canonical tag blob page
    - params_b and weights_gb of the default variant from the detail page
    - variants: [{tag, quantization, weights_gb}] — one per available quant variant
      tag is the raw ollama sub-tag (e.g. "3b" for default, "3b-q4_K_M" for quant variants)

  For each library, canonical size tags are scraped from:
    https://ollama.com/library/{library}/tags  (e.g. "8b", "70b", "scout")

  Quant variants are hyphenated tags starting with the canonical tag
    (e.g. "8b-instruct-q4_K_M", "8b-instruct-q8_0")

  Architecture data for each canonical tag is fetched from the blob page:
    1. https://ollama.com/library/{library}:{tag}         -> params_b, weights_gb, blob ID
    2. https://ollama.com/library/{library}:{tag}/blobs/{id} -> parse fields:
         block_count, head_count, head_count_kv,
         embedding_length, context_length, quantization
       key_length = attention.key_length if explicit, else embedding_length / head_count

  weights_gb for each quant variant is fetched from its detail page only (no blob needed).

USAGE
    python scripts/update_models.py              # verify existing entries (dry run)
    python scripts/update_models.py --apply      # write verification fixes
    python scripts/update_models.py --discover   # show missing entries
    python scripts/update_models.py --discover --apply   # scaffold missing entries
    python scripts/update_models.py --migrate    # show entries needing schema migration
    python scripts/update_models.py --migrate --apply    # migrate old-format entries
    python scripts/update_models.py --tag llama3.2:3b    # one entry only

TODO (next steps)
    1. Run --discover --apply to scaffold entries for the ~60 libraries not yet in models.js
    2. Add a --variants mode (or extend --discover) to fetch all quant variants for
       existing entries — currently each entry has only 1 variant (default quant).
       For each existing ollama_tag, call fetch_tags() to get variants_map[canonical],
       then fetch_variant_weights() for each, and patch the variants array in the entry.
    3. After models.js is complete, update app.js:
       - Dropdown 1: library+size from ollama_tag (no name field)
       - Dropdown 2: weight quantization from variants array
       - Separate KV cache quantization selector (f16/q8_0/q4_0)
       - Context slider up to context_length
       - Total VRAM = variants[selected].weights_gb + KV cache bytes
"""

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODELS_JS      = Path(__file__).parent.parent / "models.js"
LIBRARIES_JS = Path(__file__).parent.parent / "libraries.js"
OLLAMA_BASE    = "https://ollama.com"
GB_TOLERANCE   = 0.15

# Version tags like v0.1 are not canonical size tags.
_VERSION_RE = re.compile(r"^v\d", re.IGNORECASE)
# Non-size word tags are not canonical size tags.
_NONSIZE_RE = re.compile(
    r"^(instruct|text|chat|base|code|vision|uncensored|abliterated|latest|python)$",
    re.IGNORECASE,
)
# Quant keywords — used to detect quant variant tags.
_QUANT_RE = re.compile(r"q\d|fp\d|bf\d|f16|int\d|gguf|ggml", re.IGNORECASE)

_FIELD_ORDER = [
    "ollama_tag", "moe",
    "context_length", "params_b", "params_b_active",
    "block_count", "head_count", "head_count_kv",
    "embedding_length", "key_length",
    "variants",
]


# ── helpers ────────────────────────────────────────────────────────────────────

def die(msg: str) -> None:
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def http_get(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ── ollama.com scraping ────────────────────────────────────────────────────────

def fetch_tags(library: str) -> tuple[list[str], dict[str, list[str]]]:
    """
    Returns (canonical_tags, variants_map) where:
      canonical_tags  — size/name tags without quant suffixes (e.g. ["8b", "70b"])
      variants_map    — {canonical_tag: [quant_variant_tags]} (e.g. {"8b": ["8b-instruct-q4_K_M", ...]})
    """
    try:
        html = http_get(f"{OLLAMA_BASE}/library/{library}/tags").decode("utf-8", errors="replace")
    except Exception:
        return [], {}

    found = re.findall(
        rf'href=["\']?/library/{re.escape(library)}:([^"\'/?#\s]+)', html
    )
    seen: set = set()
    all_tags: list = []
    for tag in found:
        if tag not in seen and tag != "latest":
            seen.add(tag)
            all_tags.append(tag)

    canonical = [
        t for t in all_tags
        if "-" not in t
        and not _VERSION_RE.search(t)
        and not _NONSIZE_RE.match(t)
        and not _QUANT_RE.search(t)
    ]

    variants_map: dict[str, list[str]] = {c: [] for c in canonical}
    for tag in all_tags:
        if "-" not in tag:
            continue
        prefix = tag.split("-")[0]
        if prefix in variants_map:
            variants_map[prefix].append(tag)

    return canonical, variants_map


def _quant_from_tag(tag: str) -> str | None:
    """Extract quantization label from a variant tag, e.g. '8b-instruct-q4_K_M' -> 'Q4_K_M'."""
    m = re.search(r"(?:^|-)((q|f|bf)\d\w*)$", tag, re.IGNORECASE)
    return m.group(1).upper() if m else None


def fetch_blob_data(library: str, tag: str) -> dict:
    """
    Fetch full metadata for a canonical library:tag.
    Step 1 — detail page: params_b, weights_gb, blob ID.
    Step 2 — blob page: architecture fields (block_count, heads, embedding_length, etc.).
    """
    try:
        detail_html = http_get(
            f"{OLLAMA_BASE}/library/{library}:{tag}"
        ).decode("utf-8", errors="replace")
    except Exception:
        return {}

    r = _parse_detail(detail_html)

    m = re.search(
        rf'/library/{re.escape(library)}(?::{re.escape(tag)})?/blobs/([a-f0-9]+)',
        detail_html,
    )
    if not m:
        return r
    blob_id = m.group(1)

    try:
        blob_html = http_get(
            f"{OLLAMA_BASE}/library/{library}:{tag}/blobs/{blob_id}"
        ).decode("utf-8", errors="replace")
    except Exception:
        return r

    r.update(_parse_blob(blob_html))
    return r


def fetch_variant_weights(library: str, tag: str) -> dict | None:
    """
    Fetch weights_gb for a quant variant tag from its detail page.
    Returns {"tag": tag, "quantization": "Q4_K_M", "weights_gb": 4.9} or None on failure.
    tag is the raw ollama sub-tag (the part after the colon, e.g. "3b-q4_K_M").
    """
    quant = _quant_from_tag(tag)
    if not quant:
        return None
    try:
        html = http_get(f"{OLLAMA_BASE}/library/{library}:{tag}").decode("utf-8", errors="replace")
    except Exception:
        return None
    d = _parse_detail(html)
    if "weights_gb" not in d:
        return None
    return {"tag": tag, "quantization": quant, "weights_gb": d["weights_gb"]}


def _parse_detail(html: str) -> dict:
    """Parse params_b and weights_gb from the model detail page."""
    r: dict = {}
    m = re.search(r'>parameters</span><span[^>]*>([\d.]+)B</span>', html)
    if m:
        r["params_b"] = float(m.group(1))
    m = re.search(r'([\d.]+)(GB|MB)', html)
    if m:
        val = float(m.group(1))
        r["weights_gb"] = round(val / 1024 if m.group(2) == 'MB' else val, 2)
    return r


def _parse_blob(html: str) -> dict:
    """
    Parse the structured key-value table on the blob page.
    Keys follow the pattern {arch}.{field}, e.g. llama.block_count.
    Only language-backbone fields are read (vision sub-keys are skipped).
    """
    pairs = dict(re.findall(
        r'([a-z_][a-z0-9_.]+)</div>\s*<div[^>]*>\s*([^<]{1,60}?)\s*</div>',
        html, re.IGNORECASE,
    ))

    arch = pairs.get("general.architecture", "")
    p = arch + "."

    def iv(key: str):
        v = pairs.get(p + key)
        return int(v) if v and v.isdigit() else None

    r: dict = {}

    layers = iv("block_count")
    if layers:
        r["block_count"] = layers

    nh = iv("attention.head_count")
    if nh:
        r["head_count"] = nh

    nkv = iv("attention.head_count_kv")
    if nkv:
        r["head_count_kv"] = nkv

    hs = iv("embedding_length")
    if hs:
        r["embedding_length"] = hs

    ctx = iv("context_length")
    if ctx:
        r["context_length"] = ctx

    kl = iv("attention.key_length")
    if kl:
        r["key_length"] = kl
    elif hs and nh:
        r["key_length"] = hs // nh

    qt = pairs.get("general.file_type")
    if qt:
        r["default_quantization"] = qt  # used to build variants, not stored at top level

    return r


# ── models.js parsing ──────────────────────────────────────────────────────────

def parse_models_js(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    m = re.search(r"const\s+MODELS\s*=\s*(\[.*?\]);", text, re.DOTALL)
    if not m:
        die(f"Could not find 'const MODELS = [...]' in {path.name}")
    js = re.sub(r"^\s*//[^\n]*", "", m.group(1), flags=re.MULTILINE)
    js = re.sub(r",(\s*[}\]])", r"\1", js)
    try:
        return json.loads(js)
    except json.JSONDecodeError as exc:
        die(f"Failed to parse MODELS as JSON: {exc}")


def load_libraries() -> list[dict]:
    """Parse libraries.js — the single source of truth for tracked libraries.
    The file uses JSON-compatible syntax so we can strip the const wrapper and parse directly."""
    text = LIBRARIES_JS.read_text(encoding="utf-8")
    # Strip JS comments and const wrapper; file may start with // comment lines
    lines = [l for l in text.splitlines() if not l.strip().startswith("//")]
    import re as _re
    trimmed = "\n".join(lines).strip().removeprefix("const LIBRARIES =").rstrip(";").strip()
    # Strip trailing commas before ] or } (JS allows them, JSON does not)
    trimmed = _re.sub(r",(\s*[}\]])", r"\1", trimmed)
    return json.loads(trimmed)


def write_libraries_js(libs: list[dict]) -> None:
    """Write libraries.js with JSON-compatible quoted keys so the scraper can parse it back."""
    def jv(v): return "null" if v is None else f'"{v}"'
    header = (
        "// Two consumers:\n"
        "//   1. Browser — loaded as <script src=\"libraries.js\">, exposes LIBRARIES global\n"
        "//   2. Python scraper (scripts/update_models.py) — strips \"const LIBRARIES =\" wrapper and parses as JSON\n"
        "// Keys must therefore be quoted (valid JSON). Do not change to unquoted JS shorthand.\n"
    )
    lines = [header + "const LIBRARIES = ["]
    for lib in libs:
        lines.append(
            f'  {{ "library": {jv(lib["library"]):<22} '
            f'"organization": {jv(lib.get("organization")):<26} '
            f'"origin": {jv(lib.get("origin")):<20} '
            f'"source": {jv(lib.get("source"))} }},'
        )
    lines.append("];\n")
    LIBRARIES_JS.write_text("\n".join(lines), encoding="utf-8")


# ── file patching ──────────────────────────────────────────────────────────────

def _js_val(v) -> str:
    if v is None:           return "null"
    if isinstance(v, bool): return "true" if v else "false"
    if isinstance(v, str):  return f'"{v}"'
    return str(v)


def _format_entry(entry: dict) -> str:
    fields = [f for f in _FIELD_ORDER if f in entry]
    lines = ["  {"]
    for i, f in enumerate(fields):
        comma = "," if i < len(fields) - 1 else ""
        if f == "variants":
            lines.append(f'    "variants": [')
            variants = entry[f]
            for j, v in enumerate(variants):
                vc = "," if j < len(variants) - 1 else ""
                tag_part = f'"tag": "{v["tag"]}", ' if "tag" in v else ""
                lines.append(f'      {{{tag_part}"quantization": "{v["quantization"]}", "weights_gb": {v["weights_gb"]}}}{vc}')
            lines.append(f'    ]{comma}')
        else:
            lines.append(f'    "{f}": {_js_val(entry[f])}{comma}')
    lines.append("  }")
    return "\n".join(lines)


def patch_entry(path: Path, tag: str, changes: dict) -> None:
    text = path.read_text(encoding="utf-8")
    m = re.search(rf'"ollama_tag"\s*:\s*"{re.escape(tag)}"', text)
    if not m:
        print(f"    WARNING: '{tag}' not found — skipped", file=sys.stderr)
        return

    start = text.rfind("{", 0, m.start())
    depth, end = 0, start
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    block = text[start:end + 1]
    for field, val in changes.items():
        if field == "variants":
            continue  # variants patching handled separately
        jv      = _js_val(val)
        pattern = rf'("{re.escape(field)}"\s*:\s*)(?:"[^"]*"|-?\d+\.?\d*)'
        patched = re.sub(pattern, rf"\g<1>{jv}", block)
        if patched == block:
            print(f"    WARNING: '{field}' not found in block for '{tag}'")
        else:
            block = patched

    path.write_text(text[:start] + block + text[end + 1:], encoding="utf-8")


def insert_entry(path: Path, entry: dict) -> None:
    text = path.read_text(encoding="utf-8")
    close = text.rfind("];")
    if close == -1:
        print("    WARNING: could not locate MODELS closing ]; — skipped", file=sys.stderr)
        return
    before = text[:close].rstrip()
    sep = ",\n" if before.endswith("}") else "\n\n"
    path.write_text(before + sep + _format_entry(entry) + "\n" + text[close:], encoding="utf-8")


# ── discover ───────────────────────────────────────────────────────────────────

def discover(libraries: list[dict], existing: set[str], apply: bool, path: Path) -> None:
    print(f"Scanning {len(libraries)} libraries ...\n")
    added = 0

    for lib in libraries:
        library = lib["library"]
        canonical_tags, variants_map = fetch_tags(library)
        if not canonical_tags:
            print(f"  {library:<42} no canonical tags found")
            continue

        missing = [t for t in canonical_tags if f"{library}:{t}" not in existing]
        if not missing:
            print(f"  {library:<42} complete")
            continue

        for tag in missing:
            label = f"{library}:{tag}"
            print(f"  {label:<42}", end="", flush=True)

            data = fetch_blob_data(library, tag)
            if not data:
                print("no data — skipped")
                continue

            # Build variants list: default quant first, then others
            default_quant = data.pop("default_quantization", None)
            default_weights = data.pop("weights_gb", None)
            variants = []
            if default_quant and default_weights is not None:
                variants.append({"tag": tag, "quantization": default_quant, "weights_gb": default_weights})

            for vtag in variants_map.get(tag, []):
                vdata = fetch_variant_weights(library, vtag)
                if vdata and vdata["quantization"] != default_quant:
                    variants.append(vdata)

            required = ("block_count", "head_count_kv", "key_length", "context_length")
            missing_fields = [f for f in required if f not in data]
            if not variants:
                missing_fields.append("variants")

            entry = {
                "ollama_tag":   label,
                **{k: v for k, v in data.items() if k in _FIELD_ORDER},
                "variants":     variants if variants else [],
            }

            status = "OK" if not missing_fields else f"partial — missing: {', '.join(missing_fields)}"
            print(status)

            if apply:
                insert_entry(path, entry)
                added += 1
            else:
                print(_format_entry(entry))

    print(f"\n{'-' * 60}")
    if apply:
        print(f"  {added} entr{'y' if added == 1 else 'ies'} inserted.")
        print("  Review:  git diff models.js")
    else:
        print("  Dry run.  To apply:  python scripts/update_models.py --discover --apply")


# ── verify ─────────────────────────────────────────────────────────────────────

def verify(targets: list[tuple[str, dict]], apply: bool, path: Path) -> None:
    print(f"Verifying {len(targets)} entries ...\n")
    all_changes: dict[str, dict] = {}

    for tag, model in targets:
        library, variant = (tag.split(":", 1) if ":" in tag else (tag, "latest"))
        print(f"  {tag:<42}", end="", flush=True)

        data = fetch_blob_data(library, variant)
        if not data:
            print("no data")
            continue

        data.pop("default_quantization", None)
        data.pop("weights_gb", None)  # weights live in variants now

        diffs: dict = {}
        for field, new_val in data.items():
            old_val = model.get(field)
            if old_val != new_val:
                diffs[field] = new_val

        if diffs:
            print(f"{len(diffs)} change(s):")
            for f, v in diffs.items():
                print(f"      {f:<28} {str(model.get(f, '-')):<18} ->  {v}")
            all_changes[tag] = diffs
        else:
            print("OK")

    print(f"\n{'-' * 60}")
    print(f"  {len(all_changes)} model(s) with changes")

    if not all_changes:
        print("\n  models.js is up to date.")
        return

    if not apply:
        print("\n  Dry run.  To apply:  python scripts/update_models.py --apply")
        return

    for tag, changes in all_changes.items():
        patch_entry(path, tag, changes)
        print(f"    OK  {tag}")
    print("\n  Done.  Review:  git diff models.js")


# ── migrate ────────────────────────────────────────────────────────────────────

def _migrate_entry(entry: dict) -> dict:
    """Convert an old-format entry to the new schema."""
    out: dict = {}
    for f in _FIELD_ORDER:
        if f == "variants":
            # Build variants from top-level quantization + weights_gb if present.
            quant = entry.get("quantization")
            gb    = entry.get("weights_gb")
            if quant and gb is not None:
                out["variants"] = [{"quantization": quant, "weights_gb": gb}]
            elif entry.get("variants"):
                out["variants"] = entry["variants"]
        elif f in entry:
            out[f] = entry[f]
    # Drop fields not in new schema (name, config_url, quantization, weights_gb).
    return out


def migrate(models: list[dict], apply: bool, path: Path) -> None:
    needs = [m for m in models if "name" in m or "config_url" in m
             or ("quantization" in m and "variants" not in m)]
    print(f"  {len(needs)} entr{'y' if len(needs) == 1 else 'ies'} need migration\n")

    if not needs:
        print("  Nothing to migrate.")
        return

    for m in needs:
        tag = m.get("ollama_tag", "?")
        print(f"  {tag}")

    if not apply:
        print(f"\n  Dry run.  To apply:  python scripts/update_models.py --migrate --apply")
        return

    # Rewrite the entire MODELS array with migrated entries.
    migrated = [_migrate_entry(m) for m in models]
    text     = path.read_text(encoding="utf-8")
    m        = re.search(r"(const\s+MODELS\s*=\s*)\[.*?\];", text, re.DOTALL)
    if not m:
        die("Could not locate MODELS array for rewrite.")
    new_block  = "[\n\n" + ",\n".join(_format_entry(e) for e in migrated) + "\n];"
    path.write_text(text[:m.start(1)] + "const MODELS = " + new_block + text[m.end():],
                    encoding="utf-8")
    print(f"\n  Migrated {len(migrated)} entries.  Review:  git diff models.js")


# ── fetch variants ─────────────────────────────────────────────────────────────

def fetch_variants(targets: list[tuple[str, dict]], apply: bool, path: Path) -> None:
    """Fetch all quant variants for existing entries and update their variants arrays."""
    print(f"Fetching variants for {len(targets)} entries ...\n")
    updated = 0

    for tag, model in targets:
        library, canonical = tag.split(":", 1)
        print(f"  {tag:<42}", end="", flush=True)

        _, variants_map = fetch_tags(library)
        vtags = variants_map.get(canonical, [])

        existing_tags   = {v.get("tag") for v in model.get("variants", [])}
        existing_quants = {v["quantization"] for v in model.get("variants", [])}
        new_variants    = list(model.get("variants", []))

        # Ensure the default (canonical) variant exists at index 0.
        if canonical not in existing_tags:
            data = fetch_blob_data(library, canonical)
            default_quant   = data.pop("default_quantization", None)
            default_weights = data.pop("weights_gb", None)
            if default_quant and default_weights is not None:
                new_variants.insert(0, {"tag": canonical, "quantization": default_quant, "weights_gb": default_weights})
                existing_quants.add(default_quant)

        for vtag in vtags:
            vdata = fetch_variant_weights(library, vtag)
            if vdata and vdata["quantization"] not in existing_quants:
                new_variants.append(vdata)
                existing_quants.add(vdata["quantization"])

        added = len(new_variants) - len(model.get("variants", []))
        if added == 0 and not vtags:
            print("no new variants")
            continue
        print(f"{len(new_variants)} variants ({added:+d})")

        if apply and added > 0:
            _patch_variants(path, tag, new_variants)
            updated += 1

    print(f"\n{'-' * 60}")
    if apply:
        print(f"  {updated} entr{'y' if updated == 1 else 'ies'} updated.  Review:  git diff models.js")
    else:
        print("  Dry run.  To apply:  python scripts/update_models.py --variants --apply")


def _patch_variants(path: Path, tag: str, variants: list) -> None:
    """Replace the variants array for the given ollama_tag in models.js."""
    text = path.read_text(encoding="utf-8")
    m = re.search(rf'"ollama_tag"\s*:\s*"{re.escape(tag)}"', text)
    if not m:
        print(f"    WARNING: '{tag}' not found", file=sys.stderr)
        return

    # Find the variants array within this entry's block.
    start = text.rfind("{", 0, m.start())
    depth, end = 0, start
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    block = text[start:end + 1]
    # Build new variants lines (no leading indent — the existing whitespace before "variants" is preserved by re.sub).
    var_lines = ['"variants": [']
    for j, v in enumerate(variants):
        vc = "," if j < len(variants) - 1 else ""
        tag_part = f'"tag": "{v["tag"]}", ' if "tag" in v else ""
        var_lines.append(f'      {{{tag_part}"quantization": "{v["quantization"]}", "weights_gb": {v["weights_gb"]}}}{vc}')
    var_lines.append("    ]")
    new_variants_str = "\n".join(var_lines)

    # Replace existing variants block.
    patched = re.sub(
        r'"variants"\s*:\s*\[.*?\]',
        new_variants_str,
        block,
        flags=re.DOTALL,
    )
    path.write_text(text[:start] + patched + text[end + 1:], encoding="utf-8")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    argv         = sys.argv[1:]
    apply_flag   = "--apply"    in argv
    do_discover  = "--discover" in argv
    do_migrate   = "--migrate"  in argv
    do_variants  = "--variants" in argv
    tag_filter   = argv[argv.index("--tag") + 1] if "--tag" in argv else None

    print("will-it-llm — model metadata updater\n")
    print(f"Parsing {MODELS_JS.name} ...")
    models  = parse_models_js(MODELS_JS)
    targets = [(m["ollama_tag"], m) for m in models if m.get("ollama_tag")]

    if do_migrate:
        migrate(models, apply_flag, MODELS_JS)
    elif do_discover:
        libraries = load_libraries()
        existing  = {t for t, _ in targets}
        discover(libraries, existing, apply_flag, MODELS_JS)
    elif do_variants:
        if tag_filter:
            targets = [(t, m) for t, m in targets if t == tag_filter]
            if not targets:
                die(f"No model with ollama_tag '{tag_filter}' found.")
        fetch_variants(targets, apply_flag, MODELS_JS)
    else:
        if tag_filter:
            targets = [(t, m) for t, m in targets if t == tag_filter]
            if not targets:
                die(f"No model with ollama_tag '{tag_filter}' found.")
        verify(targets, apply_flag, MODELS_JS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(1)
