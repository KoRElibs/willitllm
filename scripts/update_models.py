#!/usr/bin/env python3
"""
scripts/update_models.py
Verify and update model metadata in models.js — without downloading model weights.

Data is fetched from two lightweight remote sources:

  1. HuggingFace config.json  (config_url field in models.js)
       → layers, num_attention_heads, num_key_value_heads,
         hidden_size, head_dim, max_context
       Request size: ~5 KB per model.
       Gated models (Meta Llama etc.) return 401 — those fields are skipped.

  2. Ollama registry manifest  (registry.ollama.ai)
       → weights_gb (exact layer size in bytes)
       → quantization (from the small metadata blob in the manifest)
       Request size: ~2–3 KB per model.  No model weights are downloaded.

Ollama does not need to be running and no models need to be pulled.
The only requirement is internet access and ollama being installed (for the
version check). Architecture verification is entirely network-based.

USAGE
    python scripts/update_models.py              # dry run — show diffs
    python scripts/update_models.py --apply      # write changes to models.js
    python scripts/update_models.py --tag llama3.2:1b   # one model only
    python scripts/update_models.py --hf-only    # skip registry, HuggingFace only
    python scripts/update_models.py --reg-only   # skip HuggingFace, registry only

REQUIREMENTS
    ollama installed (not necessarily running):  https://ollama.com/download
    Internet access to huggingface.co and registry.ollama.ai
"""

import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ── configuration ─────────────────────────────────────────────────────────────

MODELS_JS       = Path(__file__).parent.parent / "models.js"
OLLAMA_REGISTRY = "https://registry.ollama.ai"
HF_BASE         = "https://huggingface.co"

# HuggingFace config.json field → models.js field
HF_TO_FIELD = {
    "num_hidden_layers":      "layers",
    "num_attention_heads":    "num_attention_heads",
    "num_key_value_heads":    "num_key_value_heads",
    "hidden_size":            "hidden_size",
    "head_dim":               "head_dim",          # explicit (Mistral/Gemma families)
    "max_position_embeddings":"max_context",
}

# Differences in weights_gb smaller than this are treated as noise.
GB_TOLERANCE = 0.15

# Ollama layer media types
LAYER_MODEL    = "application/vnd.ollama.image.model"
LAYER_MANIFEST = ("application/vnd.oci.image.manifest.v1+json,"
                  "application/vnd.docker.distribution.manifest.v2+json")


# ── error handling ────────────────────────────────────────────────────────────

def die(msg: str) -> None:
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def check_ollama_installed() -> None:
    try:
        r = subprocess.run(["ollama", "version"], capture_output=True, text=True, timeout=5)
        print(f"  ollama installed  ({(r.stdout + r.stderr).strip()})")
    except FileNotFoundError:
        die("ollama is not installed or not in PATH.\n"
            "         Install from:  https://ollama.com/download")
    except subprocess.TimeoutExpired:
        die("'ollama version' timed out.")


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def http_get(url: str, headers: dict | None = None, timeout: int = 10) -> tuple[bytes, dict]:
    """GET a URL. Returns (body_bytes, response_headers). Raises HTTPError on non-2xx."""
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), dict(r.headers)


def http_get_json(url: str, headers: dict | None = None) -> dict | None:
    try:
        body, _ = http_get(url, headers)
        return json.loads(body)
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
        return None


# ── HuggingFace source ────────────────────────────────────────────────────────

# Return values for fetch_hf_config — lets the caller distinguish failure causes.
HF_OK      = "ok"       # fetched and parsed successfully
HF_GATED   = "gated"   # 401 — model exists but requires HuggingFace login
HF_MISSING = "missing"  # 404 — URL is wrong or repo doesn't exist
HF_ERROR   = "error"    # network timeout, bad JSON, or other unexpected error


def fetch_hf_config(config_url: str) -> tuple[str, dict]:
    """
    Fetch architecture fields from a HuggingFace config.json URL.

    Returns (status, fields) where status is one of HF_OK / HF_GATED /
    HF_MISSING / HF_ERROR and fields is a {js_field: value} dict
    (empty on any non-OK status).

    Distinguishing 401 (gated) from 404 (bad URL) matters because they call
    for different responses: gated models are expected and harmless; a 404
    means the config_url in models.js is wrong and should be corrected.
    """
    if not config_url or not config_url.startswith("https://huggingface.co"):
        return HF_ERROR, {}

    # Convert /blob/ URL → /resolve/ (returns raw file content, not HTML page).
    raw_url = config_url.replace("/blob/", "/resolve/")

    try:
        body, _ = http_get(raw_url)
        data = json.loads(body)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return HF_GATED, {}
        if exc.code == 404:
            return HF_MISSING, {}
        return HF_ERROR, {}
    except (urllib.error.URLError, json.JSONDecodeError, Exception):
        return HF_ERROR, {}

    if not data:
        return HF_ERROR, {}

    result: dict = {}
    for hf_key, js_field in HF_TO_FIELD.items():
        val = data.get(hf_key)
        if val is not None:
            result[js_field] = int(val)

    # Derive head_dim if config.json doesn't state it explicitly.
    if "head_dim" not in result:
        hs = result.get("hidden_size")
        nh = result.get("num_attention_heads")
        if hs and nh:
            result["head_dim"] = hs // nh

    return HF_OK, result


# ── Ollama registry source ────────────────────────────────────────────────────

def _registry_token(model_name: str) -> str | None:
    """
    Fetch an anonymous bearer token for the ollama registry.
    Ollama uses a standard OCI token service — we trigger a 401 to discover
    the auth endpoint, then request an anonymous pull token.
    """
    probe_url = f"{OLLAMA_REGISTRY}/v2/library/{model_name}/manifests/latest"
    try:
        http_get(probe_url, headers={"Accept": LAYER_MANIFEST})
        return None   # No auth needed (shouldn't happen, but handle it)
    except urllib.error.HTTPError as exc:
        if exc.code != 401:
            return None
        www_auth = exc.headers.get("WWW-Authenticate", "")

    realm   = re.search(r'realm="([^"]+)"',   www_auth)
    service = re.search(r'service="([^"]+)"', www_auth)
    if not realm:
        return None

    token_url = (
        f"{realm.group(1)}"
        f"?service={service.group(1) if service else ''}"
        f"&scope=repository:library/{model_name}:pull"
    )
    try:
        data = http_get_json(token_url)
        return data.get("token") if data else None
    except Exception:
        return None


def fetch_registry_data(ollama_tag: str) -> dict:
    """
    Fetch weights_gb and quantization from the ollama registry manifest.
    No model weights are downloaded — only the manifest (~2 KB) and the
    small metadata config blob (~1 KB).

    Returns a (possibly empty) dict with "weights_gb" and/or "quantization".
    """
    # Split "model:tag" → model name + tag
    if ":" in ollama_tag:
        model_name, tag = ollama_tag.split(":", 1)
    else:
        model_name, tag = ollama_tag, "latest"

    token = _registry_token(model_name)
    auth_headers = {"Authorization": f"Bearer {token}"} if token else {}

    # ── fetch manifest ────────────────────────────────────────────────────────
    manifest_url = f"{OLLAMA_REGISTRY}/v2/library/{model_name}/manifests/{tag}"
    try:
        manifest = http_get_json(
            manifest_url,
            headers={**auth_headers, "Accept": LAYER_MANIFEST},
        )
    except Exception:
        return {}

    if not manifest:
        return {}

    result: dict = {}
    config_digest: str | None = None

    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == LAYER_MODEL:
            size_bytes = layer.get("size")
            if size_bytes:
                result["weights_gb"] = round(size_bytes / 1024 ** 3, 1)

    # The manifest "config" object is a small metadata blob (not the weights).
    cfg = manifest.get("config", {})
    config_digest = cfg.get("digest")

    # ── fetch config blob for quantization ───────────────────────────────────
    if config_digest:
        blob_url = f"{OLLAMA_REGISTRY}/v2/library/{model_name}/blobs/{config_digest}"
        try:
            blob_data = http_get_json(blob_url, headers=auth_headers)
            if blob_data:
                # Ollama stores model metadata under various keys depending on version.
                quant = (
                    blob_data.get("quantization")
                    or blob_data.get("file_type")
                    or _extract_quant_from_blob(blob_data)
                )
                if quant:
                    result["quantization"] = quant
        except Exception:
            pass   # quantization is optional — don't fail the whole check

    return result


def _extract_quant_from_blob(blob: dict) -> str | None:
    """
    Try common locations for quantization info in the ollama config blob.
    The blob format varies slightly between ollama versions.
    """
    # Some versions nest it under "config" or "model"
    for key in ("config", "model", "details"):
        sub = blob.get(key)
        if isinstance(sub, dict):
            q = sub.get("quantization") or sub.get("quantization_level") or sub.get("file_type")
            if q:
                return q
    return None


# ── models.js parser ──────────────────────────────────────────────────────────

def parse_models_js(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"const\s+MODELS\s*=\s*(\[.*?\]);", text, re.DOTALL)
    if not match:
        die(f"Could not find 'const MODELS = [...]' in {path.name}")

    js_array = match.group(1)
    js_array = re.sub(r"//[^\n]*", "", js_array)         # strip JS comments
    js_array = re.sub(r",(\s*[}\]])", r"\1", js_array)   # strip trailing commas

    try:
        return json.loads(js_array)
    except json.JSONDecodeError as exc:
        die(f"Failed to parse models array as JSON: {exc}")


# ── in-place file patcher ─────────────────────────────────────────────────────

def patch_model_in_file(path: Path, tag: str, changes: dict) -> None:
    """
    Targeted in-place update of one model block in models.js.
    Preserves all JS comments and formatting.
    """
    text = path.read_text(encoding="utf-8")

    m = re.search(rf'"ollama_tag"\s*:\s*"{re.escape(tag)}"', text)
    if not m:
        print(f"    WARNING: '{tag}' not found in file — skipped", file=sys.stderr)
        return

    open_brace = text.rfind("{", 0, m.start())
    if open_brace == -1:
        print(f"    WARNING: no opening brace for '{tag}' — skipped", file=sys.stderr)
        return

    depth, close_brace = 0, open_brace
    for i, ch in enumerate(text[open_brace:], start=open_brace):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                close_brace = i
                break

    new_block = text[open_brace : close_brace + 1]

    for field, new_val in changes.items():
        json_val    = f'"{new_val}"' if isinstance(new_val, str) else repr(new_val)
        pattern     = rf'("{re.escape(field)}"\s*:\s*)(?:"[^"]*"|-?\d+\.?\d*)'
        patched     = re.sub(pattern, rf"\g<1>{json_val}", new_block)
        if patched == new_block:
            print(f"    WARNING: field '{field}' not in block for '{tag}' — edit manually")
        else:
            new_block = patched

    text = text[:open_brace] + new_block + text[close_brace + 1:]
    path.write_text(text, encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    argv      = sys.argv[1:]
    apply_flag   = "--apply"    in argv
    hf_only      = "--hf-only"  in argv
    reg_only     = "--reg-only" in argv
    tag_filter   = None

    if "--tag" in argv:
        idx = argv.index("--tag")
        if idx + 1 >= len(argv):
            die("--tag requires an argument, e.g.  --tag llama3.2:1b")
        tag_filter = argv[idx + 1]

    print("will-it-fit — model metadata updater\n")
    print("Checking ollama …")
    check_ollama_installed()

    print(f"\nParsing {MODELS_JS.name} …")
    models  = parse_models_js(MODELS_JS)
    targets = [(m["ollama_tag"], m) for m in models if m.get("ollama_tag")]

    if tag_filter:
        targets = [(t, m) for t, m in targets if t == tag_filter]
        if not targets:
            die(f"No model with ollama_tag '{tag_filter}' found in {MODELS_JS.name}")

    print(f"Checking {len(targets)} models (no downloads — registry + HuggingFace only)\n")

    all_changes:  dict[str, dict] = {}
    bad_urls:     list[tuple[str, str]] = []   # (tag, config_url) — HF returned 404
    col = 42

    for tag, model in targets:
        print(f"  {tag:<{col}}", end="", flush=True)

        fetched: dict = {}
        notes:   list = []

        # Source 1 — HuggingFace config.json (architecture params)
        if not reg_only:
            config_url = model.get("config_url", "")
            hf_status, hf_data = fetch_hf_config(config_url)
            if hf_status == HF_OK:
                fetched.update(hf_data)
            elif hf_status == HF_GATED:
                notes.append("HF gated (arch params not verified — login required)")
            elif hf_status == HF_MISSING:
                notes.append("HF 404 — config_url may be wrong (flagged below)")
                bad_urls.append((tag, config_url))
            elif hf_status == HF_ERROR:
                notes.append("HF unreachable")

        # Source 2 — Ollama registry manifest (weights_gb + quantization)
        if not hf_only:
            reg_data = fetch_registry_data(tag)
            fetched.update(reg_data)
            if not reg_data and hf_only is False:
                notes.append("registry: no data (tag may not exist in ollama library)")

        if not fetched:
            print("  ".join(notes) if notes else "no data returned")
            continue

        # Compute diff against current models.js values.
        diffs: dict = {}
        for field, new_val in fetched.items():
            old_val = model.get(field)
            if field == "weights_gb":
                if old_val is None or abs(float(new_val) - float(old_val)) >= GB_TOLERANCE:
                    diffs[field] = new_val
            elif old_val != new_val:
                diffs[field] = new_val

        if diffs:
            print(f"{len(diffs)} change(s):")
            for f, v in diffs.items():
                print(f"      {f:<28} {str(model.get(f, '—')):<18} →  {v}")
            all_changes[tag] = diffs
        else:
            print("OK")

    # ── summary ───────────────────────────────────────────────────────────────

    print(f"\n{'─' * 62}")
    print(f"  {len(all_changes)} model(s) with changes")

    # Surface bad config_url values prominently — these need manual correction.
    if bad_urls:
        print(f"\n  ⚠  {len(bad_urls)} config_url(s) returned HTTP 404 from HuggingFace.")
        print("     These URLs were generated from AI training knowledge and may be wrong.")
        print("     Check each URL and correct it in models.js:\n")
        for bad_tag, bad_url in bad_urls:
            print(f"     {bad_tag}")
            print(f"       {bad_url or '(no config_url set)'}")

    if not all_changes:
        if not bad_urls:
            print("\n  models.js is up to date.")
        return

    if not apply_flag:
        print(f"\n  Dry run — nothing written.")
        print(f"  To write changes:  python scripts/update_models.py --apply")
        return

    print(f"\n  Applying changes to {MODELS_JS.name} …")
    for tag, changes in all_changes.items():
        patch_model_in_file(MODELS_JS, tag, changes)
        print(f"    ✓  {tag}")

    print("\n  Done.")
    print("  Review changes before committing:  git diff models.js")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(1)
