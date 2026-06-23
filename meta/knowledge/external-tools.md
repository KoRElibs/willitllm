# External tools — official docs & links

Used in `coder.html` / `coder.js`. **Keep these current** — check links here before changing
any UI copy that references an external tool, and update if a URL moves or a tool ships a
config-breaking change.

**IDE market share** (Stack Overflow Developer Survey 2025, source: survey.stackoverflow.co/2025):
VS Code 75.9% · Cursor 17.9% · Neovim 14% · VSCodium 6.2% · Zed 7.3% · Windsurf 4.9%
JetBrains: IntelliJ 27.1% · PyCharm 15% · WebStorm 7.6% · Rider 7.1% · PhpStorm 5.8%

VS Code-family (VS Code + Cursor + VSCodium + Windsurf — all run VS Code extensions): ~85% coverage.

---

## Editors

### VS Code

| Resource | URL |
| --- | --- |
| Download | [code.visualstudio.com](https://code.visualstudio.com) |

The reference editor. Extensions install via the Extensions panel (Ctrl+Shift+X / Cmd+Shift+X).

### Cursor

| Resource | URL |
| --- | --- |
| Download | [cursor.com/download](https://cursor.com/download) |

VS Code fork with integrated AI (Cursor Tab, Composer, Agent mode). **Important: Cursor's built-in AI is cloud-only and requires a paid subscription.** For local Ollama models, install the Continue VS Code extension inside Cursor — it works identically to VS Code. The Cline extension also works in Cursor.

### Windsurf

Codeium's AI-focused IDE (also a VS Code fork). Built-in "Cascade" agent is cloud-only. Continue extension works inside Windsurf for local Ollama models.

### VSCodium

Open-source VS Code without Microsoft telemetry. VS Code extensions work via the Open VSX registry. Continue (`Continue.continue`) is on Open VSX.

### JetBrains IDEs

IntelliJ IDEA, PyCharm, WebStorm, Rider, PhpStorm, GoLand, etc. Continue has a JetBrains plugin (ID: 22707). JetBrains launched their own agent **Junie** in late 2025 (cloud-only). For local Ollama models, use Continue. Config file is the same `~/.continue/config.json`.

### Zed

Rust-based editor, 7.3% share (2025). Built-in AI is cloud-only. No confirmed Continue/Ollama support — skip for now, revisit.

---

## Ollama

| Resource | URL |
|---|---|
| Install / download | [ollama.com](https://ollama.com) |
| Docs (general) | [docs.ollama.com](https://docs.ollama.com) |
| Linux setup | [docs.ollama.com/linux](https://docs.ollama.com/linux) |
| macOS setup | [docs.ollama.com/macos](https://docs.ollama.com/macos) |
| Windows setup | [docs.ollama.com/windows](https://docs.ollama.com/windows) |
| Model library | [ollama.com/library](https://ollama.com/library) |

**Key facts:**
- `OLLAMA_HOST=0.0.0.0` required for remote access (binds only to loopback by default)
- `OLLAMA_KV_CACHE_TYPE` sets KV precision: `q8_0` | `q4_0` (default is `f16`)
- Linux: set env vars via systemd drop-in at `/etc/systemd/system/ollama.service.d/override.conf`
- macOS: export in `~/.zshrc`, quit menu bar app, run `ollama serve` from terminal
- Windows: `setx` or Settings → Environment Variables, then restart from tray

---

## Cline

| Resource | URL |
| --- | --- |
| VS Code Marketplace | [marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev](https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev) |
| Docs | [docs.cline.bot](https://docs.cline.bot) |
| Ollama setup | [docs.cline.bot/running-models-locally/ollama](https://docs.cline.bot/running-models-locally/ollama) |
| GitHub | [github.com/cline/cline](https://github.com/cline/cline) |

**Extension ID:** `saoudrizwan.claude-dev` (publisher: saoudrizwan — the extension was originally called "Claude Dev")  
**Install via CLI:** `code --install-extension saoudrizwan.claude-dev`

**Key facts:**
- VS Code extension for agentic coding loops (multi-step planning, file edits, shell commands)
- Configured through its own settings UI — not a JSON file
- Access settings: click the **⚙️ gear icon** in the Cline sidebar panel
- Ollama provider setup: API Provider → Ollama, Base URL → `http://localhost:11434`, Model → dropdown, Context Window → `num_ctx` value
- **Model field is a dropdown** of models already pulled to ollama — `ollama pull <model>` first or it won't appear
- **Recommended for local models**: ⚙️ → Features → enable **Use Compact Prompt** (reduces token overhead for local inference)
- "Context Window" in Cline maps to `num_ctx` in the ollama API request — no manual `/set parameter` needed

---

## Continue

| Resource | URL |
| --- | --- |
| VS Code Marketplace | [marketplace.visualstudio.com/items?itemName=Continue.continue](https://marketplace.visualstudio.com/items?itemName=Continue.continue) |
| JetBrains Marketplace | [plugins.jetbrains.com/plugin/22707-continue](https://plugins.jetbrains.com/plugin/22707-continue) |
| Website | [continue.dev](https://www.continue.dev) |
| Docs | [docs.continue.dev](https://docs.continue.dev) |
| Config reference | [docs.continue.dev/reference/config](https://docs.continue.dev/reference/config) |
| GitHub | [github.com/continuedev/continue](https://github.com/continuedev/continue) |

**Extension ID (VS Code):** `Continue.continue`  
**Install via CLI:** `code --install-extension Continue.continue`

**Key facts:**
- VS Code and JetBrains extension for coding assistance and tab autocomplete
- Config file: `~/.continue/config.json` (editable via the extension gear icon)
- Chat/code model entry goes in the `models` array; `contextLength` maps to `num_ctx`
- Autocomplete model entry goes in `tabAutocompleteModel` (top-level key, not in `models`)
- Example `tabAutocompleteModel` entry:

```json
{
  "tabAutocompleteModel": {
    "title": "starcoder2:3b",
    "provider": "ollama",
    "model": "starcoder2:3b",
    "apiBase": "http://localhost:11434",
    "contextLength": 16384
  }
}
```
