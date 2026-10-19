"""Hardware detection, local model selection and runtime fingerprints for Ollama.

``avd local doctor`` reports what this machine can run and recommends models; it never downloads
anything. ``avd local pull`` downloads one model only when ``--approve-gb`` covers its estimated
size. Detection reads the OS (``/proc``, ``sysctl``) and vendor tools when present
(``nvidia-smi``, ``rocm-smi``); it does not assume NVIDIA, and a machine without a GPU is
supported (CPU inference, slower).

Memory figures for models that are *not installed* are estimates from parameter count and
quantization (weights) plus a context allowance (KV cache, compute buffers). They are labelled as
estimates; installed models report their real size, and ``/api/ps`` the memory actually used.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from avd.config import Settings

GB = 1024**3

# ----------------------------------------------------------------------------------- hardware


@dataclass
class GPU:
    vendor: str  # nvidia | amd | apple | other
    name: str
    vram_total_gb: float | None
    vram_free_gb: float | None
    driver: str | None = None
    runtime: str | None = None  # "CUDA 12.8", "ROCm 6.4", "Metal"


@dataclass
class Hardware:
    os: str
    arch: str
    cpu_model: str
    cpu_logical: int
    cpu_flags: list[str]
    ram_total_gb: float
    ram_available_gb: float
    swap_total_gb: float
    gpus: list[GPU]
    disk_free_gb: float
    disk_path: str
    ollama_binary: str | None
    notes: list[str] = field(default_factory=list)

    @property
    def accelerator(self) -> str:
        if not self.gpus:
            return "cpu"
        return self.gpus[0].vendor

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["accelerator"] = self.accelerator
        return d


def _run(cmd: list[str], timeout: float = 5.0) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def _meminfo() -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            out[k] = int(v.split()[0]) * 1024 / GB
    except (OSError, ValueError, IndexError):
        pass
    return out


def _os_name() -> str:
    try:
        rel = dict(
            line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line
        )
        return rel.get("PRETTY_NAME", "").strip('"') or platform.platform()
    except OSError:
        return platform.platform()


def _nvidia() -> list[GPU]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []
    header = _run(["nvidia-smi"]) or ""
    m = re.search(r"CUDA Version:\s*([\d.]+)", header)
    gpus = []
    for line in out.strip().splitlines():
        name, total, free, driver = (p.strip() for p in line.split(","))
        gpus.append(
            GPU(
                "nvidia",
                name,
                float(total) / 1024,
                float(free) / 1024,
                driver,
                f"CUDA {m.group(1)}" if m else None,
            )
        )
    return gpus


def _amd() -> list[GPU]:
    out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if not out:
        return []
    try:
        data = json.loads(out)
    except ValueError:
        return []
    version = None
    with contextlib.suppress(OSError):
        version = "ROCm " + Path("/opt/rocm/.info/version").read_text().strip()
    gpus = []
    for card in data.values():
        if not isinstance(card, dict):
            continue
        total = card.get("VRAM Total Memory (B)")
        used = card.get("VRAM Total Used Memory (B)")
        t = float(total) / GB if total else None
        gpus.append(
            GPU(
                "amd",
                str(card.get("Card series") or card.get("Card SKU") or "AMD GPU"),
                t,
                (t - float(used) / GB) if t is not None and used else None,
                None,
                version,
            )
        )
    return gpus


def detect_hardware(models_dir: Path | None = None) -> Hardware:
    system = platform.system()
    notes: list[str] = []
    cpu_model, flags = platform.processor() or platform.machine(), []
    ram_total = ram_avail = swap = 0.0
    gpus: list[GPU] = []
    if system == "Linux":
        try:
            info = Path("/proc/cpuinfo").read_text()
            if m := re.search(r"^model name\s*:\s*(.+)$", info, re.M):
                cpu_model = m.group(1).strip()
            if m := re.search(r"^flags\s*:\s*(.+)$", info, re.M):
                wanted = {"avx", "avx2", "avx512f", "avx512_vnni", "avx512_bf16", "amx_tile", "fma", "f16c"}
                flags = sorted(wanted & set(m.group(1).split()))
        except OSError:
            pass
        mi = _meminfo()
        ram_total, ram_avail, swap = (
            mi.get("MemTotal", 0.0),
            mi.get("MemAvailable", 0.0),
            mi.get("SwapTotal", 0.0),
        )
        gpus = _nvidia() or _amd()
        if not gpus and Path("/dev/dri").exists():
            notes.append(
                "/dev/dri present: an integrated or unsupported GPU may exist (Ollama uses CPU unless Vulkan is enabled)"
            )
    elif system == "Darwin":
        cpu_model = (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or cpu_model).strip()
        mem = _run(["sysctl", "-n", "hw.memsize"])
        ram_total = int(mem) / GB if mem and mem.strip().isdigit() else 0.0
        ram_avail = ram_total  # macOS has no MemAvailable; Ollama budgets ~2/3 of unified memory for Metal
        if platform.machine() == "arm64":
            gpus = [
                GPU("apple", cpu_model, round(ram_total * 2 / 3, 1), None, None, "Metal (unified memory)")
            ]
    elif system == "Windows":
        gpus = _nvidia()
        notes.append("Windows: RAM figures unavailable without psutil; GPU via nvidia-smi only")
    if not gpus:
        notes.append(
            "no supported GPU detected: inference runs on CPU (expect a few tokens/s for 4-9B models)"
        )
    if swap == 0 and system == "Linux":
        notes.append("no swap: a model that does not fit in RAM is killed rather than slowed")
    mdir = models_dir or Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))
    probe = mdir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    disk = shutil.disk_usage(probe)
    return Hardware(
        os=_os_name(),
        arch=platform.machine(),
        cpu_model=cpu_model,
        cpu_logical=os.cpu_count() or 1,
        cpu_flags=flags,
        ram_total_gb=round(ram_total, 1),
        ram_available_gb=round(ram_avail, 1),
        swap_total_gb=round(swap, 1),
        gpus=gpus,
        disk_free_gb=round(disk.free / GB, 1),
        disk_path=str(probe),
        ollama_binary=shutil.which("ollama"),
        notes=notes,
    )


# ----------------------------------------------------------------------------------- models


@dataclass(frozen=True)
class Candidate:
    """A local model to consider. ``tag`` is the Ollama library tag to pull; confirm it on the
    library page before pulling (tags are not verified from this environment)."""

    tag: str
    family: str
    params_b: float
    active_params_b: float  # = params_b for dense models; smaller for mixture-of-experts
    vision: bool
    note: str = ""


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("qwen3.5:4b", "qwen3.5", 4.0, 4.0, True, "smallest; CPU-friendly baseline"),
    Candidate("qwen3.5:9b", "qwen3.5", 9.0, 9.0, True, "quality/speed middle"),
    Candidate("qwen3.5:27b", "qwen3.5", 27.0, 27.0, True, "dense; needs a large GPU or lots of RAM"),
    Candidate("qwen3.5:35b-a3b", "qwen3.5", 35.0, 3.0, True, "mixture-of-experts: big memory, fast decoding"),
)

# Bits per weight, including quantization scales (llama.cpp K-quants).
BITS_PER_WEIGHT = {"q4_K_M": 4.85, "q8_0": 8.5, "bf16": 16.0}
# Conservative allowance for KV cache + compute buffers per 1K context tokens, per billion params.
# Real use is often lower (grouped-query and linear-attention layers); installed models are measured.
CONTEXT_GB_PER_1K_PER_B = 0.004
RUNTIME_OVERHEAD_GB = 0.8
# Memory left to the OS, the API, the database and the browser on a CPU-only machine.
HOST_RESERVE_GB = 3.0


def estimate_memory_gb(c: Candidate, quant: str = "q4_K_M", num_ctx: int = 16_384) -> dict[str, float]:
    weights = c.params_b * BITS_PER_WEIGHT[quant] / 8
    ctx = c.params_b * CONTEXT_GB_PER_1K_PER_B * num_ctx / 1024
    return {
        "weights_gb": round(weights, 1),
        "context_gb": round(ctx, 1),
        "total_gb": round(weights + ctx + RUNTIME_OVERHEAD_GB, 1),
        "download_gb": round(weights, 1),
    }


def recommend(hw: Hardware, num_ctx: int = 16_384, installed: set[str] | None = None) -> dict[str, Any]:
    """Which candidates fit, and which to start with. Fit is judged against VRAM when a GPU is
    present (partial offload works but is much slower), else against available RAM minus a host
    reserve. On CPU, decoding speed scales with *active* parameters, so the recommendation prefers
    small dense or MoE models there."""
    installed = installed or set()
    gpu = hw.gpus[0] if hw.gpus else None
    if gpu and gpu.vram_total_gb:
        budget, against = (gpu.vram_free_gb or gpu.vram_total_gb) - 0.5, f"{gpu.vendor} VRAM"
    else:
        budget, against = hw.ram_available_gb - HOST_RESERVE_GB, "RAM (CPU inference)"
    rows: list[dict[str, Any]] = []
    for c in CANDIDATES:
        for quant in ("q4_K_M", "q8_0"):
            est = estimate_memory_gb(c, quant, num_ctx)
            fits = est["total_gb"] <= budget
            disk_ok = est["download_gb"] + 1 <= hw.disk_free_gb
            rows.append(
                {
                    "tag": c.tag if quant == "q4_K_M" else f"{c.tag}-{quant.lower()}",
                    "quantization": quant,
                    "params_b": c.params_b,
                    "active_params_b": c.active_params_b,
                    "vision": c.vision,
                    **est,
                    "fits_memory": fits,
                    "fits_disk": disk_ok,
                    "installed": c.tag in installed,
                    "estimate": True,
                    "note": c.note,
                }
            )
    usable = [r for r in rows if r["fits_memory"] and r["fits_disk"] and r["quantization"] == "q4_K_M"]
    if gpu is None:
        # CPU: tokens/s ~ memory bandwidth / active bytes; keep active params small.
        usable = [r for r in usable if r["active_params_b"] <= 10]
    if gpu is None:
        primary = min(
            usable, key=lambda r: (r["active_params_b"], r["params_b"]), default=None
        )  # speed first
    else:
        primary = max(usable, key=lambda r: r["params_b"], default=None)  # quality first
    bench = sorted(
        {r["tag"] for r in usable}, key=lambda t: next(r["params_b"] for r in usable if r["tag"] == t)
    )
    return {
        "memory_budget_gb": round(budget, 1),
        "judged_against": against,
        "num_ctx": num_ctx,
        "primary": primary["tag"] if primary else None,
        "benchmark_candidates": bench[:3],
        "rows": rows,
        "caveat": (
            "Sizes for models that are not installed are estimates (parameter count x quantization + "
            "context allowance). Confirm tags on the Ollama library page; `ollama show <tag>` and "
            "`/api/ps` report the real numbers after a pull."
        ),
    }


# ----------------------------------------------------------------------------------- runtime


def provider_from_settings(settings: Settings) -> Any:
    from avd.llm.ollama import OllamaProvider

    return OllamaProvider(
        settings.ollama_base_url,
        num_ctx=settings.ollama_num_ctx,
        keep_alive=settings.ollama_keep_alive,
        think=settings.ollama_think,
        allow_remote=settings.ollama_allow_remote,
        timeout=settings.llm_timeout_seconds,
        seed=settings.ollama_seed,
    )


def runtime_snapshot(settings: Settings, provider: Any | None = None) -> dict[str, Any]:
    """Everything needed to reproduce a local run: server version, model tag + digest +
    quantization, context settings and the hardware it ran on. Raises an ``LLMError`` when the
    server is down or the model is missing."""
    from avd.llm.ollama import OllamaProvider

    p = provider if isinstance(provider, OllamaProvider) else provider_from_settings(settings)
    models = {}
    for m in sorted({settings.llm_model_stage_b, settings.llm_model_stage_c}):
        info = p.model_info(m)
        models[m] = {
            "fingerprint": info.fingerprint,
            "digest": info.digest,
            "parameter_size": info.parameter_size,
            "quantization": info.quantization,
            "family": info.family,
            "native_context": info.context_length,
            "effective_context": p.effective_context(m),
            "capabilities": list(info.capabilities),
        }
    return {
        "ollama_version": p.version(),
        "base_url": p.base_url,
        "models": models,
        "num_ctx": settings.ollama_num_ctx,
        "keep_alive": settings.ollama_keep_alive,
        "think": settings.ollama_think,
        "max_concurrency": settings.llm_max_concurrency,
        "single_pass_max_tokens": settings.llm_single_pass_max_tokens,
        "chunk_target_tokens": settings.llm_chunk_target_tokens,
        "max_output_tokens": settings.llm_max_output_tokens,
        "hardware": detect_hardware().to_dict(),
    }


class ResourceSampler:
    """Samples host CPU utilisation, the Ollama server's resident memory (``/proc``), its GPU
    memory (``/api/ps`` ``size_vram``) and, with NVIDIA, GPU utilisation, in a background thread.
    Peaks are reported; nothing here is estimated."""

    PROC_NAMES = ("ollama", "llama-server", "ollama_llama_server")

    def __init__(self, provider: Any | None = None, interval: float = 0.5) -> None:
        self.provider = provider
        self.interval = interval
        self.samples: list[dict[str, float | None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _cpu_times() -> tuple[int, int] | None:
        try:
            parts = [int(x) for x in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
        except (OSError, ValueError):
            return None
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
        return sum(parts), idle

    @classmethod
    def _server_rss_gb(cls) -> float | None:
        total, found = 0, False
        for d in Path("/proc").glob("[0-9]*"):
            try:
                if (d / "comm").read_text().strip() not in cls.PROC_NAMES:
                    continue
                for line in (d / "status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1]) * 1024
                        found = True
            except (OSError, ValueError):
                continue
        return round(total / GB, 2) if found else None

    def _vram_gb(self) -> float | None:
        if self.provider is None:
            return None
        try:
            return round(sum(m.get("size_vram", 0) for m in self.provider.running()) / GB, 2)
        except Exception:
            return None

    @staticmethod
    def _gpu_util() -> float | None:
        out = _run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], 2.0)
        try:
            return float(out.split()[0]) if out else None
        except (ValueError, IndexError):
            return None

    def _loop(self) -> None:
        prev = self._cpu_times()
        while not self._stop.wait(self.interval):
            cur = self._cpu_times()
            cpu = None
            if prev and cur and cur[0] > prev[0]:
                cpu = round(100 * (1 - (cur[1] - prev[1]) / (cur[0] - prev[0])), 1)
            prev = cur
            self.samples.append(
                {
                    "t": time.monotonic(),
                    "cpu_pct": cpu,
                    "server_rss_gb": self._server_rss_gb(),
                    "vram_gb": self._vram_gb(),
                    "gpu_util_pct": self._gpu_util(),
                    "ram_available_gb": round(_meminfo().get("MemAvailable", 0.0), 2) or None,
                }
            )

    def __enter__(self) -> ResourceSampler:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(5)

    def summary(self) -> dict[str, Any]:
        def col(k: str) -> list[float]:
            return [float(v) for s in self.samples if (v := s.get(k)) is not None]

        def peak(k: str) -> float | None:
            v = col(k)
            return max(v) if v else None

        cpu = col("cpu_pct")
        gpu = col("gpu_util_pct")
        avail = col("ram_available_gb")
        return {
            "samples": len(self.samples),
            "cpu_pct_mean": round(sum(cpu) / len(cpu), 1) if cpu else None,
            "cpu_pct_peak": peak("cpu_pct"),
            "server_rss_peak_gb": peak("server_rss_gb"),
            "vram_peak_gb": peak("vram_gb"),
            "gpu_util_mean_pct": round(sum(gpu) / len(gpu), 1) if gpu else None,
            "ram_available_min_gb": min(avail) if avail else None,
        }


# ----------------------------------------------------------------------------------- CLI


def _installed_tags(settings: Settings) -> tuple[set[str], str | None, str | None]:
    from avd.llm.base import LLMError

    try:
        p = provider_from_settings(settings)
        return {m.get("name", "") for m in p.installed()}, p.version(), None
    except LLMError as exc:
        return set(), None, str(exc)


def doctor(settings: Settings) -> dict[str, Any]:
    hw = detect_hardware()
    installed, version, error = _installed_tags(settings)
    rec = recommend(hw, settings.ollama_num_ctx, installed)
    return {
        "hardware": hw.to_dict(),
        "ollama": {
            "binary": hw.ollama_binary,
            "server": settings.ollama_base_url,
            "server_version": version,
            "server_error": error,
            "installed_models": sorted(installed),
            "configured_model": settings.ollama_model,
            "configured_model_installed": settings.ollama_model in installed,
        },
        "recommendation": rec,
    }


def render_doctor(d: dict[str, Any]) -> str:
    hw, ol, rec = d["hardware"], d["ollama"], d["recommendation"]
    gpu = ", ".join(f"{g['name']} ({g['vram_total_gb']} GB, {g['runtime']})" for g in hw["gpus"]) or "none"
    lines = [
        f"OS        {hw['os']} ({hw['arch']})",
        f"CPU       {hw['cpu_model']} x{hw['cpu_logical']}  flags: {' '.join(hw['cpu_flags']) or '-'}",
        f"RAM       {hw['ram_total_gb']} GB total, {hw['ram_available_gb']} GB available, swap {hw['swap_total_gb']} GB",
        f"GPU       {gpu}",
        f"Disk      {hw['disk_free_gb']} GB free at {hw['disk_path']}",
        f"Ollama    binary: {ol['binary'] or 'not installed'}; server {ol['server']}: "
        + (f"v{ol['server_version']}" if ol["server_version"] else f"unreachable ({ol['server_error']})"),
        f"Models    installed: {', '.join(ol['installed_models']) or 'none'}; configured: {ol['configured_model']}",
        "",
        f"Memory budget {rec['memory_budget_gb']} GB ({rec['judged_against']}), num_ctx {rec['num_ctx']}",
        "",
        f"{'tag':<28}{'quant':<8}{'params':>7}{'active':>7}{'mem GB':>8}{'dl GB':>7}  fits",
    ]
    for r in rec["rows"]:
        fit = "yes" if r["fits_memory"] and r["fits_disk"] else ("no (disk)" if r["fits_memory"] else "no")
        lines.append(
            f"{r['tag']:<28}{r['quantization']:<8}{r['params_b']:>6}B{r['active_params_b']:>6}B"
            f"{r['total_gb']:>8}{r['download_gb']:>7}  {fit}{' (installed)' if r['installed'] else ''}"
        )
    lines += [
        "",
        f"Recommended start: {rec['primary'] or 'none fits'}; benchmark: {', '.join(rec['benchmark_candidates']) or '-'}",
        f"Note: {rec['caveat']}",
        *(f"Note: {n}" for n in hw["notes"]),
    ]
    if rec["primary"] and not ol["installed_models"]:
        est = next(r for r in rec["rows"] if r["tag"] == rec["primary"])
        lines += [
            "",
            "Next steps (nothing is downloaded without approval):",
            "  curl -fsSL https://ollama.com/install.sh | sh     # or the package for your OS",
            f"  avd local pull {rec['primary']} --approve-gb {int(est['download_gb']) + 1}",
        ]
    return "\n".join(lines)


def pull(settings: Settings, model: str, approve_gb: float, estimate_gb: float | None) -> dict[str, Any]:
    """Pull one model through the local server, only if the approved size covers the estimate."""
    from avd.llm.base import LLMConfigurationError
    from avd.llm.ollama import is_cloud_model

    if is_cloud_model(model):
        raise LLMConfigurationError(f"{model} is a cloud model; refused")
    if estimate_gb is None:
        raise LLMConfigurationError(
            f"{model} is not in the candidate table, so its size is unknown; check it on the Ollama "
            "library page and pass --size-gb with the listed download size"
        )
    if approve_gb < estimate_gb:
        raise LLMConfigurationError(
            f"{model} is ~{estimate_gb} GB; --approve-gb {approve_gb} does not cover it (nothing downloaded)"
        )
    hw = detect_hardware()
    if estimate_gb + 1 > hw.disk_free_gb:
        raise LLMConfigurationError(f"{model} needs ~{estimate_gb} GB; only {hw.disk_free_gb} GB free")
    p = provider_from_settings(settings)
    return p.pull(model)


def add_parser(sub: Any) -> None:
    p = sub.add_parser("local", help="local inference (Ollama): hardware, models, downloads")
    lsub = p.add_subparsers(dest="local_cmd", required=True)
    d = lsub.add_parser("doctor", help="detect hardware and recommend local models (never downloads)")
    d.add_argument("--json", action="store_true")
    d.set_defaults(fn=_doctor_main)
    pl = lsub.add_parser("pull", help="download one model after explicit size approval")
    pl.add_argument("model")
    pl.add_argument("--approve-gb", type=float, required=True, help="largest download you approve, in GB")
    pl.add_argument("--size-gb", type=float, default=None, help="download size for a model not in the table")
    pl.set_defaults(fn=_pull_main)


def _doctor_main(args: Any) -> None:
    from avd.config import get_settings

    d = doctor(get_settings())
    print(json.dumps(d, indent=2) if args.json else render_doctor(d))


def _pull_main(args: Any) -> None:
    from avd.config import get_settings
    from avd.llm.base import LLMError

    settings = get_settings()
    est = args.size_gb
    if est is None:
        for c in CANDIDATES:
            for quant in ("q4_K_M", "q8_0"):
                tag = c.tag if quant == "q4_K_M" else f"{c.tag}-{quant.lower()}"
                if tag == args.model:
                    est = estimate_memory_gb(c, quant)["download_gb"]
    try:
        r = pull(settings, args.model, args.approve_gb, est)
    except LLMError as exc:
        raise SystemExit(f"refused: {exc}") from exc
    print(json.dumps(r, indent=2))
