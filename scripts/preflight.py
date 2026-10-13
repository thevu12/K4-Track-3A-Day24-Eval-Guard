"""Check live-workflow prerequisites without provider API calls or downloads."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import importlib
import importlib.metadata
import importlib.util
import io
from pathlib import Path
import sys
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Importing these packages detects binary/version incompatibilities that a
# dependency resolver alone cannot catch. No model/engine is instantiated.
DEPENDENCIES = (
    ("dotenv", "python-dotenv"), ("numpy", "numpy"), ("pandas", "pandas"),
    ("pyarrow", "pyarrow"), ("datasets", "datasets"), ("pypdf", "pypdf"),
    ("underthesea", "underthesea"), ("rank_bm25", "rank-bm25"),
    ("qdrant_client", "qdrant-client"), ("sentence_transformers", "sentence-transformers"),
    ("flashrank", "flashrank"), ("ragas", "ragas"),
    ("langchain_community", "langchain-community"), ("langchain_openai", "langchain-openai"),
    ("openai", "openai"), ("scipy", "scipy"), ("spacy", "spacy"),
    ("presidio_analyzer", "presidio-analyzer"), ("presidio_anonymizer", "presidio-anonymizer"),
    ("nemoguardrails", "nemoguardrails"), ("jsonschema", "jsonschema"), ("pytest", "pytest"),
)


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def check_import(module: str, package: str) -> Check:
    """Suppress library chatter; report no exception body that could expose secrets."""
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            importlib.import_module(module)
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "version unavailable"
    except Exception as error:
        return Check(package, "FAIL", f"import failed ({type(error).__name__}); check runtime constraints")
    return Check(package, "PASS", f"import OK ({version})")


def check_spacy_model() -> Check:
    """Check installation only; loading/downloading the 560MB model is separate."""
    try:
        installed = importlib.util.find_spec("en_core_web_lg") is not None
    except (ImportError, ValueError, AttributeError):
        installed = False
    return Check("en_core_web_lg", "PASS" if installed else "FAIL",
                 "installed (not loaded)" if installed else
                 "missing; install with this interpreter: -m spacy download en_core_web_lg")


def check_qdrant(host: str, port: int, *, timeout: float = 5) -> Check:
    """Read only /healthz; never create, modify or delete collections."""
    try:
        if not isinstance(host, str) or not host.strip() or host != host.strip():
            raise ValueError("Invalid Qdrant host")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Invalid Qdrant port")
        # Config uses a bare host. Accept bracketed IPv6 without mistaking a
        # credential-bearing URL, path or query for a hostname.
        parsed = urlsplit("//" + host)
        if (not parsed.hostname or parsed.username is not None or parsed.password is not None
                or parsed.path or parsed.query or parsed.fragment or parsed.port is not None):
            raise ValueError("QDRANT_HOST must be a bare hostname")
        request = Request(f"http://{host}:{port}/healthz", method="GET")
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200 or response.read(1024).strip() != b"healthz check passed":
                raise ValueError("Unexpected Qdrant health response")
    except Exception as error:
        return Check("Qdrant", "FAIL", f"read-only /healthz check failed ({type(error).__name__})")
    return Check("Qdrant", "PASS", "read-only /healthz returned healthy; collections unchanged")


def run_checks(*, live: bool = False) -> list[Check]:
    """Local checks by default; --live permits the configured Qdrant HTTP probe."""
    correct_python = sys.version_info[:2] == (3, 11)
    results = [Check("Python", "PASS" if correct_python else "FAIL",
                     f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}; requires 3.11")]
    results.extend(check_import(module, package) for module, package in DEPENDENCIES)
    results.append(check_spacy_model())
    try:
        # Read presence only. Never print or send the value to any service.
        configuration = importlib.import_module("config")
        api_key = configuration.LLM_API_KEY
        key_present = isinstance(api_key, str) and bool(api_key.strip())
    except Exception as error:
        results.append(Check("configuration", "FAIL", f"cannot load config ({type(error).__name__})"))
        return results
    results.append(Check(f"{configuration.LLM_PROVIDER.upper()} API key", "PASS" if key_present else "FAIL",
                         "present; authorization/quota not tested" if key_present else
                         "missing; set it locally in .env (never commit or paste the value)"))
    results.append(check_qdrant(configuration.QDRANT_HOST, configuration.QDRANT_PORT) if live else
                   Check("Qdrant", "SKIP", "pass --live to permit the read-only configured-host HTTP check"))
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="Permit read-only Qdrant networking; never invokes a paid provider API")
    args = parser.parse_args(argv)
    results = run_checks(live=args.live)
    for result in results:
        print(f"{result.status} {result.name}: {result.detail}")
    failures = sum(result.status == "FAIL" for result in results)
    print(f"{failures} failed prerequisite checks. No provider API calls or model downloads were performed.")
    print("Model caches, API authorization/quota and live experiment quality are not certified by preflight.")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
