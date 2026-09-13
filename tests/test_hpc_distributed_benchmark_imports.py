from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarking" / "scripts" / "01_python"


def test_distributed_outer_benchmarks_do_not_import_sibling_benchmark_modules():
    """Dask schedulers/workers may not have the benchmark script dir on sys.path."""
    for name in (
        "benchmark_bayesian_rsf_outer.py",
        "benchmark_bayesian_choice_outer.py",
    ):
        text = (BENCH / name).read_text(encoding="utf-8")
        assert not re.search(r"^from\s+benchmark_[A-Za-z0-9_]+\s+import\s+", text, re.M), name
        assert not re.search(r"^import\s+benchmark_[A-Za-z0-9_]+\b", text, re.M), name
