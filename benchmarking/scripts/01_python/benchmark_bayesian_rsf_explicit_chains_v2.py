"""DataTree-compatible entry point for explicit BlackJAX chain scheduling.

This wrapper reuses the original benchmark implementation but replaces its
legacy InferenceData-only recombination helper with the compatibility-tested
implementation from ``hsa.rsf._inference_tree``.
"""

from __future__ import annotations

import benchmark_bayesian_rsf_explicit_chains as _impl
from hsa.rsf._inference_tree import combine_single_chain_inference


_impl._combine_chain_idatas = combine_single_chain_inference

# Re-export for tests and interactive diagnostics.
_combine_chain_idatas = combine_single_chain_inference


if __name__ == "__main__":
    _impl.main()
