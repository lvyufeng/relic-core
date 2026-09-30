"""relic-core: shared torch operator library for older accelerators.

Kernels for the hardware the two runtimes target -- CUDA ``sm_75`` (RTX 2080
Ti) and CPU host ops used by the offload path.  This package holds no model code
and no engine: it is the operator layer that ``relicllm`` and ``pocketllm`` both
depend on.

Layout::

    relic_core/csrc/      C++/CUDA sources, built into the extension modules
    relic_core/kernels/   the Python op layer that loads and dispatches them
"""

__version__ = "0.1.0.dev0"
