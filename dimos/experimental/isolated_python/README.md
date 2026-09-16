# Experimental Isolated Python Modules

`IsolatedPythonModule` runs a Python module in a separate dependency environment
while preserving dimOS streams, RPCs, skills, module references, and lifecycle
management. Its API is experimental and may change without compatibility aliases.

## Project layout

Place the host contract beside a `python/` project that contains the concrete
runtime:

```text
my_module/
├── contract.py
└── python/
    ├── pyproject.toml
    └── my_runtime/
        └── runtime.py
```

Define the host-visible contract:

```python skip
from dimos.core.core import rpc
from dimos.experimental.isolated_python.module import (
    IsolatedPythonModule,
    IsolatedPythonModuleConfig,
)


class MultiplierConfig(IsolatedPythonModuleConfig):
    initial_multiplier: int = 2


class Multiplier(IsolatedPythonModule):
    implementation = "my_runtime.runtime:MultiplierRuntime"
    config: MultiplierConfig

    @rpc
    def get_multiplier(self) -> int:
        raise NotImplementedError
```

The sibling runtime imports and implements that contract:

```python skip
from typing import Any

from dimos.core.core import rpc
from my_module.contract import Multiplier


class MultiplierRuntime(Multiplier):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._multiplier = self.config.initial_multiplier

    @rpc
    def get_multiplier(self) -> int:
        return self._multiplier
```

Every contract RPC and skill must be overridden with a compatible signature and
the same `@rpc` or `@skill` classification. Startup fails if the runtime inherits
a contract stub or changes its signature or classification.

## Runtime behavior

During `build()`, DimOS runs `uv sync --frozen` in the sibling project, then
installs matching host DimOS code with `uv pip install --no-deps`. Launch uses
`uv run --no-sync`; there is no host-dependency overlay. Every runtime must declare
and lock the dependencies it uses, including DimOS message and transport imports.
If `pixi.toml` exists, Pixi supplies the toolchain and `uv`.

Source checkouts install the current checkout editable. Installed hosts install
exactly their own DimOS version. The sibling project's `.python-version` and
`requires-python` select its interpreter independently of the host.

The bootstrap calls `run_runtime(stopping)` on its main thread after establishing
RPC service. The default implementation waits for shutdown. Thread-affine engines
can override it to initialize and drive their simulator there while RPC handlers
queue operations. `stop()` must signal that loop to exit.

The host contract retains the public module name and forwards contract RPCs to a
unique internal endpoint. Ordinary dimOS serialization and transport handle RPC
values, exceptions, timeouts, async methods, skills, streams, and module
references. Restarting the contract starts a fresh interpreter and reloads the
runtime package.

## Example

The source tree includes a complete example with a locked external project:

```bash
uv run python -m dimos.experimental.isolated_python.example.run
```

The example demonstrates streams, RPCs, skills, an injected module reference,
restart behavior, and automatic shutdown.
