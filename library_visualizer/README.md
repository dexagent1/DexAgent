# Library visualizer

A single static page that shows the tool library live: totals, the skills and verifiers by category, and each tool's inputs, outputs and source.

It is served at **https://dexagent1.github.io/library/** and reads [`tool_library/index.json`](../tool_library/index.json) from this repository on every load, so it reflects the library as committed. The `index.json` next to the page is a snapshot used only when the live file cannot be reached.

To deploy after changing the page: copy `index.html` and a fresh `tool_library/index.json` into the `library/` folder of the website repository.
