## NaPO LLaVA Reference Snapshot

This directory is a trimmed reference snapshot of the public NaPO repository:

- Upstream repository: `https://github.com/zhangzef/NaPO`
- Local source archive in workspace: `/path/to/sage_repro_bundle/NaPO-master.zip`
- Local extracted copy used for comparison: `/path/to/sage_repro_bundle/NaPO-master`

Purpose in this bundle:

- provide a reference implementation for the LLaVA-based NaPO baseline;
- preserve the exact historical training entrypoints used in local comparison runs;
- keep third-party comparison code separate from the main SAGE implementation.

Important licensing note:

- As of 2026-05-05, the public upstream repository did not expose a clear
  repository-level `LICENSE` file at the root.
- Some files inside this snapshot contain their own upstream copyright and
  Apache-2.0 notices inherited from LLaVA or other projects.
- Treat this directory as a third-party reference snapshot, not as code that is
  relicensed under the main bundle license.

Recommended use for paper submission:

- keep this directory separate from your core contribution code;
- cite the NaPO paper and repository in the manuscript;
- if distributing this snapshot, preserve file headers and this README;
- if you want a lower-risk archive, exclude this directory and keep only the
  wrapper script plus instructions for obtaining the original upstream code.
