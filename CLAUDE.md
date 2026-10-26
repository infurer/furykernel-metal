<!--
SPDX-FileCopyrightText: 2026 Mika Tammi
SPDX-License-Identifier: BSD-3-Clause
-->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Project overview

furykernel-metal is FuryKernel's Apple Metal target: a library that
implements the core's target interface for Apple GPUs. It holds the target
descriptions (`AppleGPU9`, Apple's GPU family 9 of the A17 Pro, A18, M3
and M4, first; `AppleGPU7` for M1 and `AppleGPU10` for M5 later) and the
lowering of checked kernels to Metal Shading Language, with the launch
tables runtimes read. Runtimes write their launchers themselves, from
those tables and the core's `FuryKernel.Emit.Rust`. The GPU lowering of
FuryKernel programs into launch schedules is planned. It depends only on
the core library, furykernel.

It never runs anything on a GPU: it emits MSL source, which the consuming
runtime compiles with Metal at run time. So it builds and is tested on any
system, and golden tests pin its output.

Like the core, it knows no models: model kernels live with the runtime that
runs them.

## Conventions

- Every file has SPDX copyright and license comments
  (`2026 Mika Tammi`, `BSD-3-Clause`). Files that can't hold comments are
  annotated in `REUSE.toml`.
- Nothing host-specific in the repository: no hostnames, absolute paths of
  anyone's machine, or names of private repositories and services. master
  is mirrored to a public GitHub repository in full.

## Git workflow

- One commit per reasonably sized feature. Docs (README, this file) change
  in the same commit as the feature they describe.
- Subjects in nixpkgs style, `<attr or area>: <summary>`. Short bodies: why,
  and facts that aren't obvious from the diff.
- Commit with `git commit -s`, and with an `Assisted-by:` trailer naming
  the AI assistant that helped, such as `Assisted-by: Claude Opus 5.5`,
  directly above the Signed-off-by. No other trailers.
- Corrections to earlier commits go in as `git commit --fixup=<commit>` (or
  `--squash`) and are autosquashed before the merge request.
- v0.2 work happens on `initialv02` and reaches master through merge
  requests on the GitLab repository, which CI mirrors to GitHub. Never push
  to GitHub by hand. Mika signs the commits with GPG before the merge.

## Common mistakes to avoid

- Committing `HANDOFF.md` or other session notes.
- Writing SPDX tag names followed by a colon in prose: `reuse lint` parses
  them anywhere in a file.
