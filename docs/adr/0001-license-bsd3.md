# 1. License: BSD-3-Clause

*Status: accepted · 2026-10-22*

**Context.** fitwise builds on lmfit, SciPy and NumPy, which are all BSD-licensed. Its users
include companies that embed fits in measurement software.

**Decision.** fitwise is licensed under BSD-3-Clause.

**Consequences.** The license matches the rest of the stack, so downstream users have no
additional license review. Code can move between fitwise and its dependencies in both directions.
