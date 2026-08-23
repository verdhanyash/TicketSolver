# rules.md — Coding-Agent Operating Rules

These rules apply to **every** AI coding session on TicketSolver. Read them at the start of
each session, alongside `CLAUDE.md` (stack decisions) and the SRS
(`Docs/SRS_Enterprise_Support_Agent_Platform.md`).

## Before starting work
1. **State the plan.** Briefly list which files you will touch before making changes.
2. **Check `CLAUDE.md` before any dependency.** Never install a new library/framework if an
   already-decided equivalent exists. If nothing covers the need, **ask first** — don't add
   a new dependency, service, or a schema change that isn't already in the SRS without
   explicit sign-off.
3. **Ask when ambiguous.** If a requirement is unclear, stop and ask rather than guessing and
   building the wrong thing.

## While working
4. **Stay scoped.** Change only what was asked. No unrelated refactors, renames, or cleanups
   unless you flag them and get agreement first.
5. **Use targeted edits.** Don't regenerate or reprint whole files for small changes — make
   minimal, surgical edits.
6. **Follow the conventions in `CLAUDE.md`** (config via pydantic-settings, secrets in `.env`,
   guardrails in code not prompts, PII redaction in logs, self-descriptive UI).

## Before marking a task complete
7. **Run the existing tests.** Run the **relevant** tests only (the ones covering what you
   changed) — not the full suite, unless explicitly asked. Fix failures before calling it done.
8. **Report honestly.** If tests fail, say so with the output. If a step was skipped, say that.
   Don't claim something works unless you verified it.

## At the end of a session
9. **Summarize concisely.** A few lines on what changed — not a full re-explanation of the
   codebase.

## Guardrails on risky actions
10. **Confirm before destructive or hard-to-reverse actions**: dropping/altering DB schema,
    deleting data, force-pushing, mass edits, deploying, or anything affecting shared/live
    systems. Prefer non-destructive alternatives.
