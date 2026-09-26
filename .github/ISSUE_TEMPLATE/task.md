---
name: Pipeline task
about: A small, checkable job for the coding pipeline
title: ""
labels: ""
assignees: ""
---

## Goal

Describe the outcome.

## Acceptance criteria

- [ ] Describe a checkable outcome.

<!-- Aim for 2–4 observable results: a command and expected output, or a test that must pass. Without at least one, the pipeline sends the spec back. -->

## Existing behavior affected

Name any behavior or existing test this changes, and what must still hold. Write "None" if there is none.

## Non-goals

List only nearby work that could be mistaken for part of this task.

## Areas touched

List the files or modules this task may change.

<!-- One exact file per line, in backticks, like: - `controller/jobs.py`. The task runs beside others in its repo only when their files don't overlap. A glob (`docs/**`) could hide a lockfile, so it makes the task run alone, as does leaving this as is. -->

## Foreman (planner) vendor

unknown

Replace with exactly `claude` or `codex` when a foreman writes this spec; leave `unknown` for an issue filed without a foreman. The controller uses it to pick who triages a failed job.

## Risk level

- [ ] Low risk
- [ ] Needs review

## Existing tests

May the worker modify or delete existing tests? **Yes / No**. Explain any limits.

<!-- To allow only some, answer Yes and list them in backticks, one per line, like: - `tests/test_greet.py`. Other existing tests stay protected. Leaving this unanswered sends the spec back. -->

## Test expectations

Name checks beyond the acceptance criteria, or write "Repo check only".

## Dependencies

If applicable, add `Depends on owner/repo#123`.

## Context

What the planning conversation settled that the sections above don't say: decisions and why, approaches ruled out, the owner's preferences. Delete this section if there's nothing to add.

<!-- A triage of a failed job is a headless run that never saw the conversation; this is all it knows of it. -->
