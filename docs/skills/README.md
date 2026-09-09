# Skill sources

Version-controlled sources for Claude Code skills authored in this repo.

The live skills directory (`.claude/skills/`) is intentionally untracked — it is internal developer
tooling, excluded from nf-core PRs — and is mounted read-only inside the agent sandbox to prevent
skill injection. Skills are therefore authored here, under version control, and installed by copying.

## Contents

| Path                                            | Installs to                                          |
| ----------------------------------------------- | ---------------------------------------------------- |
| `python-packaging.md`                           | `.claude/skills/python-packaging.md`                 |
| `evals/python-packaging.json`                   | `.claude/evals/python-packaging.json`                |
| `evals/trigger-evals/python-packaging.json`     | `.claude/evals/trigger-evals/python-packaging.json`  |

## Install

Run from the repository root:

```bash
cp docs/skills/python-packaging.md .claude/skills/
cp docs/skills/evals/python-packaging.json .claude/evals/
cp docs/skills/evals/trigger-evals/python-packaging.json .claude/evals/trigger-evals/
```

Then confirm the skill is picked up with `/python-packaging` in a new Claude Code session.

## Conventions

Each skill carries `name`, `description`, `category` (`capability_uplift` or `encoded_preference`),
`version` and `eval_path` in its frontmatter, per `.claude/rules/general.md` RULE #5.

Every skill ships two eval files: graded assertions in `evals/<skill>.json` and trigger precision/recall
queries in `evals/trigger-evals/<skill>.json`. Re-run both before and after modifying a skill and
compare pass rates; a drop is a regression.
