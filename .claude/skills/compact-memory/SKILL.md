---
name: compact-memory
description: >
  Create structured summaries of research conversations and store them in
  `/media/ak/d1c5342e-77c5-411d-a9ac-03660a90ce7d/home/ak/Gitrepos/2026_research_tracks/memories
/` for multi-agent memory sharing and conversation continuity.
  Trigger when the user asks to "summarise this conversation", "save a review",
  "log this session", "create a memory file", or references the track system
  (Tracks A–E). Also trigger when the user says "save to /media/ak/d1c5342e-77c5-411d-a9ac-03660a90ce7d/home/ak/Gitrepos/2026_research_tracks/memories
" or asks to
  record progress on a research track. Do NOT trigger for general note-taking
  or journaling unrelated to the research track system.
---

# Conversation Memory System

Produce a structured Markdown summary of a research conversation, assign it to
a track (A–E), and save it to `/media/ak/d1c5342e-77c5-411d-a9ac-03660a90ce7d/home/ak/Gitrepos/2026_research_tracks/memories
/` with a deterministic filename. The
file must be self-contained enough for a cold-start agent to resume work.

## Track Identities

| Track | Domain | Core Objects |
|-------|--------|--------------|
| **A** | Microstructure-aware ML | Stability, calibration, market-impact guarantees |
| **B** | Causal inference in microstructure | Identification, doubly robust estimation, sensitivity analysis |
| **C** | Decision-focused learning / offline RL | Regret bounds, safe policy improvement, DRO |
| **D** | Multiscale memory & multifractality | Scaling laws as first-class objects |
| **D1** | — Method papers | |
| **D2** | — Decision papers | |
| **D3** | — Causal papers | |
| **E** | Kernels, causality & graph signal processing | HSIC/KCI, RKHS embeddings, spectral graph methods |

## File Naming Convention

```
{YYYYMMDD}_TRACK_{X}_version_{N}.md
```

- `YYYYMMDD` — date in ISO-8601 compact format
- `X` — track identifier: `A`, `B`, `C`, `D`, `D1`, `D2`, `D3`, or `E`
- `N` — version number, monotonically increasing per track per day (1, 2, 3 …)

**Example**: `20260208_TRACK_D1_version_2.md`

## Workflow

1. **Identify track** — Determine the primary track from conversation content.
   If the conversation spans multiple tracks, pick the dominant one and note
   cross-track connections inside the file.
2. **Determine version** — List existing files for the same date and track:
   ```bash
   ls /media/ak/d1c5342e-77c5-411d-a9ac-03660a90ce7d/home/ak/Gitrepos/2026_research_tracks/memories
/$(date +%Y%m%d)_TRACK_{X}_version_*.md 2>/dev/null | sort -V | tail -1
   ```
   Increment the highest version found, or start at 1.
3. **Draft summary** — Fill in the template below. Every section is mandatory;
   write "None" if a section is genuinely empty.
4. **Validate** — Confirm:
   - Filename matches the convention exactly.
   - All template sections are present.
   - `Agent Memory Tags` include at least the track letter.
   - `Cross-Track Connections` references at least one other track, or
     explicitly states "None identified".
5. **Save** — Write to `/media/ak/d1c5342e-77c5-411d-a9ac-03660a90ce7d/home/ak/Gitrepos/2026_research_tracks/memories
/{filename}`.
6. **Confirm** — Print the saved path and a one-line summary to the user.

## Summary Template

```markdown
# Conversation Summary: Track {X}

**Date**: {YYYY-MM-DD}
**Version**: {N}
**Previous Version**: {filename or "N/A"}

## Key Topics Discussed

- {Topic 1}: {one-sentence description}
- {Topic 2}: {one-sentence description}

## Technical Contributions

### Concepts Developed

- **{Concept}**: {Brief explanation — enough for cold-start}

### Methods / Approaches

- **{Method name}**: {Key insight or result}

## Code / Implementation Notes

> Include key snippets or pseudocode. Use fenced code blocks with language tags.

```python
# example
```

## Open Questions

- {Question that remains unresolved}

## Next Steps

- [ ] {Actionable item with enough context to execute}

## References / Resources Mentioned

- {Author (Year)}: {Why it is relevant}

## Cross-Track Connections

- **Track {Y}**: {How this conversation connects}

## Agent Memory Tags

#track-{x} #tag1 #tag2
```

## Critical Rules

- **Never overwrite** an existing version — always increment.
- **Link previous versions** explicitly so the chain is traversable.
- **Write for cold-start** — a fresh agent with no conversation history must be
  able to understand the summary and continue the work.
- **Be precise about state** — distinguish between "explored and rejected",
  "explored and promising", and "unexplored".
- **Tag consistently** — always include `#track-{x}` as the first tag.
- **Keep code snippets minimal** — include only the essential logic or
  pseudocode, not full files.

## Quick Reference

```bash
# List all summaries for a track
ls /media/ak/d1c5342e-77c5-411d-a9ac-03660a90ce7d/home/ak/Gitrepos/2026_research_tracks/memories
/*_TRACK_B_*.md

# Latest version for a track on a given date
ls r/media/ak/d1c5342e-77c5-411d-a9ac-03660a90ce7d/home/ak/Gitrepos/2026_research_tracks/memories
/20260208_TRACK_C_*.md | sort -V | tail -1

# All tracks discussed on a date
ls /media/ak/d1c5342e-77c5-411d-a9ac-03660a90ce7d/home/ak/Gitrepos/2026_research_tracks/memories
/20260208_*.md | sed 's/.*TRACK_\([^_]*\).*/\1/' | sort -u

# Full-text search across all summaries
grep -rl "multifractal" /media/ak/d1c5342e-77c5-411d-a9ac-03660a90ce7d/home/ak/Gitrepos/2026_research_tracks/memories
/

# Count summaries per track
for t in A B C D D1 D2 D3 E; do
  echo "Track $t: $(ls /media/ak/d1c5342e-77c5-411d-a9ac-03660a90ce7d/home/ak/Gitrepos/2026_research_tracks/memories
/*_TRACK_${t}_*.md 2>/dev/null | wc -l)"
done
```

## Example

**File**: `/media/ak/d1c5342e-77c5-411d-a9ac-03660a90ce7d/home/ak/Gitrepos/2026_research_tracks/memories
20260208_TRACK_D1_version_1.md`

```markdown
# Conversation Summary: Track D1

**Date**: 2026-02-08
**Version**: 1
**Previous Version**: N/A

## Key Topics Discussed

- Multiscale memory architecture: designing hierarchical attention that respects
  power-law decay in financial time series.
- Fractal pooling: a pooling operation that preserves self-similar structure
  across temporal scales.

## Technical Contributions

### Concepts Developed

- **Multiscale Attention**: Hierarchical attention mechanism where the receptive
  field at scale s grows as s^H, with H estimated from data via DFA.
- **Fractal Pooling**: Pooling that aggregates within self-similar blocks rather
  than uniform windows, preserving long-range dependence.

### Methods / Approaches

- **ScaleNet**: Neural architecture with explicit multiscale structure; each
  layer operates at a different Hurst-informed time scale.
- **Hurst-aware Loss**: Auxiliary loss penalising deviation from target scaling
  exponent in the learned representation's fluctuation function.

## Code / Implementation Notes

```python
class MultiscaleMemory(nn.Module):
    def __init__(self, hidden: int, scales: list[int] = [1, 5, 25, 125]):
        super().__init__()
        self.memories = nn.ModuleList(
            [nn.LSTM(hidden, hidden) for _ in scales]
        )
```

## Open Questions

- How to enforce theoretical guarantees on scaling exponent preservation during
  backprop?
- Optimal scale hierarchy: geometric (1, 5, 25, …) vs data-adaptive?

## Next Steps

- [ ] Implement fractal pooling operator and unit tests.
- [ ] Benchmark on CME E-mini futures tick data (scales 1s → 1h).
- [ ] Derive sufficient conditions for scaling-law preservation under SGD.

## References / Resources Mentioned

- Mandelbrot (1997): Foundational framework for fractal scaling in finance.
- Cont (2001): Empirical stylised facts — target properties for the model.

## Cross-Track Connections

- **Track A**: Microstructure features (bid-ask, queue imbalance) as input to
  the multiscale encoder.
- **Track C**: Long-memory state representation feeds into offline RL policy.

## Agent Memory Tags

#track-d1 #multiscale #fractals #scaling-laws #deep-learning #hurst
```
