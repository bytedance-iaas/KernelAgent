# Step: Best-of-3 Per-Round Resampling

> **Paths:** `${CLAUDE_SKILL_DIR}` is the ka-kernel-opt skill directory; the
> tool scripts live at `${CLAUDE_SKILL_DIR}/tools/`.

## Purpose

Replaces steps 04-05 for one round when `BESTOF3=true`: instead of one
rewrite attempt per round, generate 3 independent candidate rewrites from
the *same* diagnosis, verify and benchmark all 3, and advance only if the
best of them beats the running best — never generate multiple diagnoses,
only multiple implementations of one.

**Gated by `BESTOF3=true` (default `false` — opt-in, not a safe default).**
Read the evidence before recommending it to a user; do not enable it as a
blanket "more thorough search" option. See "When this helps, honestly" below
before turning it on for a given problem.

## Step 1: Generate 3 independent candidates

Run `steps/04_rewrite.md` Step 1-3 **three times independently** against the
*same* `diagnosis_round_$ROUND.json` — do not resample the diagnosis, only
the rewrite. Each attempt should be a genuinely independent LLM sample (not
three deterministic variations you hand-author) — the value here comes from
sampling variance in which implementation gets tried, including the
possibility that one attempt is architecturally different from the other
two, not from mechanically sweeping three preset parameter values.

Save to `$RUN_DIR/kernel_candidate_a.py`, `_b.py`, `_c.py`.

## Step 2: Verify + benchmark all 3

For each candidate, run `steps/05_verify_accept.md` Step 1-2 independently
(up to 3 correctness-refinement attempts each, benchmark on success). A
candidate that fails all 3 refinement attempts is simply dropped from this
round's comparison — it is not required that all 3 candidates survive, only
that the round record honestly shows how many did.

Do **not** re-profile for SOL per candidate — that's tripled cost for a
number this mode doesn't act on. `05_verify_accept.md` Step 3's two-track
best-SOL tracking is also not part of this mode's gate (Step 3 below is
runtime-only); register all 3 candidates with `--sol-pct` omitted and rely
on the once-per-round parent profile from Step 02 for any SOL/efficiency
reporting.

Register every candidate that passed verification in `program_db.json` via
`program_db.py add`, all with `--parent-id` set to the round's starting
kernel (`BEST_KERNEL`, i.e. whatever `program_db.py best` returns *before*
this round started — not the previous candidate in this round; the three
candidates are siblings, not a chain) and the same `--round $ROUND` for all
three. Candidates that fail all 3 refinements are recorded in
`attempts.jsonl` only, same as the vanilla loop.

## Step 3: Pick the round's winner and apply the strict gate

Winner = fastest verified candidate this round. This mode uses a **strict**
accept rule, not `05_verify_accept.md`'s 50%-divergence-tolerant one:

- If the winner's time is **strictly less than** `BEST_TIME_MS`: accept it,
  update `BEST_TIME_MS`/`BEST_KERNEL` to the winner. This is what the next
  round (whether `BESTOF3` or not) diagnoses from.
- Otherwise (winner is flat or a regression, or all 3 candidates failed
  verification): **do not advance.** `BEST_KERNEL` stays what it was. The
  *next* round diagnoses from and forks fresh candidates off the same
  `BEST_KERNEL` again — it does not chain from this round's non-improving
  output.

This strictness is load-bearing, not a stylistic choice: a controlled,
replicated test found that a version of this mode which chained forward
from a round's winner regardless of whether it actually improved on the
running best — mirroring vanilla's own more lenient divergence-tolerant
rule — could compound a bad round for 2-3 consecutive rounds before
recovering, on two different real problems. Giving best-of-3 vanilla's
lenient rule is *not* neutral: vanilla only risks one candidate per round on
that leniency, best-of-3 risks three, so an unproductive round costs
proportionally more compute before the strict-improvement bar forces a
retry from a known-good point.

## Output

**Not** the identical reporting contract as `05_verify_accept.md` — that
step's SOL/NCU metrics come from re-profiling the new candidate to compute
`NEW_SOL`, and Step 2 above deliberately skips that per candidate (tripled
profiling cost for a number this mode doesn't gate on). Report instead:
- All 3 candidates' times side by side, and which one won.
- Whether the round advanced or was rejected by the gate.
- SOL/NCU metrics from the round's *one* parent profile (Step 02, run once
  on `BEST_KERNEL` before diagnosing) — this is the same profile the
  diagnosis was grounded in, not a re-profile of any candidate. If the
  winning candidate's own SOL is specifically needed (e.g. for a
  `perf_test.py` gate), profile it explicitly as an extra, deliberate step
  and say so — don't imply it was measured for free.
- A compact code diff (± lines only, max 20) of parent → winning candidate.
- One-line diagnosis-family note: is this round retrying the *same*
  bottleneck category as the previous round, or a fresh diagnosis? (Relevant
  to the guidance below.)

## Cost, honestly

A best-of-3 round costs roughly **1.7-2.2x** a vanilla round's true compute
(diagnosis is shared, not tripled; only the rewrite-verify-benchmark cycle
triples) — not the naive 3x a round-count comparison would suggest. Still a
real, mandatory extra cost every round it runs, regardless of whether that
round needed it.

## When this helps, honestly — read before enabling

Tested end-to-end (vanilla vs. this mode, replicated where feasible) on 4
real problems. Results were genuinely problem-dependent, not a reliable
win: real, replicated wins on 2 of 4 problems (`060_chunk_gated_delta_rule_
linear_attention`, `011_fp8_moe_gate_routing`); no reliable separation from
vanilla on the other 2 (`fp8_group_gemm`, `008_moe_sparse_routing_and_
dispatch`) once properly replicated — see `insights/BEST-OF-N RESAMPLING
(IDEA F) - EXPLORATION.md` for the full data.

The mechanism only resamples the *rewrite* step; the diagnosis each round is
still one LLM call. That predicts, and the transcripts confirm, when this
helps and when it doesn't:

- **Helps** when a round's diagnosis is likely correct but the
  *implementation* of the fix is uncertain — multiple structurally different
  ways to build it, real risk the "obvious" choice fails to compile or is
  fragile, or a genuinely non-monotonic effect (e.g. partial vs. full
  fusion) that a single sample can't reveal. Problems dominated by one or
  two big architectural opportunities (a single fusion-heavy kernel, a
  gating/routing pipeline with numerically fragile edge cases like top-k
  ties) tend to look like this.
- **Doesn't help, and can actively waste rounds** when the problem needs a
  long *sequence* of genuinely different diagnoses (dispatch restructuring,
  then a separate allocation-overhead fix, then a separate dtype/cast
  reduction, ...) — all 3 candidates in a round share one diagnosis, so if
  that diagnosis's category is already exhausted, tripling attempts within
  it doesn't find the next category; it just spends 3x confirming a dead
  end while vanilla would have moved on. Problems that are a long pipeline
  of many small, structurally different kernel launches (routing + sort +
  scatter + GEMM + cast, etc.) tend to look like this.

**A cheap, checkable-in-advance proxy**: look at the naive reference
kernel's operation/launch-count diversity before enabling this. A handful of
launches around one dominant computational block → reasonable bet. A long
pipeline of many small, structurally different ops → bad bet, prefer more
vanilla rounds instead (more diagnoses, not more candidates per diagnosis).

This is drawn from 4 problems — a real, mechanistically-grounded hypothesis,
not a proven rule. Treat `BESTOF3=true` as an experimental spend a user
opts into knowingly, not a default "more thorough" setting.
