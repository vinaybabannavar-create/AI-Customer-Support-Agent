# Decision Log

Non-obvious decisions made while building this, and the reasoning behind
each. Ordered roughly by where they show up in the pipeline.

1. **Didn't clean/normalize the tweet text beyond whitespace collapsing**
   (`src/data_prep.py`). Hashtags, emoji, typos, and @-mentions are left
   in. Real customer support traffic is noisy; a classifier/drafter that
   only works on cleaned text is testing a different, easier problem than
   the one being asked. Stripping @AmazonHelp specifically was considered
   and rejected — it's a real signal for "this tweet is directed at the
   brand," which matters if you're deduping which tweets are even in-scope.

2. **Kept the intent taxonomy to 7 labels, not the 77 of Banking77 or a
   long tail of brand-specific micro-intents.** A take-home evaluated by a
   human reader needs a taxonomy that reader can hold in their head and
   sanity-check by eye. 7 also maps cleanly onto genuinely different
   *handling* paths (see decision #4) — the taxonomy is built around "what
   changes downstream," not "how many ways can a message be phrased."

3. **The golden set's labels come from generation-time ground truth
   (this environment couldn't reach kaggle.com), followed by a manual
   spot-check pass**, not blind human labelling of real unseen data. This
   is flagged prominently in the README, the report, and
   `eval/build_golden_set.py`'s docstring rather than presented as
   equivalent to real hand-labelling — see report section 4.

4. **Escalation eligibility is gated by intent, not just by confidence.**
   Even a 99%-confidence `billing_or_charge_dispute` classification always
   escalates, because a bot shouldn't unilaterally resolve a billing
   dispute regardless of how sure the classifier is. Confidence thresholds
   only matter *within* the auto-handle-eligible intents. This is a policy
   call about what a bot should be trusted to decide, not a modeling
   choice, and it's the main reason escalation "precision" looks bad
   against the naive ground truth (see report section 3, failure #1) —
   that's treated as the ground truth being narrow, not the policy being
   wrong.

5. **Escalation is a deterministic Python function (`src/escalation.py`),
   not an LLM call.** An LLM asked "should this escalate?" will happily
   rationalize either answer depending on prompt phrasing, and it's much
   harder to unit-test or explain to a compliance reviewer than an
   if/elif chain with named constants. The classifier's job is to
   understand the message; the policy's job is to apply the brand's risk
   tolerance to that understanding — kept as two separable concerns on
   purpose.

6. **Grounding retrieval uses TF-IDF + cosine similarity, not an
   embedding API call.** Zero extra cost, zero extra latency, fully
   offline/deterministic, and transparent (you can literally see which
   words drove a match). It will miss paraphrases an embedding model would
   catch — accepted as a known limitation (see report section 5, next
   steps #4) rather than solved in this pass, because the retrieval
   interface (`ThreadIndex.top_k`) is intentionally small enough to swap
   the implementation later without touching `agent.py`.

7. **An `escalate` decision still produces a drafted reply**, not silence.
   The reply is meant for a human agent to review/edit/send, not to be
   auto-posted. "Escalate" means "needs a human in the loop before it goes
   out," not "the AI has nothing to contribute" — a human reviewer moves
   faster with a starting draft than a blank box.

8. **Both baselines were designed to be genuinely competitive on at least
   one axis**, not strawmen. The simple keyword baseline was given the
   *same* hard-escalation-keyword rule the agent's policy partially relies
   on — on purpose, so that any "the agent beats the simple baseline"
   claim would have to come from the classifier/drafter, not from a rigged
   comparison. This backfired informatively: it's exactly why the
   escalation-metrics table in the report needed the caveat in section 4.

9. **Shipped a mock LLM mode (`src/llm_client.py`) rather than requiring
   an API key to run anything.** A repo a reviewer can't run at all
   without first getting a paid API key and waiting on rate limits is a
   worse deliverable than one that's honestly labeled "these numbers need
   a real key" but runs end-to-end regardless. The risk (someone mistakes
   mock numbers for real ones) is handled by surfacing it repeatedly and
   loudly in the README and report rather than by making the repo
   unusable offline.

10. **Judge-vs-human calibration uses a FIXED, saved set of 30 drafts**
    (`outputs/calibration_drafts.jsonl`), not fresh draft generation each
    time the judge is scored. An LLM's draft output isn't perfectly
    deterministic; comparing a human's score of draft A against a judge's
    score of a freshly-regenerated draft B would silently invalidate the
    comparison. Freezing the drafts first was worth the extra script
    (`eval/make_calibration_drafts.py`).

11. **The judge rubric has a dedicated `grounded` dimension, scored
    separately from `overall`,** specifically to catch hallucinated
    promises/policies — the single worst failure mode identified in the
    problem-framing section. A judge that only outputs one holistic score
    would let a confidently-worded hallucination slip through with a
    decent score.

12. **`order_status_inquiry` was accepted as a deliberate "catch-all" risk
    rather than redesigned mid-build**, once it showed up as the mock
    classifier's fallback default and the worst-precision intent. Fixing
    the taxonomy properly (adding an explicit `other/unclear` intent)
    is called out as next-week work rather than patched reactively, to
    avoid quietly re-labelling the golden set to flatter the metric.

13. **Chose AmazonHelp over higher-traffic brands like AppleSupport**
    because its replies have a consistent, quotable house style ("DM us
    your order number") that makes grounding failures easy to spot by eye
    — useful both for building the drafting prompt and for a human grader
    skimming outputs.

14. **Did not attempt to auto-detect PII (customer names, addresses) and
    scrub it from grounding examples or drafts.** Out of scope for a
    take-home focused on the classify/draft/escalate loop, but flagged
    here because a real deployment absolutely needs this before any
    retrieved historical example gets shown to a different customer.

15. **Supported both Anthropic and Gemini as interchangeable LLM
    backends** (`LLM_PROVIDER` env var), rather than hard-coding one
    vendor. `src/llm_client.py` is the only file that knows which API is
    being called; `src/intents.py`, `src/drafting.py`, `eval/judge.py`,
    etc. never change. This was a practical call (Gemini's free tier
    removes a cost barrier to actually running this end-to-end) rather
    than an architectural preference for one API over the other.
