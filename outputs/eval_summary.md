# Eval Summary

Golden set size: 35

## Intent accuracy

| Model | Accuracy |
|---|---|
| agent | 0.457 |
| trivial_baseline | 0.143 |
| simple_baseline | 0.743 |

## Per-intent breakdown (agent)

| Intent | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| delivery_delay | 1.0 | 0.2 | 0.333 | 5 |
| order_status_inquiry | 1.0 | 0.8 | 0.889 | 5 |
| refund_or_return | 1.0 | 0.8 | 0.889 | 5 |
| damaged_or_wrong_item | 1.0 | 0.4 | 0.571 | 5 |
| account_or_login_issue | 1.0 | 0.4 | 0.571 | 5 |
| billing_or_charge_dispute | 1.0 | 0.6 | 0.75 | 5 |
| general_feedback | 0.0 | 0.0 | 0.0 | 5 |

## Escalation decision metrics

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| agent | 0.167 | 1.0 | 0.286 | 0.286 |
| trivial_baseline | 0.0 | 0.0 | 0.0 | 0.857 |
| simple_baseline | 1.0 | 1.0 | 1.0 | 1.0 |

## Reply quality (LLM judge, n=10)

| Dimension | Mean (1-5) |
|---|---|
| grounded | 0.4 |
| correct_intent_handling | 0.4 |
| tone | 0.4 |
| actionable | 0.4 |
| overall | 0.4 |
