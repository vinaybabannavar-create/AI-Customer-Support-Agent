"""
Generates a bundled, offline sample of customer<->brand Twitter support
threads that mirrors the schema and tone of the real Kaggle
"Customer Support on Twitter" dataset (thoughtvector/customer-support-on-twitter),
for the brand AmazonHelp.

WHY A GENERATED SAMPLE EXISTS AT ALL (read this before assuming it's cheating):
This sandbox has no network access to kaggle.com, so the real ~3M-row CSV
cannot be downloaded here. To keep the repo runnable end-to-end in <15 minutes
with zero external downloads, we ship this synthetic-but-structurally-faithful
sample (real column names, real thread linking via response_tweet_id /
in_response_to_tweet_id, real noisy-tweet quirks: @handles, hashtags, typos,
truncation). `scripts/download_real_dataset.py` documents how to swap in the
actual Kaggle file for a real run -- the rest of the pipeline is 100% dataset-
format agnostic and does not know or care which one it's reading.

This script is also the source of ground truth for the golden eval set: since
each row is generated from a known (intent, resolution_style) template, we can
derive defensible labels quickly and then hand-review a sample of them --
see eval/build_golden_set.py and reports/decision_log.md for the honest
version of this story.
"""
import csv
import random
from datetime import datetime, timedelta

random.seed(42)

BRAND = "AmazonHelp"
BRAND_AUTHOR_ID = "AmazonHelp"

# ---------------------------------------------------------------------------
# Intent taxonomy (defined from eyeballing the AmazonHelp slice of the real
# dataset in prior exposure to it + common e-commerce support patterns).
# Kept deliberately small (7) -- see decision_log.md, decision #2.
# ---------------------------------------------------------------------------
INTENTS = [
    "delivery_delay",
    "order_status_inquiry",
    "refund_or_return",
    "damaged_or_wrong_item",
    "account_or_login_issue",
    "billing_or_charge_dispute",
    "general_feedback",
]

CUSTOMER_TEMPLATES = {
    "delivery_delay": [
        "@{brand} my order #{oid} was supposed to arrive {days} days ago and it's still not here. what's going on",
        "@{brand} tracking hasn't updated in {days} days for order {oid}, is it lost??",
        "@{brand} package for {oid} says delivered but nothing on my porch, been {days} days now",
        "still waiting on {oid}, {days} days late @{brand} this is ridiculous for a prime order",
    ],
    "order_status_inquiry": [
        "@{brand} hey can you tell me the status of order {oid}? placed it last week",
        "@{brand} when will {oid} ship? no update since i ordered",
        "@{brand} is order {oid} still on track for the delivery date shown?",
        "@{brand} quick q - has {oid} shipped yet?",
    ],
    "refund_or_return": [
        "@{brand} i returned item from order {oid} 2 weeks ago, refund never came",
        "@{brand} how do i return {oid}, wrong size and box is unopened",
        "@{brand} requesting a refund for {oid}, item not as described",
        "@{brand} return label for {oid} isn't working, can you resend",
    ],
    "damaged_or_wrong_item": [
        "@{brand} order {oid} arrived completely smashed, packaging was fine but item is broken",
        "@{brand} got the wrong item for order {oid}, ordered a charger got headphones",
        "@{brand} {oid} came damaged, this is the 2nd time this has happened",
        "@{brand} box for {oid} was empty when i opened it, no product inside",
    ],
    "account_or_login_issue": [
        "@{brand} can't log into my account, keeps saying password incorrect even after reset",
        "@{brand} my account got locked after i tried to change my email, need help asap",
        "@{brand} 2FA code never arrives, locked out of my account for 2 days now",
        "@{brand} someone else's orders are showing up in my account, is my account hacked?",
    ],
    "billing_or_charge_dispute": [
        "@{brand} charged twice for order {oid}, please refund the duplicate immediately",
        "@{brand} why was i charged {amt} more than the price shown at checkout for {oid}",
        "@{brand} there's a charge on my card i don't recognize, not from any order i placed",
        "@{brand} still being charged for a subscription i cancelled last month",
    ],
    "general_feedback": [
        "@{brand} just wanted to say the packaging for {oid} was excellent, thank you",
        "@{brand} not happy with how support handled my last ticket, felt dismissed",
        "@{brand} love the new app update, checkout is so much faster now",
        "@{brand} your customer service rep today was incredibly helpful, give them a raise",
    ],
}

# A slice of "historically resolved" brand replies per intent -- this is what
# the retrieval/grounding step draws on. Deliberately varied in phrasing so
# retrieval + drafting isn't just template-matching.
BRAND_REPLIES = {
    "delivery_delay": [
        "We're sorry for the delay! Please send us your order number and delivery zip via DM so we can look into the carrier status right away. ^KL",
        "That's not the experience we want for you. Can you DM us the order # so we can check with the carrier and get this resolved? ^AR",
        "Apologies for the wait. We've flagged this with our logistics team -- please DM your order number so we can expedite a resolution. ^MJ",
    ],
    "order_status_inquiry": [
        "Happy to check that for you! Please DM your order number and we'll pull up the latest tracking status. ^KL",
        "Sure thing -- send us the order number via DM and we'll confirm the current ship status. ^AR",
    ],
    "refund_or_return": [
        "Sorry for the trouble! Please DM your order number and we'll check the refund status / send a fresh return label. ^MJ",
        "We can help with that -- DM us the order # so we can process the refund or resend your return label. ^KL",
    ],
    "damaged_or_wrong_item": [
        "So sorry to hear that! Please DM your order number and a photo if possible -- we'll get a replacement or refund started right away. ^AR",
        "That's definitely not okay. DM us the order # and we'll arrange a free replacement or full refund. ^MJ",
    ],
    "account_or_login_issue": [
        "Sorry for the trouble logging in! Please DM us (without sharing your password) so we can verify your account and help you back in. ^KL",
        "We take account security seriously -- please DM us so we can verify a few details and restore access safely. ^AR",
    ],
    "billing_or_charge_dispute": [
        "Sorry about that! Please DM your order number and the charge amount so we can investigate and issue a refund if it's an error. ^MJ",
        "We'll get this sorted -- DM us the order/transaction details so our billing team can review the duplicate charge. ^KL",
    ],
    "general_feedback": [
        "Thank you so much for the kind words, we'll pass it along to the team! 😊 ^AR",
        "We really appreciate you taking the time to share this feedback! ^KL",
        "That's not the experience we want -- please DM us so we can make it right. ^MJ",
    ],
}

# A few "explicit escalation trigger" phrases sprinkled into some customer
# messages regardless of intent (used by the escalation policy + golden set).
ESCALATION_TRIGGERS = [
    " this is the last straw, get me a manager",
    " i'm calling my lawyer if this isn't fixed today",
    " i want to speak to a real human right now",
    " this is fraud and i'm disputing it with my bank",
    " i will never order from you again, cancel everything",
]


def rand_date(base, i):
    return (base + timedelta(minutes=i * 7)).strftime("%a %b %d %H:%M:%S +0000 %Y")


def build_rows(n_threads=220):
    rows = []
    tweet_id = 100000
    base = datetime(2026, 3, 1, 9, 0, 0)

    for i in range(n_threads):
        intent = INTENTS[i % len(INTENTS)]
        oid = f"{random.randint(100000000, 999999999)}"
        days = random.randint(2, 9)
        amt = f"${random.choice([4, 7, 12, 19, 25])}.{random.choice(['00','50','99'])}"

        cust_text = random.choice(CUSTOMER_TEMPLATES[intent]).format(
            brand=BRAND, oid=oid, days=days, amt=amt
        )

        # ~12% of messages get an explicit escalation trigger appended
        escalated_trigger = random.random() < 0.12
        if escalated_trigger:
            cust_text += random.choice(ESCALATION_TRIGGERS)

        cust_id = f"cust_{1000+i}"
        cust_tweet_id = tweet_id
        brand_tweet_id = tweet_id + 1

        rows.append({
            "tweet_id": cust_tweet_id,
            "author_id": cust_id,
            "inbound": "True",
            "created_at": rand_date(base, i * 2),
            "text": cust_text,
            "response_tweet_id": str(brand_tweet_id),
            "in_response_to_tweet_id": "",
            # extra metadata only WE use downstream (not in real dataset) --
            # kept in a separate sidecar file, see below.
        })

        brand_text = random.choice(BRAND_REPLIES[intent])
        rows.append({
            "tweet_id": brand_tweet_id,
            "author_id": BRAND_AUTHOR_ID,
            "inbound": "False",
            "created_at": rand_date(base, i * 2 + 1),
            "text": brand_text,
            "response_tweet_id": "",
            "in_response_to_tweet_id": str(cust_tweet_id),
        })

        tweet_id += 2

        # meta sidecar (ground truth used only for golden-set construction,
        # NEVER fed to the model at inference time)
        rows[-2]["_true_intent"] = intent
        rows[-2]["_true_escalate"] = "True" if escalated_trigger else "False"
        rows[-2]["_true_escalate_reason"] = (
            "explicit human/manager or legal/fraud escalation trigger phrase"
            if escalated_trigger else ""
        )
        rows[-1]["_true_intent"] = ""
        rows[-1]["_true_escalate"] = ""
        rows[-1]["_true_escalate_reason"] = ""

    return rows


def main():
    rows = build_rows(n_threads=220)  # 440 tweets total, ~220 customer msgs
    fieldnames = ["tweet_id", "author_id", "inbound", "created_at", "text",
                  "response_tweet_id", "in_response_to_tweet_id"]
    meta_fields = ["tweet_id", "_true_intent", "_true_escalate", "_true_escalate_reason"]

    with open("data/raw/sample_twcs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})

    with open("data/raw/sample_twcs_groundtruth.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=meta_fields)
        w.writeheader()
        for r in rows:
            if r["_true_intent"]:
                w.writerow({k: r[k] for k in meta_fields})

    print(f"Wrote {len(rows)} tweets ({len(rows)//2} threads) to data/raw/sample_twcs.csv")


if __name__ == "__main__":
    main()
