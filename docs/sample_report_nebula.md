# Sample Report: Nebula (Trustpilot)

**App / source:** [Nebula on Trustpilot](https://www.trustpilot.com/review/asknebula.com)  
**Dataset:** `nebula` (`trustpilot`)  
**Sample size:** 100 reviews  
**Sentiment model:** fine-tuned RoBERTa (`data/models/roberta_sentiment`)  
**Generated from:** `GET /api/v1/analysis?dataset=nebula` / `scripts/visualize_insights.py --dataset nebula`

---

## Executive summary

Overall tone is **positive** (avg sentiment **+0.66**). Native Trustpilot average rating is **4.24 / 5**. Most reviewers praise readings and product usefulness, while a vocal minority flags billing, cancellations, and trust concerns.

> NLP analysis of 100 reviews (trustpilot=100). Sentiment — positive: 83, neutral: 2, negative: 15. Overall tone: positive (avg sentiment +0.66). Average native rating: 4.24.

---

## Sentiment distribution

| Label | Count | Share |
|-------|------:|------:|
| Positive | 83 | 83% |
| Neutral | 2 | 2% |
| Negative | 15 | 15% |

![Sentiment and rating overview](../data/charts/nebula_insights_overview.png)

---

## Rating distribution

| Stars | Count | Share |
|------:|------:|------:|
| 5★ | 63 | 63.0% |
| 4★ | 17 | 17.0% |
| 3★ | 10 | 10.0% |
| 2★ | 1 | 1.0% |
| 1★ | 9 | 9.0% |

**Average rating:** 4.24 (native Trustpilot stars only).

---

## Key insights

### High priority risks

1. **Common language in negative reviews**  
   From 15 negative-sentiment reviews, top terms/phrases include: *money*, *sketch*, *time*, *fraudulently took money*.

2. **Subscription / billing is a recurring topic**  
   10 reviews mention pricing/subscription (avg sentiment −0.40).

3. **Low star ratings need response playbooks**  
   10 reviews are rated 1–2 stars in this sample.

### Strengths

4. **Product/reading quality is frequently discussed**  
   19 reviews mention product/reading quality (avg sentiment +0.33).

5. **Majority sentiment is positive**  
   83 of 100 reviews are positive (83%). Amplify these themes in marketing.

---

## Negative keywords & phrases

![Top terms in negative reviews](../data/charts/nebula_negative_keywords.png)

Frequent complaint language clusters around money/account charges and perceived fraudulent billing — useful for triage tags and support macros.

---

## Recommended actions

1. **Billing & cancellations:** make cancel/refund flows clearer and confirm charges stop immediately.
2. **Trust & transparency:** publish pricing upfront and respond publicly to scam/fraud accusations.
3. **Advisor quality:** raise reading specificity standards and reduce generic/template responses.
4. **Credits/pricing UX:** clarify credit burn rate before chat and reduce surprise upsells.
5. **Support / communication:** set clear SLA for replies and close the loop on open issues.
6. **Response ops:** reply to every new 1–2★ Trustpilot review within 24–48 hours.

---

## How to regenerate this report

```bash
source .venv/bin/activate
python scripts/train_roberta_sentiment.py   # if model missing
python scripts/visualize_insights.py --dataset nebula
```

Or via the API (with the server running):

```bash
curl -s "http://127.0.0.1:8000/api/v1/analysis?dataset=nebula" | python3 -m json.tool
```
