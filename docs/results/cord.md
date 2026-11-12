# Results on CORD receipts: Qwen3-1.7B, `rows` view

Receipts: 400 to fit the scorer, 400 to choose approval cuts, 200 held out for every number below. Extraction took 13 s per receipt on an 8-core CPU, including three re-samples and the stated-confidence question.

## How good is the extraction before any routing

Field accuracy 82.8%; receipts with every field right 18.5%.

| field | accuracy | receipts | accuracy when the field is printed | printed on |
|---|---|---|---|---|
| total | 88.0% | 200 | 91.2% | 193 |
| subtotal | 67.8% | 199 | 91.6% | 131 |
| tax | 81.2% | 197 | 65.5% | 84 |
| service | 83.5% | 200 | 41.7% | 24 |
| discount | 96.0% | 200 | 75.0% | 4 |
| cash | 91.9% | 198 | 89.8% | 128 |
| change | 93.0% | 200 | 85.1% | 87 |
| card | 93.0% | 199 | 55.2% | 29 |
| item_count | 51.0% | 200 | 51.0% | 200 |

## Which score finds the wrong values

AUROC is the chance a right value outscores a wrong one. AURC is the average error rate over all approval levels. The last column is read off the test set itself, so it is optimistic; the certified numbers are below.

| score | AUROC (95% CI) | AURC, lower is better | approved at ≤2% observed error |
|---|---|---|---|
| model probability of the value | 0.736 (0.701-0.768) | 0.0886 (0.0723-0.1080) | 0.0% (0.0-0.0) |
| model's stated confidence | 0.410 (0.380-0.441) | 0.1994 (0.1744-0.2262) | 0.0% (0.0-0.0) |
| agreement of 3 re-samples | 0.567 (0.546-0.593) | 0.1518 (0.1330-0.1707) | 0.0% (0.0-0.0) |
| combined, agreement signals only | 0.834 (0.807-0.858) | 0.0539 (0.0441-0.0658) | 20.7% (15.1-28.4) |
| combined, source signals only | 0.926 (0.908-0.941) | 0.0312 (0.0250-0.0391) | 61.9% (46.7-70.6) |
| combined, rules signals only | 0.843 (0.819-0.867) | 0.0506 (0.0417-0.0617) | 34.9% (8.4-42.5) |
| combined, model signals only | 0.914 (0.895-0.931) | 0.0340 (0.0274-0.0424) | 55.3% (47.1-63.7) |
| combined, all signals | 0.949 (0.935-0.961) | 0.0260 (0.0208-0.0320) | 69.9% (62.6-75.3) |
| combined, all signals (logistic regression instead) | 0.916 (0.896-0.934) | 0.0356 (0.0276-0.0453) | 50.2% (41.8-68.6) |

The combined scorer is boosted trees, picked on the fit receipts alone by out-of-fold AURC (logistic 0.0448, boosted 0.0352).

Largest standardised weights in the logistic-regression scorer, the version whose reasoning can be read out: `label_on_row` +1.75, `ocr_conf` +1.67, `printed` -1.64, `null:item_count` -0.93, `sum_ok` +0.77, `null:service` +0.76, `tokens` -0.76, `label_in_doc` -0.75.

## Certified auto-approval

For each target, the cut was chosen on the calibration receipts so that, with 95% confidence, the error rate among approved decisions is at most the target. "Approved on test" and "wrong on test" are what then happened on receipts the cut never saw.

### Target: at most 2% wrong among approved

| decision | error rate with no routing | cut | approved on test | wrong on test (95% interval) |
|---|---|---|---|---|
| total | 12.0% | none certified | 0 of 200 | review all |
| subtotal | 32.2% | none certified | 0 of 199 | review all |
| tax | 18.8% | none certified | 0 of 197 | review all |
| service | 16.5% | 0.88 | 151 of 200 (76%) | 2 = 1.3% (0.4%-4.7%) |
| discount | 4.0% | 0.92 | 185 of 200 (92%) | 2 = 1.1% (0.3%-3.9%) |
| cash | 8.1% | none certified | 0 of 198 | review all |
| change | 7.0% | 0.98 | 73 of 200 (36%) | 1 = 1.4% (0.2%-7.4%) |
| card | 7.0% | none certified | 0 of 199 | review all |
| item_count | 49.0% | none certified | 0 of 200 | review all |
| **whole receipt** | 81.5% | none certified | 0 of 200 | review all |

### Target: at most 5% wrong among approved

| decision | error rate with no routing | cut | approved on test | wrong on test (95% interval) |
|---|---|---|---|---|
| total | 12.0% | 0.74 | 162 of 200 (81%) | 5 = 3.1% (1.3%-7.0%) |
| subtotal | 32.2% | none certified | 0 of 199 | review all |
| tax | 18.8% | none certified | 0 of 197 | review all |
| service | 16.5% | 0.73 | 168 of 200 (84%) | 7 = 4.2% (2.0%-8.3%) |
| discount | 4.0% | 0.55 | 196 of 200 (98%) | 5 = 2.6% (1.1%-5.8%) |
| cash | 8.1% | 0.9 | 137 of 198 (69%) | 1 = 0.7% (0.1%-4.0%) |
| change | 7.0% | 0.78 | 180 of 200 (90%) | 5 = 2.8% (1.2%-6.3%) |
| card | 7.0% | none certified | 0 of 199 | review all |
| item_count | 49.0% | none certified | 0 of 200 | review all |
| **whole receipt** | 81.5% | none certified | 0 of 200 | review all |

### Target: at most 10% wrong among approved

| decision | error rate with no routing | cut | approved on test | wrong on test (95% interval) |
|---|---|---|---|---|
| total | 12.0% | 0.5 | 177 of 200 (88%) | 9 = 5.1% (2.7%-9.4%) |
| subtotal | 32.2% | 0.77 | 123 of 199 (62%) | 2 = 1.6% (0.4%-5.7%) |
| tax | 18.8% | 0.8 | 153 of 197 (78%) | 7 = 4.6% (2.2%-9.1%) |
| service | 16.5% | 0.5 | 176 of 200 (88%) | 12 = 6.8% (3.9%-11.5%) |
| discount | 4.0% | 0.5 | 196 of 200 (98%) | 5 = 2.6% (1.1%-5.8%) |
| cash | 8.1% | 0.74 | 185 of 198 (93%) | 9 = 4.9% (2.6%-9.0%) |
| change | 7.0% | 0.5 | 196 of 200 (98%) | 11 = 5.6% (3.2%-9.8%) |
| card | 7.0% | 0.5 | 188 of 199 (94%) | 6 = 3.2% (1.5%-6.8%) |
| item_count | 49.0% | none certified | 0 of 200 | review all |
| **whole receipt** | 81.5% | none certified | 0 of 200 | review all |

## Certified cut against the usual way of picking one

The receipts were re-split 200 times. Each time, every field got a cut chosen on the calibration part two ways and checked on the held-out part. "Missed target" is the share of those cuts whose held-out error rate among approved values came out above target. The held-out part is small, so a cut whose true error rate is under target can still miss by chance; the comparison between the two rows is the point.

| target | method | missed target | mean share approved | mean error among approved | nothing approved |
|---|---|---|---|---|---|
| 2% | loosest cut that looked fine on calibration | 44% | 56% | 2.4% | 4% |
| 2% | certified cut | 3% | 13% | 0.2% | 83% |
| 5% | loosest cut that looked fine on calibration | 40% | 75% | 4.8% | 1% |
| 5% | certified cut | 8% | 54% | 2.0% | 30% |
| 10% | loosest cut that looked fine on calibration | 16% | 81% | 6.7% | 0% |
| 10% | certified cut | 4% | 77% | 5.0% | 11% |

The same comparison for each decision on its own, as missed target / mean share approved:

| decision | usual way, 2% | certified, 2% | usual way, 5% | certified, 5% | usual way, 10% | certified, 10% |
|---|---|---|---|---|---|---|
| total | 50% / 66% | 4% / 4% | 46% / 83% | 10% / 68% | 3% / 87% | 2% / 87% |
| subtotal | 46% / 36% | 0% / 0% | 44% / 53% | 8% / 19% | 14% / 60% | 6% / 56% |
| tax | 46% / 59% | 2% / 3% | 48% / 80% | 10% / 56% | 29% / 87% | 8% / 85% |
| service | 46% / 84% | 8% / 58% | 29% / 87% | 12% / 86% | 1% / 88% | 1% / 88% |
| discount | 48% / 96% | 8% / 35% | 0% / 97% | 0% / 96% | 0% / 97% | 0% / 97% |
| cash | 56% / 60% | 2% / 2% | 52% / 77% | 16% / 63% | 44% / 94% | 14% / 87% |
| change | 48% / 79% | 4% / 13% | 44% / 96% | 15% / 87% | 0% / 97% | 0% / 97% |
| card | 30% / 18% | 0% / 0% | 48% / 94% | 4% / 11% | 4% / 97% | 3% / 97% |
| item_count | 30% / 3% | 0% / 0% | 48% / 7% | 0% / 0% | 52% / 20% | 1% / 0% |
| **whole receipt** | 28% / 1% | 0% / 0% | 31% / 1% | 0% / 0% | 46% / 3% | 0% / 0% |

## What the routing is worth

Per receipt, on the test receipts. Labour assumes a reviewer at $18 an hour who takes 12 s to open a receipt and 6 s per field and catches every error in what they check. The last columns price each wrong value that ships at three levels, because that price decides everything: skipping review only pays while the error rate among approved fields is below the cost of one check divided by the cost of one error (12.0% at $0.25, 3.0% at $1.00, 0.6% at $5.00). These are placeholders for a customer's own figures.

| policy | receipts no one opens | fields checked | wrong values shipped per 100 receipts | labour | total at $0.25 an error | total at $1.00 an error | total at $5.00 an error |
|---|---|---|---|---|---|---|---|
| a person checks every field | 0% | 100% | 0.0 | $0.329 | $0.329 | $0.329 | $0.329 |
| nothing is checked | 100% | 0% | 154.0 | $0.000 | $0.385 | $1.540 | $7.700 |
| certified cuts, target 2% | 0% | 77% | 2.5 | $0.268 | $0.274 | $0.293 | $0.393 |
| certified cuts, target 5% | 0% | 53% | 11.5 | $0.203 | $0.231 | $0.318 | $0.778 |
| certified cuts, target 10% | 0% | 22% | 30.5 | $0.120 | $0.196 | $0.425 | $1.645 |

## Does rebuilding the rows matter

The same 200 receipts, extracted from two renderings of the same OCR: docling's reading-order export (`reading`) and rows rebuilt from the OCR boxes (`rows`).

| view | field accuracy | receipts with every field right | total | subtotal | tax | service | discount | cash | change | card | item_count |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `rows` | 82.8% | 18.5% | 88% | 68% | 81% | 84% | 96% | 92% | 93% | 93% | 51% |
| `reading` | 58.8% | 4.0% | 63% | 49% | 56% | 50% | 76% | 43% | 62% | 85% | 46% |

## How many labelled receipts a target needs

Fewest approved, labelled decisions that can certify a target at 95% confidence. This is arithmetic, not a measurement, and it is the first thing to tell a customer who asks for 99.9%.

| target error rate | with 0 wrong | with 1 wrong | with 3 wrong |
|---|---|---|---|
| 10.0% | 29 | 46 | 76 |
| 5.0% | 59 | 93 | 153 |
| 2.0% | 149 | 236 | 386 |
| 1.0% | 299 | 473 | 773 |
| 0.5% | 598 | 947 | 1,549 |
| 0.1% | 2,995 | 4,742 | 7,752 |
