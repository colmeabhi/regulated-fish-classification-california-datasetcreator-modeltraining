# Hierarchical classifier — `test` split

757 images. 95% Wilson confidence intervals in brackets.

## Headline

| Metric | Value |
|---|---|
| End-to-end species accuracy | 73.7% [70.5–76.7] (558/757) |
| Stage 1 category accuracy | 85.2% [82.5–87.6] (645/757) |
| **Prohibited predicted as legal** | **0.8% [0.2–3.0] (2/236)** |
| Prohibited predicted as restricted | 13.6% [9.8–18.5] (32/236) |
| Restricted predicted as legal | 8.9% [6.3–12.4] (30/337) |
| Legal flagged as prohibited (false alarm) | 1.1% [0.3–3.9] (2/184) |

## Per-category recall / precision (Stage 1)

| Category | Recall | Precision |
|---|---|---|
| prohibited | 85.6% [80.5–89.5] (202/236) | 91.4% [87.0–94.4] (202/221) |
| restricted | 86.1% [81.9–89.3] (290/337) | 82.6% [78.3–86.2] (290/351) |
| legal | 83.2% [77.1–87.9] (153/184) | 82.7% [76.6–87.5] (153/185) |

## Category confusion matrix (rows = true)

| true \ pred | prohibited | restricted | legal |
|---|---|---|---|
| prohibited | 202 | 32 | 2 |
| restricted | 17 | 290 | 30 |
| legal | 2 | 29 | 153 |

## Stage 2 with oracle routing

Species accuracy when each image is sent to the species model of its *true* category; the gap to end-to-end accuracy is the cost of Stage 1 mistakes.

| Category | Oracle Stage 2 accuracy |
|---|---|
| prohibited | 83.1% [77.7–87.3] (196/236) |
| restricted | 81.0% [76.5–84.8] (273/337) |
| legal | 88.0% [82.6–92.0] (162/184) |

## Per-species

| Species | True category | n | End-to-end recall | Category correct | Predicted legal |
|---|---|---|---|---|---|
| bronzespotted | prohibited | 8 | 100.0% [67.6–100.0] (8/8) | 100.0% [67.6–100.0] (8/8) | 0.0% [0.0–32.4] (0/8) |
| broomtail_grouper | prohibited | 6 | 16.7% [3.0–56.4] (1/6) | 100.0% [61.0–100.0] (6/6) | 0.0% [0.0–39.0] (0/6) |
| coho_salmon | prohibited | 34 | 44.1% [28.9–60.5] (15/34) | 64.7% [47.9–78.5] (22/34) | 0.0% [0.0–10.2] (0/34) |
| cowcod | prohibited | 8 | 25.0% [7.1–59.1] (2/8) | 87.5% [52.9–97.8] (7/8) | 0.0% [0.0–32.4] (0/8) |
| garibaldi | prohibited | 37 | 81.1% [65.8–90.5] (30/37) | 91.9% [78.7–97.2] (34/37) | 0.0% [0.0–9.4] (0/37) |
| giant_sea_bass | prohibited | 54 | 98.1% [90.2–99.7] (53/54) | 98.1% [90.2–99.7] (53/54) | 0.0% [0.0–6.6] (0/54) |
| green_sturgeon | prohibited | 19 | 57.9% [36.3–76.9] (11/19) | 57.9% [36.3–76.9] (11/19) | 0.0% [0.0–16.8] (0/19) |
| gulf_grouper | prohibited | 8 | 62.5% [30.6–86.3] (5/8) | 100.0% [67.6–100.0] (8/8) | 0.0% [0.0–32.4] (0/8) |
| quillback | prohibited | 6 | 66.7% [30.0–90.3] (4/6) | 66.7% [30.0–90.3] (4/6) | 33.3% [9.7–70.0] (2/6) ⚠️ |
| steelhead_ocean | prohibited | 21 | 66.7% [45.4–82.8] (14/21) | 90.5% [71.1–97.3] (19/21) | 0.0% [0.0–15.5] (0/21) |
| white_shark | prohibited | 27 | 88.9% [71.9–96.1] (24/27) | 88.9% [71.9–96.1] (24/27) | 0.0% [0.0–12.5] (0/27) |
| yelloweye | prohibited | 8 | 75.0% [40.9–92.9] (6/8) | 75.0% [40.9–92.9] (6/8) | 0.0% [0.0–32.4] (0/8) |
| barred_sand_bass | restricted | 37 | 89.2% [75.3–95.7] (33/37) | 100.0% [90.6–100.0] (37/37) | 0.0% [0.0–9.4] (0/37) |
| california_halibut | restricted | 28 | 89.3% [72.8–96.3] (25/28) | 96.4% [82.3–99.4] (27/28) | 3.6% [0.6–17.7] (1/28) ⚠️ |
| california_sheephead | restricted | 29 | 62.1% [44.0–77.3] (18/29) | 89.7% [73.6–96.4] (26/29) | 10.3% [3.6–26.4] (3/29) ⚠️ |
| canary | restricted | 18 | 72.2% [49.1–87.5] (13/18) | 83.3% [60.8–94.2] (15/18) | 16.7% [5.8–39.2] (3/18) ⚠️ |
| chinook_salmon | restricted | 30 | 40.0% [24.6–57.7] (12/30) | 60.0% [42.3–75.4] (18/30) | 3.3% [0.6–16.7] (1/30) ⚠️ |
| copper | restricted | 42 | 64.3% [49.2–77.0] (27/42) | 81.0% [66.7–90.0] (34/42) | 19.0% [10.0–33.3] (8/42) ⚠️ |
| kelp_bass | restricted | 28 | 75.0% [56.6–87.3] (21/28) | 89.3% [72.8–96.3] (25/28) | 10.7% [3.7–27.2] (3/28) ⚠️ |
| leopard_shark | restricted | 19 | 78.9% [56.7–91.5] (15/19) | 89.5% [68.6–97.1] (17/19) | 5.3% [0.9–24.6] (1/19) ⚠️ |
| pacific_halibut | restricted | 8 | 100.0% [67.6–100.0] (8/8) | 100.0% [67.6–100.0] (8/8) | 0.0% [0.0–32.4] (0/8) |
| spotted_sand_bass | restricted | 31 | 61.3% [43.8–76.3] (19/31) | 90.3% [75.1–96.7] (28/31) | 9.7% [3.3–24.9] (3/31) ⚠️ |
| vermilion | restricted | 37 | 83.8% [68.9–92.3] (31/37) | 86.5% [72.0–94.1] (32/37) | 8.1% [2.8–21.3] (3/37) ⚠️ |
| white_seabass | restricted | 30 | 70.0% [52.1–83.3] (21/30) | 76.7% [59.1–88.2] (23/30) | 13.3% [5.3–29.7] (4/30) ⚠️ |
| black | legal | 42 | 76.2% [61.5–86.5] (32/42) | 85.7% [72.2–93.3] (36/42) | 85.7% [72.2–93.3] (36/42) |
| blue | legal | 57 | 86.0% [74.7–92.7] (49/57) | 87.7% [76.8–93.9] (50/57) | 87.7% [76.8–93.9] (50/57) |
| gopher | legal | 45 | 75.6% [61.3–85.8] (34/45) | 82.2% [68.7–90.7] (37/45) | 82.2% [68.7–90.7] (37/45) |
| olive | legal | 40 | 67.5% [52.0–79.9] (27/40) | 75.0% [59.8–85.8] (30/40) | 75.0% [59.8–85.8] (30/40) |

## Prohibited-threshold sweep

Rule: predict *prohibited* if P(prohibited) ≥ t, otherwise the more likely of restricted/legal. Pick t on the **validation** split; picking it on test makes the test number optimistic.

| t | Prohibited → legal | Prohibited recall | Legal flagged prohibited | Category accuracy |
|---|---|---|---|---|
| 0.5 | 0.8% [0.2–3.0] (2/236) | 85.6% [80.5–89.5] (202/236) | 1.1% [0.3–3.9] (2/184) | 85.2% [82.5–87.6] (645/757) |
| 0.4 | 0.8% [0.2–3.0] (2/236) | 86.4% [81.5–90.2] (204/236) | 1.1% [0.3–3.9] (2/184) | 85.1% [82.4–87.4] (644/757) |
| 0.3 | 0.8% [0.2–3.0] (2/236) | 86.9% [82.0–90.6] (205/236) | 2.2% [0.8–5.5] (4/184) | 84.5% [81.8–86.9] (640/757) |
| 0.2 | 0.8% [0.2–3.0] (2/236) | 88.1% [83.4–91.7] (208/236) | 2.2% [0.8–5.5] (4/184) | 84.7% [81.9–87.1] (641/757) |
| 0.1 | 0.8% [0.2–3.0] (2/236) | 91.1% [86.8–94.1] (215/236) | 3.3% [1.5–6.9] (6/184) | 84.8% [82.1–87.2] (642/757) |
| 0.05 | 0.4% [0.1–2.4] (1/236) | 93.2% [89.3–95.8] (220/236) | 3.8% [1.9–7.6] (7/184) | 84.7% [81.9–87.1] (641/757) |

## Leakage check

0 of 757 images share an iNaturalist observation with a training image (status: **ok**).

Critical errors (prohibited not predicted prohibited) are listed in `hierarchical_test_critical_errors.csv`.
