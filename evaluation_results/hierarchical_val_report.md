# Hierarchical classifier — `val` split

734 images. 95% Wilson confidence intervals in brackets.

## Headline

| Metric | Value |
|---|---|
| End-to-end species accuracy | 74.7% [71.4–77.7] (548/734) |
| Stage 1 category accuracy | 85.4% [82.7–87.8] (627/734) |
| **Prohibited predicted as legal** | **6.5% [3.9–10.6] (14/215)** |
| Prohibited predicted as restricted | 11.2% [7.6–16.1] (24/215) |
| Restricted predicted as legal | 7.1% [4.8–10.2] (25/353) |
| Legal flagged as prohibited (false alarm) | 1.8% [0.6–5.2] (3/166) |

## Per-category recall / precision (Stage 1)

| Category | Recall | Precision |
|---|---|---|
| prohibited | 82.3% [76.7–86.8] (177/215) | 90.3% [85.4–93.7] (177/196) |
| restricted | 88.4% [84.6–91.3] (312/353) | 86.4% [82.5–89.6] (312/361) |
| legal | 83.1% [76.7–88.1] (138/166) | 78.0% [71.3–83.4] (138/177) |

## Category confusion matrix (rows = true)

| true \ pred | prohibited | restricted | legal |
|---|---|---|---|
| prohibited | 177 | 24 | 14 |
| restricted | 16 | 312 | 25 |
| legal | 3 | 25 | 138 |

## Stage 2 with oracle routing

Species accuracy when each image is sent to the species model of its *true* category; the gap to end-to-end accuracy is the cost of Stage 1 mistakes.

| Category | Oracle Stage 2 accuracy |
|---|---|
| prohibited | 86.0% [80.8–90.0] (185/215) |
| restricted | 81.0% [76.6–84.8] (286/353) |
| legal | 87.3% [81.4–91.6] (145/166) |

## Per-species

| Species | True category | n | End-to-end recall | Category correct | Predicted legal |
|---|---|---|---|---|---|
| bronzespotted | prohibited | 7 | 100.0% [64.6–100.0] (7/7) | 100.0% [64.6–100.0] (7/7) | 0.0% [0.0–35.4] (0/7) |
| broomtail_grouper | prohibited | 9 | 22.2% [6.3–54.7] (2/9) | 66.7% [35.4–87.9] (6/9) | 11.1% [2.0–43.5] (1/9) ⚠️ |
| coho_salmon | prohibited | 24 | 75.0% [55.1–88.0] (18/24) | 79.2% [59.5–90.8] (19/24) | 8.3% [2.3–25.8] (2/24) ⚠️ |
| cowcod | prohibited | 7 | 71.4% [35.9–91.8] (5/7) | 85.7% [48.7–97.4] (6/7) | 0.0% [0.0–35.4] (0/7) |
| garibaldi | prohibited | 31 | 93.5% [79.3–98.2] (29/31) | 93.5% [79.3–98.2] (29/31) | 0.0% [0.0–11.0] (0/31) |
| giant_sea_bass | prohibited | 20 | 100.0% [83.9–100.0] (20/20) | 100.0% [83.9–100.0] (20/20) | 0.0% [0.0–16.1] (0/20) |
| green_sturgeon | prohibited | 18 | 38.9% [20.3–61.4] (7/18) | 55.6% [33.7–75.4] (10/18) | 22.2% [9.0–45.2] (4/18) ⚠️ |
| gulf_grouper | prohibited | 7 | 85.7% [48.7–97.4] (6/7) | 100.0% [64.6–100.0] (7/7) | 0.0% [0.0–35.4] (0/7) |
| quillback | prohibited | 13 | 46.2% [23.2–70.9] (6/13) | 46.2% [23.2–70.9] (6/13) | 30.8% [12.7–57.6] (4/13) ⚠️ |
| steelhead_ocean | prohibited | 34 | 58.8% [42.2–73.6] (20/34) | 73.5% [56.9–85.4] (25/34) | 8.8% [3.0–23.0] (3/34) ⚠️ |
| white_shark | prohibited | 37 | 100.0% [90.6–100.0] (37/37) | 100.0% [90.6–100.0] (37/37) | 0.0% [0.0–9.4] (0/37) |
| yelloweye | prohibited | 8 | 37.5% [13.7–69.4] (3/8) | 62.5% [30.6–86.3] (5/8) | 0.0% [0.0–32.4] (0/8) |
| barred_sand_bass | restricted | 25 | 84.0% [65.3–93.6] (21/25) | 96.0% [80.5–99.3] (24/25) | 4.0% [0.7–19.5] (1/25) ⚠️ |
| california_halibut | restricted | 33 | 66.7% [49.6–80.2] (22/33) | 97.0% [84.7–99.5] (32/33) | 3.0% [0.5–15.3] (1/33) ⚠️ |
| california_sheephead | restricted | 31 | 74.2% [56.8–86.3] (23/31) | 87.1% [71.1–94.9] (27/31) | 0.0% [0.0–11.0] (0/31) |
| canary | restricted | 18 | 61.1% [38.6–79.7] (11/18) | 77.8% [54.8–91.0] (14/18) | 22.2% [9.0–45.2] (4/18) ⚠️ |
| chinook_salmon | restricted | 32 | 53.1% [36.4–69.1] (17/32) | 78.1% [61.2–89.0] (25/32) | 0.0% [0.0–10.7] (0/32) |
| copper | restricted | 41 | 73.2% [58.1–84.3] (30/41) | 82.9% [68.7–91.5] (34/41) | 17.1% [8.5–31.3] (7/41) ⚠️ |
| kelp_bass | restricted | 35 | 82.9% [67.3–91.9] (29/35) | 94.3% [81.4–98.4] (33/35) | 5.7% [1.6–18.6] (2/35) ⚠️ |
| leopard_shark | restricted | 22 | 95.5% [78.2–99.2] (21/22) | 95.5% [78.2–99.2] (21/22) | 0.0% [0.0–14.9] (0/22) |
| pacific_halibut | restricted | 7 | 100.0% [64.6–100.0] (7/7) | 100.0% [64.6–100.0] (7/7) | 0.0% [0.0–35.4] (0/7) |
| spotted_sand_bass | restricted | 30 | 80.0% [62.7–90.5] (24/30) | 86.7% [70.3–94.7] (26/30) | 6.7% [1.8–21.3] (2/30) ⚠️ |
| vermilion | restricted | 41 | 82.9% [68.7–91.5] (34/41) | 85.4% [71.6–93.1] (35/41) | 12.2% [5.3–25.5] (5/41) ⚠️ |
| white_seabass | restricted | 38 | 57.9% [42.2–72.1] (22/38) | 89.5% [75.9–95.8] (34/38) | 7.9% [2.7–20.8] (3/38) ⚠️ |
| black | legal | 37 | 75.7% [59.9–86.6] (28/37) | 86.5% [72.0–94.1] (32/37) | 86.5% [72.0–94.1] (32/37) |
| blue | legal | 50 | 64.0% [50.1–75.9] (32/50) | 66.0% [52.2–77.6] (33/50) | 66.0% [52.2–77.6] (33/50) |
| gopher | legal | 39 | 89.7% [76.4–95.9] (35/39) | 89.7% [76.4–95.9] (35/39) | 89.7% [76.4–95.9] (35/39) |
| olive | legal | 40 | 80.0% [65.2–89.5] (32/40) | 95.0% [83.5–98.6] (38/40) | 95.0% [83.5–98.6] (38/40) |

## Prohibited-threshold sweep

Rule: predict *prohibited* if P(prohibited) ≥ t, otherwise the more likely of restricted/legal. Pick t on the **validation** split; picking it on test makes the test number optimistic.

| t | Prohibited → legal | Prohibited recall | Legal flagged prohibited | Category accuracy |
|---|---|---|---|---|
| 0.5 | 7.0% [4.3–11.2] (15/215) | 81.9% [76.2–86.4] (176/215) | 1.8% [0.6–5.2] (3/166) | 85.3% [82.5–87.7] (626/734) |
| 0.4 | 6.5% [3.9–10.6] (14/215) | 83.3% [77.7–87.7] (179/215) | 2.4% [0.9–6.0] (4/166) | 85.4% [82.7–87.8] (627/734) |
| 0.3 | 5.1% [2.9–8.9] (11/215) | 84.7% [79.2–88.9] (182/215) | 3.6% [1.7–7.7] (6/166) | 85.4% [82.7–87.8] (627/734) |
| 0.2 | 4.2% [2.2–7.8] (9/215) | 86.0% [80.8–90.0] (185/215) | 4.2% [2.1–8.4] (7/166) | 85.4% [82.7–87.8] (627/734) |
| 0.1 | 3.7% [1.9–7.2] (8/215) | 90.2% [85.5–93.5] (194/215) | 5.4% [2.9–10.0] (9/166) | 85.6% [82.8–87.9] (628/734) |
| 0.05 | 3.3% [1.6–6.6] (7/215) | 92.1% [87.7–95.0] (198/215) | 7.2% [4.2–12.2] (12/166) | 84.5% [81.7–86.9] (620/734) |

## Leakage check

0 of 734 images share an iNaturalist observation with a training image (status: **ok**).

Critical errors (prohibited not predicted prohibited) are listed in `hierarchical_val_critical_errors.csv`.
