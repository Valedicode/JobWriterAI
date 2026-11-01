# Audit rubric

How the author labels the 80 audit pairs (NLP_PROJECT_PLAN.md §6). The audit puts a noise estimate
on the dataset's labels, whose origin is unknown, and gives a second, small reference set.

Fixed before labelling starts; any later change is reported.

## Blind

- Don't look at the dataset's label, the teacher's score, or any model's output for these rows,
  before or during labelling. The files from `ml.dataprep.audit_sample` contain none of them.
- Label in the sheet's order. Don't go back and change a label after seeing later pairs, except to
  fix a typo.
- A second annotator labels the 30 rows marked `overlap`, alone, with this rubric and no other
  instruction.

## The question

**Would a recruiter screening for this job reasonably move this candidate forward?** Judge the
resume against this posting only, not the candidate's general quality.

## Procedure per pair

1. Read the job. Note its must-haves: the core skills or tools, the field, the seniority or years,
   and any hard requirement (licence, degree, clearance, language).
2. Read the resume against that list.
3. Pick the label, then a confidence.

## Labels

| Label | When |
|---|---|
| **Good Fit** | Meets the must-haves: right field, the core skills, about the required seniority. Gaps are minor or "nice to have" items. |
| **Potential Fit** | Meets some must-haves, with gaps that are learnable or partly covered: an adjacent field, related tools, one level below the seniority, most but not all core skills. A recruiter might move them forward. |
| **No Fit** | Misses the core of the job: a different field, the main skills absent, far off the seniority, or a hard requirement clearly unmet. |

Tie-breakers:

- Between two labels, take the lower one unless the resume shows evidence for the higher.
- A hard requirement the resume doesn't mention counts as unmet only if it is a real gate (a
  licence, a clearance). Otherwise treat it as a gap.
- Ignore layout, writing quality and the candidate's name, age, gender, nationality or photo.
  Location counts only when the posting makes it a requirement.

## Confidence

| | |
|---|---|
| 3 | Clear-cut |
| 2 | Leaning, but a recruiter could disagree |
| 1 | Guess: the posting or the resume is too vague |

## Notes column

Optional. Briefly say why for confidence 1, or anything odd about the pair (an empty or truncated
text, a duplicate, a resume in another language). Never copy resume text into the notes.

## Time

About 2–3 minutes per pair, so 80 pairs take about 3–4 hours. Split them into sittings of about
20 pairs.
