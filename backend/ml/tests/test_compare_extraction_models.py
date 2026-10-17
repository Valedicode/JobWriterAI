"""The comparison's metrics measure what they claim (no API calls)."""
from ml.labeling.compare_extraction_models import quality, summarize

SOURCE = """Ada Lovelace
Experience
Analytical Engines Ltd.
Designed the first algorithm for the analytical engine in collaboration.
Wrote extensive notes on the Bernoulli numbers for the general public.
"""
CV = {"name": "Ada Lovelace", "experience": [{"company": "Analytical Engines Ltd.", "responsibilities": [
    "Designed the first algorithm for the analytical engine in collaboration.",
    "Wrote extensive notes on the Bernoulli numbers for the general public."]}]}


def test_a_faithful_extraction_is_verbatim_and_covers_the_source():
    q = quality(CV, SOURCE)
    assert (q["not_verbatim"], q["bullets"]) == (0.0, 2)
    assert q["coverage"] >= 0.8  # only the "Experience" heading isn't copied


def test_reworded_lines_and_dropped_content_are_counted():
    reworded = {"experience": [{"responsibilities": [
        "Pioneered algorithm design for an early mechanical computing machine."]}]}
    q = quality(reworded, SOURCE)
    assert q["not_verbatim"] == 1.0
    assert q["coverage"] < 0.2


def test_the_summary_compares_runs_and_counts_failures():
    ok = {"model": "m", "status": "ok", "cv": CV, "slips": 1, "not_verbatim": 0.0, "coverage": 1.0,
          "latency_s": 2.0, "cost_usd": 0.001}
    moved = {**ok, "resume": "r1", "run": 2, "slips": 0,
             "cv": {"leadership_activities": [{"highlights": CV["experience"][0]["responsibilities"]}]}}
    records = [{**ok, "resume": "r1", "run": 1}, moved, {"model": "m", "resume": "r2", "run": 1, "status": "error",
                                                         "error": "TimeoutError: slow", "latency_s": 120.0}]
    s = summarize(records, ["m"], extractions_planned=1000)["m"]
    assert s["failed"] == 1 and s["identical_runs"] == 0.0
    assert s["same_bullet_placement"] == 0.0  # the same bullets, under a different entry
    assert s["resumes_with_slips"] == 0.5
    assert s["cost_full_extraction_usd"] == 1.0


def test_glued_words_in_a_scraped_source_still_match_a_faithful_copy():
    source = "Professional ProfileExpert in functional testing.6-Year Record of Proven ResultsSenior software QA tester"
    cv = {"summary": "Expert in functional testing. 6-Year Record of Proven Results",
          "experience": [{"responsibilities": ["Senior software QA tester with system experience"]}]}
    q = quality(cv, source)
    assert q["not_verbatim"] == 0.5  # the bullet adds words the source doesn't have
    assert q["coverage"] > 0.6  # all but the glued-on heading
