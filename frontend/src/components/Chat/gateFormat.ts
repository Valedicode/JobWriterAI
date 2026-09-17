/**
 * Small formatting helpers shared by the gate panels and the preview renderer.
 * The orchestrator sends machine step names (`cv_review_header`) and snake_case
 * preview keys (`weak_bullets_to_remove`); the UI shows people-readable text.
 */

const STEP_LABELS: Record<string, string> = {
  present_score: 'Compatibility',
  approve_selection: 'Bullet selection',
  approve_rewrite: 'Rewrite',
  cover_letter_language: 'Cover letter language',
  cover_letter_recipient: 'Cover letter recipient',
  approve_cover_letter: 'Cover letter',
  cv_review_header: 'Header',
  cv_review_education: 'Education',
  cv_review_experience: 'Experience',
  cv_review_leadership: 'Leadership & activities',
  cv_review_skills_projects: 'Skills & projects',
  cv_review_assessment: 'Overall assessment',
};

/** "cv_review_header" → "Header"; unknown steps get title-cased. */
export const humanizeStep = (step: string): string =>
  STEP_LABELS[step] ?? titleCase(step.replace(/^cv_review_/, ''));

/** "weak_bullets_to_remove" → "Weak bullets to remove". */
export const humanizeKey = (key: string): string => {
  const s = key.replace(/_/g, ' ').trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
};

function titleCase(s: string): string {
  return s
    .replace(/_/g, ' ')
    .split(' ')
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}
