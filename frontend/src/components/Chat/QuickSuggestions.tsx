interface QuickSuggestionsProps {
  onSuggestionClick: (text: string) => void;
}

const SUGGESTIONS: { label: string; prompt: string; icon: React.ReactNode }[] = [
  {
    label: 'Improve my resume',
    prompt: 'How can I improve my resume?',
    icon: (
      <path
        d="M9 12h6m-6 4h4m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h6l6 6v10a2 2 0 01-2 2z"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    ),
  },
  {
    label: 'Match me to a job',
    prompt: 'Match my resume to a job description.',
    icon: (
      <path
        d="M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    ),
  },
  {
    label: 'Which skills to highlight',
    prompt: 'What skills should I highlight?',
    icon: (
      <path
        d="M12 3l2.09 4.24L18.8 8l-3.4 3.31.8 4.69L12 13.8 7.8 16l.8-4.69L5.2 8l4.71-.76L12 3z"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    ),
  },
  {
    label: 'Check my formatting',
    prompt: 'Review my CV formatting and ATS readiness.',
    icon: (
      <path
        d="M4 7h16M4 12h10M4 17h7"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    ),
  },
];

export const QuickSuggestions = ({ onSuggestionClick }: QuickSuggestionsProps) => {
  return (
    <div className="mt-6 flex flex-wrap justify-center gap-2">
      {SUGGESTIONS.map((s) => (
        <button
          key={s.label}
          type="button"
          onClick={() => onSuggestionClick(s.prompt)}
          className="group inline-flex items-center gap-2 rounded-full border border-border bg-surface px-3.5 py-2 text-sm text-ink-muted transition-colors hover:border-border-strong hover:bg-surface-sunken hover:text-ink"
        >
          <svg
            className="h-4 w-4 text-primary-text"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            aria-hidden="true"
          >
            {s.icon}
          </svg>
          {s.label}
        </button>
      ))}
    </div>
  );
};
