interface QuickSuggestionsProps {
  onSuggestionClick: (text: string) => void;
}

export const QuickSuggestions = ({ onSuggestionClick }: QuickSuggestionsProps) => {
  const suggestions = [
    'How can I improve my resume?',
    'Match my resume to a job description',
    'What skills should I highlight?',
  ];

  return (
    <div className="mt-8 hidden grid-cols-1 gap-3 sm:grid sm:grid-cols-2 lg:grid-cols-3">
      {suggestions.map((suggestion, index) => (
        <button
          key={index}
          onClick={() => onSuggestionClick(suggestion)}
          className="group rounded-lg border border-border bg-surface-sunken px-4 py-3 text-left text-sm text-ink transition-colors hover:border-primary hover:bg-primary-surface hover:text-primary-text"
        >
          {suggestion}
        </button>
      ))}
    </div>
  );
};
