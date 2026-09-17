import { ThemeToggle } from './ThemeToggle';

interface HeaderProps {
  isDark: boolean;
  toggleTheme: () => void;
}

export const Header = ({ isDark, toggleTheme }: HeaderProps) => {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-canvas/85 backdrop-blur-md">
      <div className="container mx-auto flex h-16 items-center justify-between px-6">
        <a
          href="/"
          className="font-serif text-[1.4rem] font-normal tracking-[-0.005em] text-ink"
        >
          JobWriterAI
        </a>

        <div className="flex items-center gap-3">
          <span className="hidden rounded-full border border-border px-2.5 py-1 text-meta font-medium uppercase tracking-[0.16em] text-ink-faint sm:inline">
            Pre-launch
          </span>
          <ThemeToggle isDark={isDark} toggleTheme={toggleTheme} />
        </div>
      </div>
    </header>
  );
};
