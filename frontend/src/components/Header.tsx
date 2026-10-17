import { ThemeToggle } from './ThemeToggle';

interface HeaderProps {
  isDark: boolean;
  toggleTheme: () => void;
  /** Chat screen: a slim full-width bar with the stepper in the middle. */
  center?: React.ReactNode;
  /** Shown before the theme toggle, e.g. the job being tailored for. */
  trailing?: React.ReactNode;
}

export const Header = ({ isDark, toggleTheme, center, trailing }: HeaderProps) => {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-canvas/85 backdrop-blur-md">
      <div
        className={`flex items-center justify-between gap-4 px-6 ${
          center ? 'h-14' : 'container mx-auto h-16'
        }`}
      >
        <a
          href="/"
          className="font-serif text-[1.4rem] font-normal tracking-[-0.005em] text-ink"
        >
          JobWriterAI
        </a>

        {center && <div className="hidden min-w-0 flex-1 justify-start md:flex">{center}</div>}

        <div className="flex items-center gap-3">
          {trailing}
          <span className="hidden rounded-full border border-border px-2.5 py-1 text-meta font-medium uppercase tracking-[0.16em] text-ink-faint sm:inline">
            Pre-launch
          </span>
          <ThemeToggle isDark={isDark} toggleTheme={toggleTheme} />
        </div>
      </div>
    </header>
  );
};
