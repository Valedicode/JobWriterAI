'use client';

import { GateDemo } from './GateDemo';

interface LandingHeroProps {
  /** Start the default (job tailoring) flow. */
  onStart: () => void;
}

export const LandingHero = ({ onStart }: LandingHeroProps) => {
  return (
    <section className="mx-auto grid w-full max-w-6xl gap-14 px-6 pb-16 pt-10 lg:grid-cols-[1.05fr_0.95fr] lg:items-start lg:gap-12 lg:pb-24 lg:pt-16">
      <div className="hero-rise">
        <h1 className="max-w-[15ch] font-serif text-[2.6rem] font-normal leading-[1.05] tracking-[-0.005em] text-ink sm:text-5xl lg:text-6xl">
          You see every change{' '}
          <span className="text-primary-text">before it&rsquo;s yours.</span>
        </h1>

        <p className="mt-6 max-w-[46ch] text-lede text-ink-muted">
          JobWriterAI scores your fit, proposes the edits, and stops at each
          step. Nothing is written into your resume until you approve it.
        </p>

        <div className="mt-8">
          <button
            type="button"
            onClick={onStart}
            className="group inline-flex items-center gap-2 rounded-full bg-primary px-6 py-3 text-sm font-semibold text-primary-fg shadow-[0_12px_28px_-10px] shadow-primary/50 transition-transform duration-200 hover:-translate-y-0.5 active:scale-[0.98]"
          >
            Start with your CV
            <svg
              className="h-4 w-4 transition-transform duration-200 group-hover:translate-x-0.5"
              viewBox="0 0 20 20"
              fill="none"
              aria-hidden="true"
            >
              <path
                d="M4 10h11M11 5l5 5-5 5"
                stroke="currentColor"
                strokeWidth="1.75"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </div>
      </div>

      <div className="hero-rise-delayed relative mx-auto w-full max-w-sm lg:mx-0 lg:justify-self-end">
        <GateDemo />
      </div>
    </section>
  );
};
