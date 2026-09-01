'use client';

import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
  /** Short label for what failed, e.g. "the conversation" or "this preview". */
  label?: string;
  /** Rendered instead of the default card when provided. */
  fallback?: ReactNode;
  /** Called after an error is caught (for logging / telemetry). */
  onError?: (error: Error, info: ErrorInfo) => void;
}

interface State {
  error: Error | null;
}

/**
 * Catches render errors in a subtree so one malformed message, preview payload,
 * or third-party hiccup can't blank the whole app. Reset by re-mounting (key
 * change) or the "Try again" button.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('ErrorBoundary caught:', error, info.componentStack);
    this.props.onError?.(error, info);
  }

  private reset = () => this.setState({ error: null });

  render() {
    if (!this.state.error) return this.props.children;
    if (this.props.fallback) return this.props.fallback;

    const what = this.props.label ?? 'this section';
    return (
      <div
        role="alert"
        className="mx-auto my-6 max-w-md rounded-xl border border-danger-border bg-danger-surface p-5 text-center"
      >
        <p className="text-sm font-semibold text-ink">
          Something went wrong showing {what}.
        </p>
        <p className="mt-1 text-sm text-ink-muted">
          The rest of the page still works. You can retry this part below.
        </p>
        <button
          type="button"
          onClick={this.reset}
          className="mt-4 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-fg transition-colors hover:bg-primary-hover"
        >
          Try again
        </button>
      </div>
    );
  }
}
