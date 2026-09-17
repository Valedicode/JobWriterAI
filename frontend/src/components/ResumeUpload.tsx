import { formatFileSize } from '@/lib/utils';
import type { ResumeInfo } from '@/types';

interface ResumeUploadProps {
  uploadedFile: File | null;
  isDragging: boolean;
  uploadError: string | null;
  isUploading?: boolean;
  cvData?: ResumeInfo | null;
  needsClarification?: boolean;
  clarificationQuestions?: string[] | null;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
  onDragOver: (e: React.DragEvent) => void;
  onDragLeave: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent) => void;
  onFileInputChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onClickUpload: () => void;
  onRemoveFile: () => void;
}

export const ResumeUpload = ({
  uploadedFile,
  isDragging,
  uploadError,
  isUploading = false,
  cvData,
  needsClarification = false,
  clarificationQuestions,
  fileInputRef,
  onDragOver,
  onDragLeave,
  onDrop,
  onFileInputChange,
  onClickUpload,
  onRemoveFile,
}: ResumeUploadProps) => {
  return (
    <div>
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,application/pdf"
        onChange={onFileInputChange}
        aria-label="Resume PDF file"
        tabIndex={-1}
        className="sr-only"
      />

      {/* PDF Upload Area */}
      {!uploadedFile ? (
        <div
          role="button"
          tabIndex={0}
          aria-label="Upload your resume. Drag and drop a PDF here, or press Enter to browse."
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onClick={onClickUpload}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              onClickUpload();
            }
          }}
          className={`flex min-h-[200px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
            isDragging
              ? 'border-primary bg-primary-surface'
              : 'border-border-strong bg-surface-sunken hover:border-primary hover:bg-primary-surface/40'
          }`}
        >
          <div className="mb-4">
            <svg
              className={`mx-auto h-16 w-16 transition-colors ${
                isDragging ? 'text-primary-text' : 'text-ink-faint'
              }`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
          </div>
          <p className="mb-2 text-base font-medium text-ink">
            {isDragging ? 'Drop your PDF here' : 'Drag and drop your resume'}
          </p>
          <p className="mb-4 text-sm text-ink-muted">or</p>
          <button
            type="button"
            tabIndex={-1}
            onClick={(e) => {
              e.stopPropagation();
              onClickUpload();
            }}
            className="rounded-lg bg-primary px-6 py-2.5 text-sm font-medium text-primary-fg transition-colors hover:bg-primary-hover"
          >
            Choose File
          </button>
          <p className="mt-4 text-meta text-ink-muted">
            PDF only, max 10MB
          </p>
        </div>
      ) : (
        <div className="rounded-xl border border-border bg-surface-sunken p-4">
          <div className="flex items-start gap-3">
            <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-lg bg-primary-surface">
              <svg
                className="h-7 w-7 text-primary-text"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate font-medium text-ink">
                {uploadedFile.name}
              </p>
              <p className="mt-0.5 text-label tabular-nums text-ink-muted">
                {formatFileSize(uploadedFile.size)}
              </p>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onRemoveFile();
              }}
              className="flex-shrink-0 rounded-lg p-1.5 text-ink-faint transition-colors hover:bg-surface-sunken hover:text-ink"
              aria-label="Remove file"
            >
              <svg
                className="h-5 w-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
        </div>
      )}
      
      {/* Upload Progress */}
      {isUploading && (
        <div role="status" className="mt-4 rounded-lg border border-border-strong bg-surface-sunken p-4">
          <div className="flex items-center gap-3">
            <div className="h-5 w-5 animate-spin rounded-full border-2 border-primary border-t-transparent"></div>
            <p className="text-sm font-medium text-ink-muted">
              Processing your resume…
            </p>
          </div>
        </div>
      )}

      {/* Success Message */}
      {cvData && !isUploading && (
        <div role="status" className="mt-4 rounded-lg border border-success-border bg-success-surface p-4">
          <div className="flex items-start gap-3">
            <svg
              className="h-5 w-5 flex-shrink-0 text-success-text"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <div className="min-w-0 flex-1">
              <p className="font-medium text-success-text">
                Resume processed successfully!
              </p>
              {cvData.name?.trim() && (
                <p className="mt-0.5 text-sm text-success-text [overflow-wrap:anywhere]">
                  Found: {cvData.name}
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Clarification Needed */}
      {needsClarification && clarificationQuestions && !isUploading && (
        <div role="status" className="mt-4 rounded-lg border border-warning-border bg-warning-surface p-4">
          <div className="flex items-start gap-3">
            <svg
              className="h-5 w-5 flex-shrink-0 text-warning-text"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <div className="flex-1">
              <p className="font-medium text-warning-text">
                Additional information needed
              </p>
              <p className="mt-0.5 text-sm text-warning-text">
                Please answer questions in the chat to complete your profile.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Error Message */}
      {uploadError && (
        <div role="alert" className="mt-4 rounded-lg border border-danger-border bg-danger-surface p-4">
          <div className="flex items-start gap-3">
            <svg
              className="h-5 w-5 flex-shrink-0 text-danger-text"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <p className="text-sm font-medium text-danger-text">{uploadError}</p>
          </div>
        </div>
      )}
    </div>
  );
};

