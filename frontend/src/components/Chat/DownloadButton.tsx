import { useState } from 'react';
import { downloadFile, triggerFileDownload, getErrorMessage } from '@/lib/api';
import type { DownloadableFile } from '@/types';

interface DownloadButtonProps {
  file: DownloadableFile;
}

export const DownloadButton = ({ file }: DownloadButtonProps) => {
  const [isDownloading, setIsDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDownload = async () => {
    setIsDownloading(true);
    setError(null);

    try {
      const blob = await downloadFile(file.filename);
      triggerFileDownload(blob, file.filename);
    } catch (err) {
      const errorMessage = getErrorMessage(err);
      setError(errorMessage);
      console.error('Download error:', err);
    } finally {
      setIsDownloading(false);
    }
  };

  const getFileIcon = () => {
    if (file.file_type === 'cover_letter') {
      return (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
        </svg>
      );
    }
    return (
      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    );
  };

  const getFileLabel = () => {
    const isCoverLetter =
      file.file_type === 'cover_letter' || file.filename.toLowerCase().includes('cover');

    if (isCoverLetter) {
      return 'Cover Letter (PDF)';
    }
    return 'CV/Resume (PDF)';
  };

  return (
    <div className="mt-3 inline-block">
      <button
        onClick={handleDownload}
        disabled={isDownloading}
        className="group relative inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-fg shadow-sm transition-colors hover:bg-primary-hover hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-surface disabled:cursor-not-allowed disabled:opacity-50"
      >
        {isDownloading ? (
          <>
            <svg className="h-5 w-5 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span>Downloading...</span>
          </>
        ) : (
          <>
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span>Download {getFileLabel()}</span>
          </>
        )}
      </button>

      {error && (
        <div role="alert" className="mt-2 text-meta text-danger-text">
          {error}
        </div>
      )}

      <div className="mt-1.5 flex items-center gap-1.5 text-meta text-ink-muted">
        {getFileIcon()}
        <span className="truncate max-w-[200px]">{file.filename}</span>
      </div>
    </div>
  );
};
