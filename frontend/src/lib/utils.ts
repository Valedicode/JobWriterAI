const MAX_SIZE_BYTES = 10 * 1024 * 1024; // 10MB

export const validateFile = (file: File): string | null => {
  const isPdfType = file.type === 'application/pdf';
  const isPdfExt = /\.pdf$/i.test(file.name);

  // Some OSes / drag sources don't set a MIME type; fall back to the extension.
  if (!isPdfType && !isPdfExt) {
    return 'Please upload a PDF file.';
  }

  if (file.size === 0) {
    return 'That file looks empty. Please choose a different PDF.';
  }

  if (file.size > MAX_SIZE_BYTES) {
    return 'That file is over 10MB. Please upload a smaller PDF.';
  }

  return null;
};

export const formatFileSize = (bytes: number): string => {
  if (!Number.isFinite(bytes) || bytes < 0) return '—';
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};
