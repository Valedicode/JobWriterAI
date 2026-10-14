/**
 * Hook for downloading generated files
 */

import { useState, useCallback } from 'react';
import { downloadFile, triggerFileDownload, getErrorMessage } from '@/lib/api';

interface UseFileDownloadReturn {
  isDownloading: boolean;
  downloadError: string | null;
  downloadGeneratedFile: (downloadUrl: string, filename: string) => Promise<void>;
  clearDownloadError: () => void;
}

export const useFileDownload = (): UseFileDownloadReturn => {
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const downloadGeneratedFile = useCallback(async (downloadUrl: string, filename: string) => {
    setIsDownloading(true);
    setDownloadError(null);

    try {
      const blob = await downloadFile(downloadUrl);
      triggerFileDownload(blob, filename);
    } catch (error) {
      const errorMessage = getErrorMessage(error);
      setDownloadError(errorMessage);
      console.error('Download error:', error);
    } finally {
      setIsDownloading(false);
    }
  }, []);

  const clearDownloadError = useCallback(() => {
    setDownloadError(null);
  }, []);

  return {
    isDownloading,
    downloadError,
    downloadGeneratedFile,
    clearDownloadError,
  };
};



