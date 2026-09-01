'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import { translateAudio, getErrorMessage } from '@/lib/api';
import type { TranslationRequest } from '@/types';

interface ChatInputProps {
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
  inputText: string;
  isLoading: boolean;
  onInputChange: (value: string) => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  onSendMessage: () => void;
}

export const ChatInput = ({
  textareaRef,
  inputText,
  isLoading,
  onInputChange,
  onKeyDown,
  onSendMessage,
}: ChatInputProps) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      cleanupAudioVisualization();
    };
  }, []);

  // Setup audio visualization
  const setupAudioVisualization = useCallback((stream: MediaStream) => {
    try {
      audioContextRef.current = new AudioContext();
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      analyserRef.current.smoothingTimeConstant = 0.3; // Lower for more responsive
      
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      
      const bufferLength = analyserRef.current.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      
      const updateAudioLevel = () => {
        if (analyserRef.current && isRecording) {
          analyserRef.current.getByteFrequencyData(dataArray);
          // Calculate average volume
          const average = dataArray.reduce((a, b) => a + b) / bufferLength;
          // Normalize to 0-1 range, with better sensitivity
          const normalized = Math.min(average / 100, 1);
          setAudioLevel(normalized);
          
          // Continue animation loop
          if (isRecording) {
            animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
          }
        }
      };
      
      // Start the animation loop
      updateAudioLevel();
    } catch (err) {
      console.error('Error setting up audio visualization:', err);
    }
  }, [isRecording]);

  // Cleanup audio visualization
  const cleanupAudioVisualization = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    setAudioLevel(0);
  }, []);

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      setupAudioVisualization(stream);

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      });

      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
        cleanupAudioVisualization();
        setIsRecording(false);
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
        setRecordingTime(0);

        // Auto-process the audio
        await processAudio(blob);
      };

      mediaRecorder.start();
      setIsRecording(true);

      // Start timer
      let time = 0;
      timerRef.current = setInterval(() => {
        time += 1;
        setRecordingTime(time);
      }, 1000);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to start recording';
      setError(`Microphone access denied: ${errorMessage}`);
      console.error('Recording error:', err);
    }
  }, [setupAudioVisualization, cleanupAudioVisualization]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
    }
  }, [isRecording]);

  const processAudio = async (blob: Blob) => {
    setIsProcessing(true);
    setError(null);

    try {
      const file = new File([blob], 'audio.webm', { type: blob.type });

      const request: TranslationRequest = {
        file,
        model: 'whisper-1',
        response_format: 'text',
      };

      const response = await translateAudio(request);

      if (response.success && response.text) {
        // Insert into text input (trim to remove trailing newlines)
        const cleanedText = response.text.trimEnd();
        onInputChange(cleanedText);
        // Focus textarea and move cursor to end
        setTimeout(() => {
          if (textareaRef.current) {
            textareaRef.current.focus();
            const length = cleanedText.length;
            textareaRef.current.setSelectionRange(length, length);
          }
        }, 100);
      } else {
        throw new Error(response.message || 'Translation failed');
      }
    } catch (err) {
      const errorMessage = getErrorMessage(err);
      setError(errorMessage);
      console.error('Translation error:', errorMessage);
    } finally {
      setIsProcessing(false);
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Audio Visualization Component - Compact circles around button
  const AudioVisualization = ({ level }: { level: number }) => {
    const circles = [0, 1, 2];
    
    return (
      <>
        {circles.map((index) => {
          // Compact animation around button
          const baseScale = 1 + index * 0.12;
          const audioScale = level * (0.3 + index * 0.15);
          const finalScale = baseScale + audioScale;
          
          // Opacity that responds to audio
          const baseOpacity = 0.12 - index * 0.03;
          const audioOpacity = level * (0.25 - index * 0.06);
          const finalOpacity = Math.max(0.05, Math.min(0.3, baseOpacity + audioOpacity));
          
          return (
            <div
              key={index}
              className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary"
              style={{
                width: `${24 + index * 8}px`,
                height: `${24 + index * 8}px`,
                transform: `scale(${finalScale})`,
                opacity: finalOpacity,
                transition: 'transform 0.1s ease-out, opacity 0.1s ease-out',
              }}
            />
          );
        })}
      </>
    );
  };

  return (
    <div className="border-t border-border bg-surface-sunken p-4">
      <div className="mx-auto max-w-3xl">

        {/* Processing Indicator */}
        {isProcessing && (
          <div role="status" className="mb-4 rounded-xl border border-border-strong bg-surface p-4">
            <div className="flex items-center gap-3">
              <svg className="h-5 w-5 animate-spin text-primary" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span className="text-sm font-medium text-ink">
                Translating to English…
              </span>
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div role="alert" className="mb-4 rounded-xl border border-danger-border bg-danger-surface p-3 text-sm text-danger-text">
            {error}
          </div>
        )}

        <div className="relative flex items-center rounded-2xl border border-border-strong bg-surface px-4 py-3 shadow-sm transition-colors focus-within:border-primary focus-within:shadow-md">
          {/* Text Input */}
          <textarea
            ref={textareaRef}
            value={inputText}
            onChange={(e) => onInputChange(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask any question or use voice input"
            disabled={isRecording || isProcessing}
            maxLength={4000}
            className="flex-1 resize-none border-0 bg-transparent text-base text-ink placeholder:text-ink-faint focus:outline-none focus:ring-0 disabled:opacity-50"
            rows={1}
            style={{ minHeight: '32px', maxHeight: '200px' }}
          />
          
          {/* Right Side Icons */}
          <div className="ml-3 flex items-center gap-2">
            {/* Recording Timer (shown inline when recording) */}
            {isRecording && (
              <div className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-danger-surface">
                <div className="h-1.5 w-1.5 rounded-full bg-danger animate-pulse"></div>
                <span className="font-mono text-meta font-medium tabular-nums text-danger-text">
                  {formatTime(recordingTime)}
                </span>
              </div>
            )}

            {/* Microphone/Stop Button with Animation */}
            <button
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isProcessing}
              className={`relative flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-lg transition-colors ${
                isRecording
                  ? 'bg-danger text-danger-fg hover:brightness-95'
                  : isProcessing
                  ? 'opacity-50 cursor-not-allowed text-ink-faint'
                  : 'text-ink-faint hover:bg-surface-sunken hover:text-ink'
              }`}
              aria-label={isRecording ? "Stop recording" : "Voice input"}
              title={isRecording ? "Click to stop and transcribe" : "Click to start voice recording"}
            >
              {/* Animated Circles (only when recording) */}
              {isRecording && (
                <div className="absolute inset-0 flex items-center justify-center overflow-visible pointer-events-none">
                  <AudioVisualization level={audioLevel} />
                </div>
              )}
              
              {/* Icon */}
              <div className="relative z-10">
                {isRecording ? (
                  <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
                    <rect x="6" y="6" width="12" height="12" rx="2" />
                  </svg>
                ) : (
                  <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                  </svg>
                )}
              </div>
            </button>
            
            {/* Send Button */}
            <button
              onClick={onSendMessage}
              disabled={!inputText.trim() || isLoading || isRecording || isProcessing}
              className={`flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full bg-primary text-primary-fg transition-transform hover:bg-primary-hover disabled:cursor-not-allowed disabled:opacity-50 ${
                inputText.trim() && !isLoading && !isRecording && !isProcessing ? 'hover:scale-105' : ''
              }`}
              aria-label="Send message"
            >
              {isLoading ? (
                <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              ) : (
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              )}
            </button>
          </div>
        </div>
        <p className="mt-2 text-meta text-ink-muted">
          Powered by AI. Your data is processed securely and privately.
        </p>
      </div>
    </div>
  );
};

