import { Message, DownloadableFile } from '@/types';
import { DownloadButton } from './DownloadButton';

interface ChatMessageProps {
  message: Message;
  generatedFiles?: DownloadableFile[] | null;
  isFadingOut?: boolean;
}

export const ChatMessage = ({ message, generatedFiles, isFadingOut = false }: ChatMessageProps) => {
  // Simple markdown-like formatting for plain text
  // Converts ## and ### headers, **text** to bold, *text* to italic, [text](url) to links, and preserves line breaks
  const formatText = (text: string) => {
    // Split by line breaks to preserve structure
    return text.split('\n').map((line, lineIndex) => {
      // Check if line is a ### header (h3)
      const h3Match = line.match(/^###\s+(.+)$/);
      if (h3Match) {
        return (
          <h3 key={lineIndex} className="mt-4 mb-2 text-base font-semibold text-ink first:mt-0">
            {h3Match[1]}
          </h3>
        );
      }
      
      // Check if line is a ## header (h2)
      const h2Match = line.match(/^##\s+(.+)$/);
      if (h2Match) {
        return (
          <h2 key={lineIndex} className="mt-6 mb-3 text-lg font-bold text-ink first:mt-0">
            {h2Match[1]}
          </h2>
        );
      }
      
      // Process formatting: links first, then bold/italic
      const parts: (string | React.ReactElement)[] = [];
      let lastIndex = 0;
      let key = 0;
      
      // Match markdown links: [text](url)
      const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
      const matches: Array<{ start: number; end: number; type: 'link' | 'bold' | 'italic'; text: string; url?: string }> = [];
      
      let match;
      // Find all link matches
      while ((match = linkRegex.exec(line)) !== null) {
        matches.push({ 
          start: match.index, 
          end: match.index + match[0].length, 
          type: 'link', 
          text: match[1], 
          url: match[2] 
        });
      }
      
      // Helper function to process bold/italic within a text segment
      const processBoldItalic = (textSegment: string, baseKey: number): (string | React.ReactElement)[] => {
        const segmentParts: (string | React.ReactElement)[] = [];
        let segmentLastIndex = 0;
        let segmentKey = baseKey;
        
        // Match **bold** and *italic* patterns
        const boldRegex = /\*\*([^*]+)\*\*/g;
        const italicRegex = /(?<!\*)\*([^*]+)\*(?!\*)/g;
        
        const formatMatches: Array<{ start: number; end: number; type: 'bold' | 'italic'; text: string }> = [];
        
        let formatMatch;
        while ((formatMatch = boldRegex.exec(textSegment)) !== null) {
          formatMatches.push({ 
            start: formatMatch.index, 
            end: formatMatch.index + formatMatch[0].length, 
            type: 'bold', 
            text: formatMatch[1] 
          });
        }
        while ((formatMatch = italicRegex.exec(textSegment)) !== null) {
          // Check if not part of a bold match
          const isPartOfBold = formatMatches.some(m => formatMatch!.index >= m.start && formatMatch!.index < m.end);
          if (!isPartOfBold) {
            formatMatches.push({ 
              start: formatMatch.index, 
              end: formatMatch.index + formatMatch[0].length, 
              type: 'italic', 
              text: formatMatch[1] 
            });
          }
        }
        
        // Sort matches by position
        formatMatches.sort((a, b) => a.start - b.start);
        
        // Build segment parts
        formatMatches.forEach((m) => {
          // Add text before match
          if (m.start > segmentLastIndex) {
            segmentParts.push(textSegment.substring(segmentLastIndex, m.start));
          }
          // Add formatted match
          if (m.type === 'bold') {
            segmentParts.push(<strong key={`${lineIndex}-format-${segmentKey++}`}>{m.text}</strong>);
          } else {
            segmentParts.push(<em key={`${lineIndex}-format-${segmentKey++}`}>{m.text}</em>);
          }
          segmentLastIndex = m.end;
        });
        
        // Add remaining text
        if (segmentLastIndex < textSegment.length) {
          segmentParts.push(textSegment.substring(segmentLastIndex));
        }
        
        return segmentParts.length > 0 ? segmentParts : [textSegment];
      };
      
      // Sort all matches by position
      matches.sort((a, b) => a.start - b.start);
      
      // Build parts array
      matches.forEach((m) => {
        // Add text before match
        if (m.start > lastIndex) {
          const beforeText = line.substring(lastIndex, m.start);
          const beforeParts = processBoldItalic(beforeText, key);
          parts.push(...beforeParts);
          key += beforeParts.filter(p => typeof p !== 'string').length;
        }
        // Add formatted match
        if (m.type === 'link') {
          const linkText = processBoldItalic(m.text, key);
          parts.push(
            <a
              key={`${lineIndex}-link-${key++}`}
              href={m.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary-text underline underline-offset-2 [overflow-wrap:anywhere] hover:opacity-80"
            >
              {linkText}
            </a>
          );
          key += linkText.filter(p => typeof p !== 'string').length;
        }
        lastIndex = m.end;
      });
      
      // Add remaining text
      if (lastIndex < line.length) {
        const remainingText = line.substring(lastIndex);
        const remainingParts = processBoldItalic(remainingText, key);
        parts.push(...remainingParts);
      }
      
      // If no matches, process the whole line for bold/italic
      if (parts.length === 0) {
        parts.push(...processBoldItalic(line, key));
      }
      
      return (
        <span key={lineIndex}>
          {parts}
          {lineIndex < text.split('\n').length - 1 && <br />}
        </span>
      );
    });
  };

  return (
    <div
      className={`flex gap-4 transition-opacity duration-500 ${
        message.role === 'user' ? 'justify-end' : 'justify-start'
      } ${isFadingOut ? 'opacity-0' : 'opacity-100 animate-fade-in'}`}
    >
      {message.role === 'assistant' && (
        <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-primary text-primary-fg">
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
      )}
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          message.role === 'user'
            ? 'bg-primary text-primary-fg'
            : 'bg-surface-sunken text-ink'
        }`}
      >
        <div className="text-body whitespace-pre-wrap [overflow-wrap:anywhere]">
          {formatText(message.content)}
        </div>
        {message.role === 'assistant' && generatedFiles && generatedFiles.length > 0 && (
          <div className="mt-3 pt-3 border-t border-border">
            <div className="flex flex-wrap gap-4">
              {generatedFiles.map((file, index) => (
                <DownloadButton key={index} file={file} />
              ))}
            </div>
          </div>
        )}
      </div>
      {message.role === 'user' && (
        <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-surface-sunken">
          <svg className="h-5 w-5 text-ink-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
          </svg>
        </div>
      )}
    </div>
  );
};


