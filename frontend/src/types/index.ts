// ============================================
// Core Types
// ============================================

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  generatedFiles?: DownloadableFile[] | null;
}

// ============================================
// CV Agent Types (matching backend schemas)
// ============================================

export interface EducationEntry {
  institution?: string;
  degree?: string;
  location?: string;
  dates?: string;
  grade?: string;
  details?: string[];
}

export interface ExperienceEntry {
  position?: string;
  company?: string;
  location?: string;
  duration?: string;
  responsibilities?: string[];
}

export interface ProjectEntry {
  name?: string;
  description?: string;
  technologies?: string[];
  outcomes?: string[];
}

export interface LeadershipEntry {
  role?: string;
  organization?: string;
  location?: string;
  dates?: string;
  description?: string;
  highlights?: string[];
}

export interface LanguageEntry {
  language: string;
  fluency?: string;
}

export interface ResumeInfo {
  name: string;
  email: string;
  phone: string;
  location?: string;
  github_url?: string;
  linkedin_url?: string;
  portfolio_url?: string;
  summary?: string;
  skills: string[];
  education: EducationEntry[];
  experience: ExperienceEntry[];
  projects?: ProjectEntry[];
  leadership_activities?: LeadershipEntry[];
  languages?: LanguageEntry[];
}

export interface CVExtractionResponse {
  success: boolean;
  cv_data?: ResumeInfo;
  needs_clarification: boolean;
  questions?: string[];
  message: string;
}

export interface CVClarificationRequest {
  cv_data: Record<string, any>;
  clarifications: string;
}

export interface CVClarificationResponse {
  success: boolean;
  updated_cv_data?: ResumeInfo;
  message: string;
}

// ============================================
// Job Agent Types
// ============================================

export interface JobRequirements {
  job_title: string;
  /** The role in one to three words, e.g. "Software Developer"; may be empty. */
  short_title?: string;
  job_level: string;
  required_skills: string[];
  required_skill_groups?: string[][];
  preferred_skills: string[];
  learnable_skills?: string[];
  required_languages?: string[];
  education_requirements?: string[];
  years_experience?: number;
  employment_type: string;
  location: string;
  work_mode?: 'onsite' | 'hybrid' | 'remote' | null;
  location_constraint?: string | null;
  hours_per_week?: string | null;
  start_date?: string | null;
  work_permit?: string | null;
  posting_language?: string | null;
  country?: string | null;
  responsibilities: string[];
  qualifications: string[];
  key_requirements: string[];
  recipient_name?: string;
}

export interface JobExtractionResponse {
  success: boolean;
  job_data?: JobRequirements;
  message: string;
}

export interface CompanyInfo {
  company_name: string;
  industry: string;
  company_size?: string;
  mission_statement?: string;
  core_values: string[];
  recent_news: string[];
  company_culture: string;
  products_services: string[];
}

export interface CompanyResearchResponse {
  success: boolean;
  company_data?: CompanyInfo;
  message: string;
}

// ============================================
// Writer Agent Types
// ============================================

export interface CVTailoringPlan {
  matching_experiences: string[];
  matching_skills: string[];
  relevant_projects: string[];
  keywords_to_incorporate: string[];
  reordering_suggestions: string;
  emphasis_points: string[];
  reasoning: string;
}

export interface CVJobAlignmentResponse {
  success: boolean;
  tailoring_plan?: CVTailoringPlan;
  message: string;
}

export interface GenerateTailoredCVResponse {
  success: boolean;
  pdf_path?: string;
  latex_preview?: string;
  message: string;
}

export interface CoverLetterContent {
  opening_paragraph: string;
  body_paragraph_1: string;
  body_paragraph_2: string;
  body_paragraph_3?: string;
  closing_paragraph: string;
}

export interface GenerateCoverLetterResponse {
  success: boolean;
  pdf_path?: string;
  content?: CoverLetterContent;
  message: string;
}

// ============================================
// Error Types
// ============================================

export interface ErrorResponse {
  success: boolean;
  error: string;
  detail?: string;
}

// ============================================
// API Request Types
// ============================================

export interface JobURLRequest {
  urls: string[];
}

export interface JobTextRequest {
  job_text: string;
}

export interface CVJobAlignmentRequest {
  cv_data: Record<string, any>;
  job_data: Record<string, any>;
}

export interface GenerateTailoredCVRequest {
  cv_data: Record<string, any>;
  tailoring_plan: Record<string, any>;
  output_filename: string;
}

export interface GenerateCoverLetterRequest {
  cv_data: Record<string, any>;
  job_data: Record<string, any>;
  company_data?: Record<string, any>;
  output_filename: string;
  recipient_info?: string;
}

export interface CompanyResearchRequest {
  company_name: string;
}

// ============================================
// Shared downloadable file metadata
// ============================================

export interface DownloadableFile {
  filename: string;
  file_type: string;
  download_url: string;
}

// ============================================
// Audio Transcription Types
// ============================================

export type TranscriptionModel = 
  | 'whisper-1' 
  | 'gpt-4o-transcribe' 
  | 'gpt-4o-mini-transcribe' 
  | 'gpt-4o-transcribe-diarize';

export type TranscriptionResponseFormat = 
  | 'json' 
  | 'text' 
  | 'srt' 
  | 'verbose_json' 
  | 'vtt' 
  | 'diarized_json';

export interface TranscriptionRequest {
  file: File;
  model?: TranscriptionModel;
  response_format?: TranscriptionResponseFormat;
  language?: string;
  prompt?: string;
  temperature?: number;
  timestamp_granularities?: ('word' | 'segment')[];
  chunking_strategy?: 'auto';
}

export interface TranscriptionSegment {
  id: number;
  seek: number;
  start: number;
  end: number;
  text: string;
  tokens: number[];
  temperature: number;
  avg_logprob: number;
  compression_ratio: number;
  no_speech_prob: number;
  speaker?: string; // For diarized_json format
}

export interface TranscriptionWord {
  word: string;
  start: number;
  end: number;
}

export interface TranscriptionResponse {
  success: boolean;
  text?: string | null;
  segments?: TranscriptionSegment[] | null;
  words?: TranscriptionWord[] | null;
  message: string;
}

export interface TranslationRequest {
  file: File;
  model?: 'whisper-1';
  response_format?: 'json' | 'text' | 'srt' | 'verbose_json' | 'vtt';
  prompt?: string;
  temperature?: number;
}

export interface TranslationResponse {
  success: boolean;
  text?: string | null;
  segments?: TranscriptionSegment[] | null;
  words?: TranscriptionWord[] | null;
  message: string;
}

// ============================================
// Orchestrator Types (LangGraph backend)
// ============================================

export type OrchestratorFlow = 'job_tailoring' | 'cv_review' | 'discovery';

export type OrchestratorGateKind = 'approval' | 'choice' | 'input';

export type OrchestratorGateAction = 'approve' | 'reject' | 'edit' | 'choose';

export interface OrchestratorGatePayload {
  step: string;
  kind: OrchestratorGateKind;
  narration: string;
  preview: Record<string, unknown>;
  allowed_actions: OrchestratorGateAction[];
  choices?: string[] | null;
}

/** Selection: include (tailor it), exclude (keep as is), leave_out (drop from the
 * tailored CV). Rewrite: accept, keep_original, custom. */
export type BulletDecisionChoice =
  | 'accept'
  | 'keep_original'
  | 'custom'
  | 'include'
  | 'exclude'
  | 'leave_out';

export interface BulletDecision {
  choice: BulletDecisionChoice;
  text?: string | null;
}

export interface OrchestratorGateResolution {
  action: OrchestratorGateAction;
  /** Step of the gate being answered; a stale answer (step mismatch) is refused with 409. */
  step?: string;
  feedback?: string;
  choice?: string;
  /** Per-bullet decisions keyed by bullet_id, for approve_selection / approve_rewrite gates. */
  decisions?: Record<string, BulletDecision>;
  /** Answers keyed by question id, for gates whose preview lists `questions` (confirm_eligibility). */
  answers?: Record<string, string>;
}

/** One question in a confirm_eligibility gate preview. */
export interface EligibilityQuestion {
  id: string;
  requirement: string;
  kind: 'language' | 'education';
  prompt: string;
  language?: string;
  options?: string[];
  placeholder?: string;
  matched_with?: string;
}

export interface OrchestratorStartRequest {
  flow: OrchestratorFlow;
  cv_data: ResumeInfo;
  job_data?: JobRequirements;
  company_data?: CompanyInfo;
}

export interface OrchestratorMessageRequest {
  session_id: string;
  kind: 'chat' | 'gate_resolution';
  text?: string;
  resolution?: OrchestratorGateResolution;
}

export interface OrchestratorResponse {
  success: boolean;
  session_id: string;
  narration: string;
  /** Every assistant narration produced since the previous response, in order. */
  narrations?: string[];
  pending_gate: OrchestratorGatePayload | null;
  generated_files: DownloadableFile[];
  done: boolean;
  message: string;
  /** Id of the graph run this request started, for finding its trace. */
  run_id?: string | null;
}

export interface OrchestratorStateResponse {
  success: boolean;
  session_id: string;
  flow?: string | null;
  pending_gate: OrchestratorGatePayload | null;
  generated_files: DownloadableFile[];
  last_narration: string;
  done: boolean;
}

// ============================================
// Documents API Types (the finished CV and cover letter in the editor)
// ============================================

/**
 * A piece of printed CV text and the CV fields it came from (empty for the
 * template's own words). `value` is the field's raw value where the printed
 * text is formatted (dates); `kind` 'list' marks a comma-separated list field.
 */
export interface CvSegment {
  text: string;
  src: string[];
  value?: string;
  kind?: 'list';
}

export interface CvRow {
  left: string;
  right: string;
  bold: boolean;
  left_segs: CvSegment[];
  right_segs: CvSegment[];
}

/** An entry of an "entries" section (rows plus bullets) or an "itemize" one (label plus text). */
export interface CvEntry {
  rows?: CvRow[];
  bullets?: string[];
  bullet_segs?: CvSegment[][];
  /** The bullet list's own path, for adding and removing bullets. */
  bullets_src?: string[];
  label?: string;
  text?: string;
  label_segs?: CvSegment[];
  text_segs?: CvSegment[];
}

export interface CvSection {
  type: string;
  title: string;
  style: 'entries' | 'itemize';
  entries: CvEntry[];
}

export interface CvHeader {
  name: string;
  label: string;
  summary: string;
  contact_parts: string[];
  name_segs: CvSegment[];
  label_segs: CvSegment[];
  summary_segs: CvSegment[];
  contact_segs: CvSegment[][];
}

export interface CvEdit {
  op: 'set' | 'insert' | 'delete';
  path: string;
  value?: string | string[];
  index?: number;
}

export interface CvEditRequest {
  revision: number;
  edits: CvEdit[];
}

export interface CoverLetterEditRequest {
  revision: number;
  fields: Record<string, string>;
}

export interface DocumentResetRequest {
  revision: number;
}

interface DocumentViewBase {
  revision: number;
  /** The revision the PDF shows; differs from `revision` when generating it failed. */
  rendered_revision: number | null;
  /** True when the document differs from the version the flow generated. */
  edited: boolean;
  filename: string;
  download_url: string | null;
  language: string;
}

export interface CvDocumentView extends DocumentViewBase {
  header: CvHeader;
  sections: CvSection[];
}

export interface CoverLetterParagraph {
  field: string;
  text: string;
}

/** Everything the cover letter PDF prints. */
export interface CoverLetterPrint {
  name: string;
  phone: string;
  email: string;
  target_job: string;
  link_1_url: string;
  link_1_label: string;
  link_2_url: string;
  link_2_label: string;
  document_title: string;
  date_label: string;
  current_date: string;
  betreff?: string;
  salutation: string;
  paragraph_fields: CoverLetterParagraph[];
  sign_off: string;
}

export interface CoverLetterView extends DocumentViewBase {
  /** The editable values: paragraph fields, recipient, sign_off, and betreff for a German letter. */
  fields: Record<string, string>;
  letter: CoverLetterPrint;
}

export interface DocumentsResponse {
  success: boolean;
  session_id: string;
  cv: CvDocumentView | null;
  cover_letter: CoverLetterView | null;
  /** False while a proposed change waits for confirmation in the chat. */
  editable: boolean;
}

export interface DocumentSaveResponse {
  session: OrchestratorResponse;
  documents: DocumentsResponse;
}

// ============================================
// Application State Types
// ============================================

export interface ApplicationState {
  sessionId?: string;
  cvData?: ResumeInfo;
  jobData?: JobRequirements;
  companyData?: CompanyInfo;
  tailoringPlan?: CVTailoringPlan;
  generatedFiles: GeneratedFile[];
}

export interface GeneratedFile {
  id: string;
  filename: string;
  type: 'cv' | 'cover_letter';
  path: string;
  timestamp: Date;
}
