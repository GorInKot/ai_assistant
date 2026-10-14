export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface ConversationSummary {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ChatSource {
  file_name: string;
  relative_path: string;
  url: string;
  download_url: string;
  pages?: number[];
  sections?: string[];
  source_type?: string;
}

export interface GroundingCitation {
  source: string;
  file_name: string;
  page: number | null;
  section: string | null;
  quote: string;
  claims: number[];
}

// Проверка обоснованности ответа по найденным фрагментам базы знаний.
export interface Grounding {
  supported: number;
  partial: number;
  total: number;
  ratio: number;
  unsupported: string[];
  citations: GroundingCitation[];
}

export interface AskAttachment {
  filename: string;
  content_base64: string;
  mime: string;
}

export interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources: ChatSource[];
  grounding?: Grounding | null;
  no_exact_match: boolean;
  // 1 — полезно, -1 — не полезно, null — не оценён
  rating?: number | null;
  created_at: string;
  attachment?: AskAttachment | null;
}

export interface ConversationDetail extends ConversationSummary {
  messages: ChatMessage[];
}

export interface AskResponse {
  answer: string;
  sources: ChatSource[];
  grounding?: Grounding | null;
  no_exact_match: boolean;
  conversation_id: number | null;
  // id сохранённого сообщения ассистента (null, если беседа не ведётся)
  message_id?: number | null;
  attachment?: AskAttachment | null;
}

export interface ProfileOptions {
  divisions: string[];
  subdivisions_by_division: Record<string, string[]>;
}

export interface ProfileData {
  full_name: string;
  division: string;
  subdivision: string;
  subdivision_type: string;
  job_title: string;
  email: string;
}

export interface ProfileResponse {
  profile: ProfileData;
  options: ProfileOptions;
}

export interface CurrentUser {
  email: string;
  full_name: string;
  created_at: string | null;
  roles: string[];
}

export interface AdminUser {
  id: number;
  email: string;
  full_name: string | null;
  division: string | null;
  subdivision: string | null;
  created_at: string | null;
  roles: string[];
}


export interface KnowledgeGap {
  topic: string;
  problem: "no_document" | "weak_document";
  problem_label: string;
  requests: number;
  people: number;
  divisions: string[];
  not_useful: number;
  unanswered: number;
  warned: number;
  examples: string[];
  sources: { relative_path: string; count: number }[];
  last_asked: string | null;
}

export interface GapDocument {
  relative_path: string;
  file_name: string;
  used?: number;
  useful?: number;
  not_useful?: number;
}

export interface KnowledgeGapsReport {
  period_days: number;
  generated_at: string;
  grouping: "embeddings" | "words";
  summary: {
    questions: number;
    people: number;
    answered: number;
    refused: number;
    clarifications: number;
    warned: number;
    rated: number;
    useful: number;
    not_useful: number;
    usefulness: number | null;
  };
  gaps: KnowledgeGap[];
  documents: {
    weak: GapDocument[];
    unused: GapDocument[];
    unused_checked: boolean;
  };
  cad?: CadDemand;
}

// Спрос на конструктор деталей: что просили построить, но это пока не поддерживается.
export interface CadDemandTopic {
  topic: string;
  kind: "refused" | "partial";
  requests: number;
  people: number;
  divisions: string[];
  refused: number;
  partial: number;
  examples: string[];
  last_asked: string | null;
}

export interface CadDemand {
  requests: number;
  built: number;
  unsupported: number;
  clarify: number;
  errors: number;
  topics: CadDemandTopic[];
  grouping: "embeddings" | "words";
}

export interface CadStatus {
  available: boolean;
  version: string | null;
  editor: boolean;
}

export interface CadCheck {
  name: string;
  ok: boolean;
  detail: string;
}

export interface CadResult {
  clarify: string | null;
  // что из описания пока не поддерживается (деталь при этом не построена)
  unsupported?: string | null;
  can_build?: string[];
  attempts: number;
  title?: string;
  summary?: string[];
  assumptions?: string[];
  spec?: Record<string, unknown>;
  script?: { commands: unknown[] };
  checks?: CadCheck[];
  metrics?: { size_mm: number[]; volume_cm3: number; mass_g: number; material: string };
  images?: Record<string, string>;
  step_base64?: string;
}

export interface CadHistoryItem {
  id: number;
  created_at: string | null;
  description: string;
  outcome: "built" | "clarify" | "unsupported" | "error";
  title: string | null;
  detail: string | null;
  // деталь можно собрать заново по сохранённой спецификации
  can_rebuild: boolean;
}
