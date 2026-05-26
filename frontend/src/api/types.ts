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

export interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources: ChatSource[];
  no_exact_match: boolean;
  created_at: string;
}

export interface ConversationDetail extends ConversationSummary {
  messages: ChatMessage[];
}

export interface AskResponse {
  answer: string;
  sources: ChatSource[];
  no_exact_match: boolean;
  conversation_id: number | null;
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
