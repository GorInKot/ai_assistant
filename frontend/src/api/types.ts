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
  no_exact_match: boolean;
  created_at: string;
  attachment?: AskAttachment | null;
}

export interface ConversationDetail extends ConversationSummary {
  messages: ChatMessage[];
}

export interface AskResponse {
  answer: string;
  sources: ChatSource[];
  no_exact_match: boolean;
  conversation_id: number | null;
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

export interface ResponsibilityArea {
  id: number;
  slug: string;
  name: string;
  description: string | null;
}

export interface Employee {
  id: number;
  user_id: number | null;
  email: string;
  full_name: string;
  position: string | null;
  division: string | null;
  subdivision: string | null;
  phone: string | null;
  is_active: boolean;
  responsibility_area_slugs: string[];
}

export interface EmployeeImportResult {
  created: number;
  updated: number;
  skipped: number;
  errors: string[];
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

export type RequestStatus = "new" | "in_progress" | "done" | "rejected";

export interface RequestTypeSlot {
  id?: number;
  name: string;
  question: string;
  required: boolean;
  sort_order?: number;
}

export interface RequestTypeDef {
  id?: number;
  type_slug: string;
  title: string;
  responsibility_area_slug: string;
  is_anonymous: boolean;
  is_active: boolean;
  trigger_keywords: string[];
  examples: string[];
  sort_order?: number;
  slots: RequestTypeSlot[];
}

export interface RequestItem {
  id: number;
  type_slug: string;
  type_title: string;
  is_anonymous: boolean;
  status: RequestStatus;
  summary: string | null;
  payload: Record<string, string>;
  requester_name: string | null;
  requester_email: string | null;
  assigned_employee_id: number | null;
  assigned_employee_name: string | null;
  conversation_id: number | null;
  created_at: string;
  updated_at: string;
}
