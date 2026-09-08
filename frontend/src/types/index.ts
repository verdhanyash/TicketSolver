// Shared domain types mirrored from the backend API (FR-16, FR-8, FR-10).

export type TicketOutcome = "auto_resolved" | "resolved" | "pending_approval" | "escalated" | "rejected" | "open";

export interface Ticket {
  id: string;
  subject: string;
  body?: string;
  category?: string;
  severity?: string;
  outcome: TicketOutcome;
  proposed_resolution?: string;
  createdAt: string;
}

export interface DailyPoint {
  date: string;
  total: number;
  auto_resolved: number;
  resolved: number;
  pending_approval: number;
  escalated: number;
  rejected: number;
  in_progress: number;
}

export interface VolumeOut {
  total_tickets: number;
  by_status: Record<string, number>;
  daily: DailyPoint[];
  auto_resolution_rate_pct: number;
}

export interface CostOut {
  has_usage_data: boolean;
  llm_calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface EvaluationSummary {
  total_tickets: number;
  correct_outcomes: number;
  outcome_accuracy_pct: number;
  resolution_correctness_pct: number;
  guardrail_intercept_pct: number;
  escalation_correctness_pct: number;
  escalation_metrics: {
    precision: number;
    recall: number;
    f1: number;
  };
  confusion_matrix: Record<string, Record<string, number>>;
  scenario_breakdown: Record<string, { total: number; correct: number; accuracy_pct: number }>;
}

export interface EvaluationReport {
  eval_run_id: string;
  generated_at: string;
  summary: EvaluationSummary;
  quality_gates: Record<string, boolean>;
  model_metrics?: {
    ticket_classifier_m6?: {
      category_accuracy?: number;
      category_macro_f1?: number;
      severity_accuracy?: number;
      severity_macro_f1?: number;
    };
    cost_router_m7?: {
      xgboost_accuracy?: number;
      xgboost_macro_f1?: number;
      baseline_accuracy?: number;
      baseline_macro_f1?: number;
      advantage_f1?: number;
      gate_passed?: boolean;
    };
    confidence_calibration_m7?: {
      brier_score?: number;
      accuracy?: number;
      log_loss?: number;
      brier_gate_passed?: boolean;
    };
  };
}

export interface QualityOut {
  available: boolean;
  report?: EvaluationReport | null;
  error?: string | null;
}

export interface QueueOut {
  pending_count: number;
  oldest_pending_age_minutes?: number | null;
}

export interface DashboardSummaryOut {
  generated_at: string;
  volume: VolumeOut;
  cost: CostOut;
  quality: QualityOut;
  queue: QueueOut;
}

export interface ApprovalOut {
  id: string;
  ticket_id: string;
  action_kind: string;
  amount_usd?: number | null;
  reason: string;
  status: "pending" | "approved" | "rejected" | "corrected";
  decided_by?: string | null;
  decision_note?: string | null;
  corrected_action?: string | null;
  created_at: string;
  decided_at?: string | null;
}

export interface Approval {
  ticketId: string;
  proposedAction?: string;
  reason: string;
  trace?: Trace;
}

export interface DecisionIn {
  decision: "approved" | "rejected" | "corrected";
  decided_by?: string;
  decision_note?: string;
  corrected_action?: string;
}

export interface TraceNode {
  id: string;
  label: string;
  kind: "agent" | "tool_call" | "memory" | "guardrail" | "decision";
}

export interface TraceEdge {
  from: string;
  to: string;
}

export interface Trace {
  ticketId: string;
  nodes: TraceNode[];
  edges: TraceEdge[];
  summary: string;
}
