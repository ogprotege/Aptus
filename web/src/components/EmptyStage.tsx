import type { ReactNode } from "react";

interface EmptyStageProps {
  title: string;
  children: ReactNode;
  actionLabel: string;
  onAction: () => void;
  tone?: "path" | "omitted";
}

export function EmptyStage({ title, children, actionLabel, onAction, tone = "path" }: EmptyStageProps) {
  const omitted = tone === "omitted";
  return (
    <section className={omitted ? "empty-stage evidence-omitted" : "empty-stage"}>
      <span className="empty-glyph" aria-hidden="true">⌁</span>
      <h2>{title}</h2>
      <p>{children}</p>
      <button type="button" className={omitted ? "button button-secondary" : "button button-primary"} onClick={onAction}>
        {actionLabel}
      </button>
    </section>
  );
}
