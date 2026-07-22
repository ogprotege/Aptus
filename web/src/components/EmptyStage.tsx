import type { ReactNode } from "react";

interface EmptyStageProps {
  title: string;
  children: ReactNode;
  actionLabel: string;
  onAction: () => void;
}

export function EmptyStage({ title, children, actionLabel, onAction }: EmptyStageProps) {
  return (
    <section className="empty-stage">
      <span className="empty-glyph" aria-hidden="true">⌁</span>
      <h2>{title}</h2>
      <p>{children}</p>
      <button type="button" className="button button-primary" onClick={onAction}>
        {actionLabel}
      </button>
    </section>
  );
}
