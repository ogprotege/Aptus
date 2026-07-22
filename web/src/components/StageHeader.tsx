import type { ReactNode } from "react";

interface StageHeaderProps {
  eyebrow: string;
  title: string;
  lede: string;
  meta?: ReactNode;
}

export function StageHeader({ eyebrow, title, lede, meta }: StageHeaderProps) {
  return (
    <header className="stage-header">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1 id="stage-heading" tabIndex={-1}>{title}</h1>
        <p className="stage-lede">{lede}</p>
      </div>
      {meta ? <div className="stage-meta">{meta}</div> : null}
    </header>
  );
}
