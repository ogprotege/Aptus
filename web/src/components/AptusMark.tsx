export function AptusMark({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 1024 1024"
      role="img"
      aria-label="Aptus calibrated A"
    >
      <path
        d="M226 806 L460 244 C470 217 489 202 512 202 C535 202 554 217 564 244 L798 806"
        fill="none"
        stroke="currentColor"
        strokeWidth="88"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        className="aptus-mark-accent"
        d="M256 608 H768"
        fill="none"
        strokeWidth="64"
        strokeLinecap="round"
      />
    </svg>
  );
}
