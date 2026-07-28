export function AptusMark({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 1024 1024"
      role="img"
      aria-label="Aptus calibrated A"
    >
      <path
        d="M218 806 447 232c17-43 78-43 96 0L772 806"
        fill="none"
        stroke="currentColor"
        strokeWidth="82"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        className="aptus-mark-accent"
        d="M245 625H745"
        fill="none"
        strokeWidth="58"
        strokeLinecap="round"
      />
    </svg>
  );
}
