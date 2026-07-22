export function AptusMark({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 1024 1024"
      role="img"
      aria-label="Aptus calibrated A"
    >
      <path
        d="M218 806 447 232c17-43 78-43 96 0l26 65"
        fill="none"
        stroke="currentColor"
        strokeWidth="82"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        className="aptus-mark-accent"
        d="M566 297 785 806"
        fill="none"
        strokeWidth="82"
        strokeLinecap="round"
      />
      <path
        d="M345 625h322"
        fill="none"
        stroke="currentColor"
        strokeWidth="58"
        strokeLinecap="round"
      />
      <path
        className="aptus-mark-accent"
        d="M667 625h82V510h76"
        fill="none"
        strokeWidth="34"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle className="aptus-mark-point" cx="825" cy="510" r="35" />
    </svg>
  );
}
