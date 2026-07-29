export function AptusMark({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 1024 1024"
      role="img"
      aria-label="Aptus calibrated A"
    >
      <path
        d="M200 820 L512 210 L824 820"
        fill="none"
        stroke="currentColor"
        strokeWidth="74"
        strokeLinejoin="miter"
        strokeLinecap="butt"
        strokeMiterlimit="12"
      />
      <path
        className="aptus-mark-accent"
        d="M281.2 580 L742.8 580 L777.6 648 L246.4 648 Z"
        stroke="none"
      />
    </svg>
  );
}
