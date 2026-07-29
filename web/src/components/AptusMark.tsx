export function AptusMark({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 1024 1024"
      role="img"
      aria-label="Aptus tile mark"
    >
      <rect
        className="aptus-mark-tile"
        x="72"
        y="72"
        width="880"
        height="880"
        rx="132"
      />
      <path
        d="M268 780 L512 302 L756 780"
        fill="none"
        stroke="currentColor"
        strokeWidth="76"
        strokeLinejoin="miter"
        strokeLinecap="butt"
        strokeMiterlimit="12"
      />
      <path
        className="aptus-mark-accent"
        d="M356 606 L668 606 L692 654 L332 654 Z"
        stroke="none"
      />
    </svg>
  );
}
