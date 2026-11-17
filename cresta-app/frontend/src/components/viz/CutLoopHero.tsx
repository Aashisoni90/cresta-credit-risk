/*
  The thesis, drawn. Left: the circular trap — a distress label defined from
  the same period's accounting that then feeds back in as features. Right:
  what CRESTA does — cut the loop, separate it in time, drop the circular
  features, and forecast FY(t+1) from FY(t).
  One pulse travels the forward arrow on load; nothing else moves.
*/
export default function CutLoopHero() {
  return (
    <div className="w-full">
      <svg
        viewBox="0 0 720 220"
        className="h-auto w-full"
        role="img"
        aria-label="Diagram: the circular distress-label trap on the left, cut and unrolled into a forward one-year prediction on the right"
      >
        <defs>
          <marker
            id="arrow"
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path d="M0 0L10 5L0 10z" fill="var(--amber)" />
          </marker>
        </defs>

        {/* ---- left: the loop, broken ---- */}
        <g fontFamily="var(--font-mono)" fontSize="11">
          <text
            x="120"
            y="24"
            fill="var(--ink-500)"
            textAnchor="middle"
            fontFamily="var(--font-body)"
          >
            label circularity
          </text>
          <circle
            cx="120"
            cy="120"
            r="58"
            fill="none"
            stroke="var(--ink-600)"
            strokeWidth="1.5"
            strokeDasharray="4 5"
          />
          {/* severed ends */}
          <path
            d="M120 62 q 22 4 30 20"
            fill="none"
            stroke="var(--amber)"
            strokeWidth="1.5"
          />
          <path
            d="M120 62 q -22 4 -30 20"
            fill="none"
            stroke="var(--amber)"
            strokeWidth="1.5"
          />
          <line
            x1="112"
            y1="52"
            x2="128"
            y2="72"
            stroke="var(--amber)"
            strokeWidth="1.8"
          />
          <line
            x1="128"
            y1="52"
            x2="112"
            y2="72"
            stroke="var(--amber)"
            strokeWidth="1.8"
          />
          <text x="120" y="108" fill="var(--ink-300)" textAnchor="middle">
            distress
          </text>
          <text x="120" y="124" fill="var(--ink-300)" textAnchor="middle">
            label
          </text>
          <text x="120" y="140" fill="var(--ink-500)" textAnchor="middle">
            = f(accounting)
          </text>
        </g>

        {/* ---- bridge ---- */}
        <text
          x="270"
          y="116"
          fill="var(--ink-500)"
          fontSize="20"
          fontFamily="var(--font-body)"
        >
          →
        </text>

        {/* ---- right: cut, unrolled, forward ---- */}
        <g fontFamily="var(--font-mono)" fontSize="11">
          <text
            x="500"
            y="24"
            fill="var(--ink-500)"
            textAnchor="middle"
            fontFamily="var(--font-body)"
          >
            forward-looking, non-circular
          </text>

          <rect
            x="320"
            y="96"
            width="118"
            height="48"
            rx="6"
            fill="var(--ink-850)"
            stroke="var(--ink-600)"
          />
          <text x="379" y="116" fill="var(--ink-200)" textAnchor="middle">
            features
          </text>
          <text x="379" y="132" fill="var(--ink-500)" textAnchor="middle">
            FY t
          </text>

          <line
            x1="446"
            y1="120"
            x2="556"
            y2="120"
            stroke="var(--amber)"
            strokeWidth="1.6"
            markerEnd="url(#arrow)"
          />
          {/* the pulse */}
          <circle r="3" fill="var(--amber)">
            <animate
              attributeName="cx"
              from="446"
              to="552"
              dur="1.6s"
              begin="0.3s"
              repeatCount="1"
              fill="freeze"
            />
            <animate
              attributeName="cy"
              from="120"
              to="120"
              dur="1.6s"
              begin="0.3s"
              repeatCount="1"
              fill="freeze"
            />
            <animate
              attributeName="opacity"
              values="0;1;1;0"
              dur="1.6s"
              begin="0.3s"
              repeatCount="1"
              fill="freeze"
            />
          </circle>
          <text x="500" y="112" fill="var(--ink-500)" textAnchor="middle">
            1-year lag
          </text>

          <rect
            x="560"
            y="96"
            width="118"
            height="48"
            rx="6"
            fill="var(--amber-glow)"
            stroke="var(--amber-dim)"
          />
          <text x="619" y="116" fill="var(--amber)" textAnchor="middle">
            distress label
          </text>
          <text x="619" y="132" fill="var(--ink-400)" textAnchor="middle">
            FY t+1
          </text>

          <text
            x="500"
            y="176"
            fill="var(--ink-500)"
            textAnchor="middle"
            fontSize="10"
          >
            6 circular features removed · 38 model inputs
          </text>
        </g>
      </svg>
    </div>
  );
}
