"use client";

import { useApi, endpoints } from "@/lib/api";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import {
  LayoutGrid,
  Building2,
  GaugeCircle,
  ShieldCheck,
  Scissors,
  Menu,
  X,
} from "lucide-react";

const NAV = [
  { href: "/portfolio", label: "Portfolio", icon: LayoutGrid },
  { href: "/company", label: "Company", icon: Building2, match: "/company" },
  { href: "/model", label: "Model", icon: GaugeCircle },
  { href: "/data", label: "Data & Features", icon: ShieldCheck },
];

export default function SideRail() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  const isActive = (href: string, match?: string) =>
    match ? pathname.startsWith(match) : pathname === href;

  return (
    <>
      {/* mobile top bar */}
      <div className="flex items-center justify-between border-b border-ink-700 bg-ink-900 px-4 py-3 lg:hidden">
        <Brand />
        <button
          onClick={() => setOpen((v) => !v)}
          aria-label={open ? "Close menu" : "Open menu"}
          className="rounded p-1 text-ink-300 hover:text-ink-50"
        >
          {open ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {open && (
        <nav className="border-b border-ink-700 bg-ink-900 px-2 py-2 lg:hidden">
          {NAV.map((item) => (
            <RailLink
              key={item.href}
              {...item}
              active={isActive(item.href, item.match)}
              onClick={() => setOpen(false)}
            />
          ))}
        </nav>
      )}

      {/* desktop rail */}
      <aside className="sticky top-0 hidden h-screen w-[184px] shrink-0 flex-col border-r border-ink-700 bg-ink-900 lg:flex">
        <div className="border-b border-ink-700 px-4 py-4">
          <Brand />
        </div>
        <nav className="flex flex-1 flex-col gap-0.5 p-2">
          {NAV.map((item) => (
            <RailLink
              key={item.href}
              {...item}
              active={isActive(item.href, item.match)}
            />
          ))}
        </nav>
        <div className="border-t border-ink-700 p-4">
          <p className="text-2xs uppercase tracking-wider text-ink-500">
            Build
          </p>
          <BuildVersion />
        </div>
      </aside>
    </>
  );
}

function Brand() {
  return (
    <Link href="/" className="flex items-center gap-2">
      <span className="grid h-6 w-6 place-items-center rounded-sm bg-amber-glow text-amber">
        <Scissors size={13} strokeWidth={2.4} />
      </span>
      <span className="font-display text-sm font-semibold tracking-[0.14em] text-ink-50">
        CRESTA
      </span>
    </Link>
  );
}

function RailLink({
  href,
  label,
  icon: Icon,
  active,
  onClick,
}: {
  href: string;
  label: string;
  icon: typeof LayoutGrid;
  active: boolean;
  onClick?: () => void;
}) {
  return (
    <Link
      href={href}
      onClick={onClick}
      aria-current={active ? "page" : undefined}
      className={`flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors ${
        active
          ? "bg-ink-800 text-ink-50"
          : "text-ink-400 hover:bg-ink-850 hover:text-ink-100"
      }`}
    >
      <Icon
        size={16}
        strokeWidth={2}
        className={active ? "text-amber" : ""}
      />
      {label}
    </Link>
  );
}

/** Served model version, read from the API — never a hardcoded build string,
 *  which silently misstates which model produced the numbers on screen. */
function BuildVersion() {
  const { data } = useApi("model-info", endpoints.modelInfo);
  return (
    <p className="tnum mt-1 text-xs text-ink-400">{data?.version ?? "\u2014"}</p>
  );
}
