"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Search } from "lucide-react";
import AppShell from "@/components/shell/AppShell";
import { useApi, endpoints } from "@/lib/api";
import {
  Panel,
  SectionHeader,
  SkeletonRows,
} from "@/components/ui/primitives";

export default function CompanyIndex() {
  const router = useRouter();
  const [q, setQ] = useState("");
  const { data, isLoading } = useApi(q.length >= 1 ? `find:${q}` : null, () =>
    endpoints.companies({ search: q, per_page: 10 }),
  );

  return (
    <AppShell
      title="Company deep-dive"
      subtitle="Search the universe, then open a firm to see its estimate and the attribution behind it."
    >
      <Panel className="mx-auto max-w-xl">
        <SectionHeader eyebrow="find a firm" title="Search" />
        <div className="relative">
          <Search
            size={15}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-500"
          />
          <input
            autoFocus
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="ticker (SYN0142) or name"
            className="tnum w-full rounded-md border border-ink-600 bg-ink-900 py-2.5 pl-9 pr-3 text-sm text-ink-100 placeholder:text-ink-500 focus:border-amber"
          />
        </div>

        <div className="mt-3 min-h-[80px]">
          {isLoading && <SkeletonRows rows={6} label="search results" />}
          {data && (
            <ul className="divide-y divide-ink-800">
              {data.companies.map((c) => (
                <li key={c.ticker}>
                  <button
                    onClick={() =>
                      router.push(
                        `/company/${encodeURIComponent(c.ticker)}`,
                      )
                    }
                    className="flex w-full items-center justify-between py-2.5 text-left hover:text-ink-50"
                  >
                    <span>
                      <span className="tnum text-sm text-ink-100">
                        {c.ticker.split(" ")[0]}
                      </span>
                      <span className="ml-2 text-xs text-ink-400">
                        {c.name}
                      </span>
                    </span>
                    <span className="text-xs text-ink-500">{c.sector}</span>
                  </button>
                </li>
              ))}
              {data.companies.length === 0 && (
                <li className="py-3 text-sm text-ink-500">No match.</li>
              )}
            </ul>
          )}
        </div>
      </Panel>
    </AppShell>
  );
}
