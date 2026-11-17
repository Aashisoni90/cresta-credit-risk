"use client";

import { useState, useMemo } from "react";
import { ChevronDown, ChevronUp, ChevronsUpDown } from "lucide-react";

export interface Column<Row> {
  key: string;
  header: string;
  align?: "left" | "right";
  sortable?: boolean;
  sortValue?: (row: Row) => number | string;
  render: (row: Row) => React.ReactNode;
  width?: string;
}

export default function DataTable<Row>({
  columns,
  rows,
  getKey,
  onRowClick,
  initialSort,
}: {
  columns: Column<Row>[];
  rows: Row[];
  getKey: (row: Row) => string;
  onRowClick?: (row: Row) => void;
  initialSort?: { key: string; dir: "asc" | "desc" };
}) {
  const [sort, setSort] = useState(initialSort ?? null);

  const sorted = useMemo(() => {
    if (!sort) return rows;
    const col = columns.find((c) => c.key === sort.key);
    if (!col?.sortValue) return rows;
    const dir = sort.dir === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      const av = col.sortValue!(a);
      const bv = col.sortValue!(b);
      if (av < bv) return -1 * dir;
      if (av > bv) return 1 * dir;
      return 0;
    });
  }, [rows, sort, columns]);

  const toggle = (key: string) => {
    setSort((s) =>
      s?.key === key
        ? { key, dir: s.dir === "asc" ? "desc" : "asc" }
        : { key, dir: "desc" },
    );
  };

  return (
    <div className="overflow-x-auto rounded-card border border-ink-700">
      <table className="w-full min-w-[640px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-ink-700 bg-ink-900">
            {columns.map((c) => (
              <th
                key={c.key}
                scope="col"
                style={{ width: c.width }}
                // Sort direction is conveyed by a chevron glyph alone, which
                // a screen reader does not read. aria-sort states it.
                aria-sort={
                  !c.sortable
                    ? undefined
                    : sort?.key === c.key
                      ? sort.dir === "asc"
                        ? "ascending"
                        : "descending"
                      : "none"
                }
                className={`px-3 py-2.5 text-2xs font-medium uppercase tracking-wider text-ink-400 ${
                  c.align === "right" ? "text-right" : "text-left"
                }`}
              >
                {c.sortable ? (
                  <button
                    onClick={() => toggle(c.key)}
                    aria-label={`Sort by ${c.header}`}
                    className={`inline-flex items-center gap-1 rounded hover:text-ink-100 focus:outline-none focus-visible:ring-1 focus-visible:ring-amber ${
                      c.align === "right" ? "flex-row-reverse" : ""
                    }`}
                  >
                    {c.header}
                    {sort?.key === c.key ? (
                      sort.dir === "asc" ? (
                        <ChevronUp size={12} />
                      ) : (
                        <ChevronDown size={12} />
                      )
                    ) : (
                      <ChevronsUpDown size={12} className="text-ink-600" />
                    )}
                  </button>
                ) : (
                  c.header
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr
              key={getKey(row)}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
              // A row that responds to a mouse must respond to a keyboard
              // (WCAG 2.1.1, Level A). Without tabIndex/role/onKeyDown the
              // entire table was unreachable without a pointer.
              tabIndex={onRowClick ? 0 : undefined}
              role={onRowClick ? "button" : undefined}
              onKeyDown={
                onRowClick
                  ? (e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        onRowClick(row);
                      }
                    }
                  : undefined
              }
              className={`border-b border-ink-800 last:border-0 ${
                onRowClick
                  ? "cursor-pointer hover:bg-ink-850 focus:outline-none focus-visible:bg-ink-850 focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-amber"
                  : ""
              }`}
            >
              {columns.map((c) => (
                <td
                  key={c.key}
                  className={`px-3 py-2.5 ${
                    c.align === "right"
                      ? "text-right tabular-nums"
                      : "text-left"
                  }`}
                >
                  {c.render(row)}
                </td>
              ))}
            </tr>
          ))}
          {sorted.length === 0 && (
            <tr>
              <td
                colSpan={columns.length}
                className="px-3 py-8 text-center text-sm text-ink-500"
              >
                No rows match.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
