"use client";

import { Download } from "lucide-react";
import { exportUrl, useApi, endpoints } from "@/lib/api";

/*
  §12.6: "CSV and PDF from portfolio and company views, with model version and
  generation timestamp stamped on the export."

  The stamp lives in the CSV preamble, written server-side — not assembled
  here, where it could drift from the model that actually produced the rows.
  The tooltip surfaces the same version so the user knows before downloading.
*/
export default function ExportButton({
  kind,
  params = {},
  label = "Export CSV",
}: {
  kind: "portfolio" | "company";
  params?: Record<string, string | number>;
  label?: string;
}) {
  const { data } = useApi("export-manifest", endpoints.exportManifest);
  const version = data?.model_version ?? "";

  return (
    <a
      href={exportUrl(kind, params)}
      download
      title={
        version
          ? `CSV stamped with model ${version}, generation timestamp and the cost assumption`
          : "CSV stamped with model version and generation timestamp"
      }
      className="inline-flex items-center gap-1.5 rounded-md border border-ink-600 px-2.5 py-1.5 text-2xs text-ink-300 hover:bg-ink-850 hover:text-ink-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber"
    >
      <Download size={13} aria-hidden="true" />
      {label}
    </a>
  );
}
