import SideRail from "./SideRail";
import StatusStrip from "./StatusStrip";

export default function AppShell({
  title,
  subtitle,
  actions,
  children,
}: {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-screen flex-col bg-ink-950 lg:flex-row">
      <SideRail />
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex flex-wrap items-end justify-between gap-3 border-b border-ink-700 bg-ink-900 px-5 pb-4 pt-5">
          <div>
            <h1 className="font-display text-xl font-semibold tracking-tight text-ink-50">
              {title}
            </h1>
            {subtitle && (
              <p className="mt-1 max-w-2xl text-sm text-ink-400">{subtitle}</p>
            )}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </header>
        <StatusStrip />
        <main className="flex-1 px-5 py-6">
          <div className="mx-auto max-w-[1180px]">{children}</div>
        </main>
      </div>
    </div>
  );
}
