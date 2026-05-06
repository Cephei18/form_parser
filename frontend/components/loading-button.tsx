import type { ButtonHTMLAttributes } from "react";

interface LoadingButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  isLoading?: boolean;
  loadingLabel?: string;
}

export function LoadingButton({
  children,
  isLoading,
  loadingLabel = "Processing...",
  disabled,
  className,
  ...rest
}: LoadingButtonProps) {
  return (
    <button
      type="button"
      {...rest}
      disabled={disabled || isLoading}
      className={`inline-flex items-center justify-center rounded-2xl bg-ink px-5 py-3 text-sm font-semibold text-white shadow-soft transition hover:-translate-y-0.5 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:translate-y-0 ${className ?? ""}`}
    >
      {isLoading ? (
        <span className="inline-flex items-center gap-2">
          <span className="h-2 w-2 animate-pulse rounded-full bg-white" />
          {loadingLabel}
        </span>
      ) : (
        children
      )}
    </button>
  );
}
