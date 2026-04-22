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
      className={`inline-flex items-center justify-center rounded-lg bg-brand px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-brand/90 disabled:cursor-not-allowed disabled:opacity-60 ${className ?? ""}`}
    >
      {isLoading ? loadingLabel : children}
    </button>
  );
}
