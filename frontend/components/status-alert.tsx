interface StatusAlertProps {
  kind: "success" | "error";
  message: string;
}

export function StatusAlert({ kind, message }: StatusAlertProps) {
  const className =
    kind === "success"
      ? "border-ok/30 bg-ok/10 text-ok"
      : "border-warn/30 bg-warn/10 text-warn";

  return (
    <div className={`rounded-lg border px-4 py-3 text-sm ${className}`} role="status">
      {message}
    </div>
  );
}
