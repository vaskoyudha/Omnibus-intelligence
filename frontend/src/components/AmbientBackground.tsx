export default function AmbientBackground() {
  return (
    <div
      className="fixed inset-0 -z-10 overflow-hidden pointer-events-none"
      aria-hidden="true"
    >
      {/* Deep purple haze — top right (FlowFunds outer glow) */}
      <div
        className="absolute -top-40 -right-40 w-[800px] h-[800px] rounded-full blur-3xl animate-gradient-shift"
        style={{ background: 'rgba(120, 80, 200, 0.12)' }}
      />

      {/* Indigo haze — bottom left */}
      <div
        className="absolute -bottom-40 -left-40 w-[700px] h-[700px] rounded-full blur-3xl animate-gradient-shift"
        style={{
          background: 'rgba(90, 60, 180, 0.1)',
          animationDelay: '-5s',
        }}
      />

      {/* Center dark lavender pulse */}
      <div
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[900px] rounded-full blur-3xl animate-pulse-glow"
        style={{ background: 'rgba(100, 70, 200, 0.06)' }}
      />

      {/* Accent green subtle glow — top center */}
      <div
        className="absolute top-1/3 left-1/2 -translate-x-1/2 w-[400px] h-[400px] rounded-full blur-3xl animate-float"
        style={{ background: 'rgba(170, 255, 0, 0.03)' }}
      />

      {/* Deep purple — bottom right */}
      <div
        className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] rounded-full blur-3xl animate-gradient-shift"
        style={{
          background: 'rgba(80, 40, 160, 0.08)',
          animationDelay: '-10s',
        }}
      />
    </div>
  );
}
