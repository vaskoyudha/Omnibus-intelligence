export default function AmbientBackground() {
  return (
    <div
      className="fixed inset-0 -z-10 overflow-hidden pointer-events-none"
      aria-hidden="true"
    >
      {/* Blue orb — top right */}
      <div
        className="absolute -top-32 -right-32 w-[700px] h-[700px] rounded-full blur-3xl animate-gradient-shift"
        style={{ background: 'rgba(59, 130, 246, 0.15)' }}
      />

      {/* Violet orb — bottom left */}
      <div
        className="absolute -bottom-32 -left-32 w-[600px] h-[600px] rounded-full blur-3xl animate-gradient-shift"
        style={{
          background: 'rgba(139, 92, 246, 0.12)',
          animationDelay: '-5s',
        }}
      />

      {/* Cyan orb — center */}
      <div
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full blur-3xl animate-pulse-glow"
        style={{ background: 'rgba(34, 211, 238, 0.08)' }}
      />

      {/* Accent orb — top left */}
      <div
        className="absolute top-1/4 left-1/4 w-[350px] h-[350px] rounded-full blur-3xl animate-float"
        style={{ background: 'rgba(37, 99, 235, 0.1)' }}
      />

      {/* Extra warm orb — bottom right for depth */}
      <div
        className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] rounded-full blur-3xl animate-gradient-shift"
        style={{
          background: 'rgba(168, 85, 247, 0.07)',
          animationDelay: '-10s',
        }}
      />
    </div>
  );
}
