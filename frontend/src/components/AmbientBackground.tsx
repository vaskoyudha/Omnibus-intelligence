export default function AmbientBackground() {
  return (
    <div
      className="fixed inset-0 -z-10 overflow-hidden pointer-events-none"
      aria-hidden="true"
    >
      {/* Blue orb — top right */}
      <div
        className="absolute -top-40 -right-40 w-[500px] h-[500px] rounded-full blur-3xl animate-gradient-shift"
        style={{ background: 'rgba(219, 234, 254, 0.4)' }}
      />

      {/* Violet orb — bottom left */}
      <div
        className="absolute -bottom-40 -left-40 w-[400px] h-[400px] rounded-full blur-3xl animate-gradient-shift"
        style={{
          background: 'rgba(237, 233, 254, 0.3)',
          animationDelay: '-5s',
        }}
      />

      {/* Cyan orb — center */}
      <div
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full blur-3xl animate-pulse-glow"
        style={{ background: 'rgba(236, 254, 255, 0.2)' }}
      />

      {/* Small accent orb — top left */}
      <div
        className="absolute top-1/4 left-1/4 w-[200px] h-[200px] rounded-full blur-3xl animate-float"
        style={{ background: 'rgba(191, 219, 254, 0.15)' }}
      />
    </div>
  );
}
