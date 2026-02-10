interface SkeletonLoaderProps {
  variant?: 'text' | 'card' | 'paragraph';
  lines?: number;
}

export default function SkeletonLoader({ variant = 'card', lines = 3 }: SkeletonLoaderProps) {
  if (variant === 'text') {
    return (
      <div className="space-y-3" role="status" aria-label="Memuat...">
        {Array.from({ length: lines }).map((_, i) => (
          <div
            key={i}
            className={`h-4 rounded-lg animate-shimmer ${
              i === lines - 1 ? 'w-3/4' : 'w-full'
            }`}
          />
        ))}
        <span className="sr-only">Memuat...</span>
      </div>
    );
  }

  if (variant === 'paragraph') {
    return (
      <div className="space-y-4" role="status" aria-label="Memuat...">
        <div className="h-6 w-1/3 rounded-lg animate-shimmer" />
        <div className="space-y-2">
          {Array.from({ length: lines }).map((_, i) => (
            <div
              key={i}
              className={`h-4 rounded-lg animate-shimmer ${
                i === lines - 1 ? 'w-2/3' : 'w-full'
              }`}
              style={{ animationDelay: `${i * 0.1}s` }}
            />
          ))}
        </div>
        <span className="sr-only">Memuat...</span>
      </div>
    );
  }

  // Card variant (default)
  return (
    <div className="glass rounded-2xl p-6 space-y-4" role="status" aria-label="Memuat...">
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 rounded-xl animate-shimmer flex-shrink-0" />
        <div className="space-y-2 flex-1">
          <div className="h-5 w-1/3 rounded-lg animate-shimmer" />
          <div className="h-3 w-1/4 rounded-lg animate-shimmer" />
        </div>
      </div>
      <div className="space-y-2">
        {Array.from({ length: lines }).map((_, i) => (
          <div
            key={i}
            className={`h-4 rounded-lg animate-shimmer ${
              i === lines - 1 ? 'w-3/4' : 'w-full'
            }`}
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </div>
      <span className="sr-only">Memuat...</span>
    </div>
  );
}
