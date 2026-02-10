'use client';

import { useState, FormEvent } from 'react';

interface SearchBarProps {
  onSearch: (query: string) => void;
  isLoading: boolean;
}

export default function SearchBar({ onSearch, isLoading }: SearchBarProps) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSearch(query.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-3xl mx-auto">
      <div className="relative flex items-center">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ketik pertanyaan hukum Anda..."
          className="w-full px-6 py-4 text-lg rounded-xl border border-border bg-white/60 backdrop-blur-sm focus:border-accent focus:shadow-glow outline-none transition-all text-text-primary placeholder-text-muted"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading || !query.trim()}
          className="absolute right-2 px-6 py-2 bg-gradient-to-r from-accent to-accent-dark text-white font-semibold rounded-xl hover:shadow-lg hover:shadow-accent/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          {isLoading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Mencari...
            </span>
          ) : (
            'Tanya'
          )}
        </button>
      </div>
      <p className="mt-2 text-sm text-text-muted text-center">
        Contoh: &quot;Apa saja hak pekerja menurut UU Cipta Kerja?&quot; atau &quot;Bagaimana prosedur perizinan usaha?&quot;
      </p>
    </form>
  );
}
