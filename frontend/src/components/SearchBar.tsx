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
          className="w-full px-6 py-4 text-lg rounded-xl dark-input outline-none"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading || !query.trim()}
          className="absolute right-2 px-6 py-2 bg-gradient-to-r from-[#AAFF00] to-[#88CC00] text-[#0A0A0F] font-semibold rounded-xl hover:shadow-lg hover:shadow-[#AAFF00]/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
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
    </form>
  );
}
