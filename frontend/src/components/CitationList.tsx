'use client';

import { useState } from 'react';
import { CitationInfo, mapCitation } from '@/lib/api';

interface CitationListProps {
  citations: CitationInfo[];
}

export default function CitationList({ citations }: CitationListProps) {
  const [expandedCitations, setExpandedCitations] = useState<Set<number>>(new Set());

  if (!citations || citations.length === 0) {
    return null;
  }

  const toggleExpand = (index: number) => {
    setExpandedCitations(prev => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  const expandAll = () => {
    setExpandedCitations(new Set(citations.map((_, i) => i)));
  };

  const collapseAll = () => {
    setExpandedCitations(new Set());
  };

  const getTypeConfig = (type: string) => {
    const configs: Record<string, { bg: string; text: string; border: string; icon: string }> = {
      'UU': { bg: 'bg-blue-500/15', text: 'text-blue-300', border: 'border-blue-500/20', icon: '📜' },
      'PP': { bg: 'bg-emerald-500/15', text: 'text-emerald-300', border: 'border-emerald-500/20', icon: '📋' },
      'Perpres': { bg: 'bg-purple-500/15', text: 'text-purple-300', border: 'border-purple-500/20', icon: '🏛️' },
      'Permen': { bg: 'bg-orange-500/15', text: 'text-orange-300', border: 'border-orange-500/20', icon: '📑' },
      'Perda': { bg: 'bg-teal-500/15', text: 'text-teal-300', border: 'border-teal-500/20', icon: '🏢' },
    };
    return configs[type] || { bg: 'bg-white/5', text: 'text-text-primary', border: 'border-border', icon: '📄' };
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.7) return 'text-emerald-400 bg-emerald-500/10';
    if (score >= 0.4) return 'text-amber-400 bg-amber-500/10';
    return 'text-text-muted bg-white/5';
  };

  const formatScore = (score: number) => {
    if (isNaN(score) || score === undefined || score === null) return 0;
    return Math.round(score * 100);
  };

  return (
    <div className="p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-text-primary flex items-center gap-2">
          <svg className="w-5 h-5 text-[#AAFF00]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          Sumber Referensi
          <span className="ml-1 px-2 py-0.5 text-sm bg-[#AAFF00]/10 text-[#AAFF00] rounded-full font-medium">
            {citations.length}
          </span>
        </h3>

        <div className="flex items-center gap-2 text-sm no-print">
          <button onClick={expandAll} className="text-text-muted hover:text-[#AAFF00] flex items-center gap-1 transition-colors">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
            Buka Semua
          </button>
          <span className="text-border">|</span>
          <button onClick={collapseAll} className="text-text-muted hover:text-[#AAFF00] flex items-center gap-1 transition-colors">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
            </svg>
            Tutup Semua
          </button>
        </div>
      </div>

      {/* Citation Cards */}
      <div className="space-y-3">
        {citations.map((citationInfo, index) => {
          const citation = mapCitation(citationInfo);
          const typeConfig = getTypeConfig(citation.document_type);
          const isExpanded = expandedCitations.has(index);
          const scorePercent = formatScore(citation.relevance_score);

          return (
            <div
              key={index}
              className={`glass border rounded-xl overflow-hidden transition-all duration-200 ${isExpanded
                  ? 'border-[#AAFF00]/30 shadow-md shadow-[#AAFF00]/5'
                  : 'border-border hover:border-[#AAFF00]/15 hover:shadow-sm'
                }`}
            >
              {/* Citation Header */}
              <button
                onClick={() => toggleExpand(index)}
                className="w-full p-4 flex items-center gap-4 text-left focus:outline-none focus:ring-2 focus:ring-[#AAFF00]/30 focus:ring-inset"
              >
                <div className={`flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center font-bold text-sm ${isExpanded ? 'bg-[#AAFF00] text-[#0A0A0F]' : 'bg-[#AAFF00]/10 text-[#AAFF00]'
                  }`}>
                  {index + 1}
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`px-2 py-0.5 text-xs font-medium rounded-md ${typeConfig.bg} ${typeConfig.text}`}>
                      {typeConfig.icon} {citation.document_type}
                    </span>
                    <span className={`px-2 py-0.5 text-xs font-medium rounded-md ${getScoreColor(citation.relevance_score)}`}>
                      {scorePercent}% relevan
                    </span>
                  </div>
                  <h4 className="font-medium text-text-primary text-sm truncate pr-4">
                    {citation.document_title}
                  </h4>
                  {citation.pasal && (
                    <p className="text-sm text-[#AAFF00] font-medium mt-0.5">
                      {citation.pasal}
                    </p>
                  )}
                </div>

                <div className="flex-shrink-0">
                  <svg
                    className={`w-5 h-5 text-text-muted transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`}
                    fill="none" viewBox="0 0 24 24" stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </button>

              {/* Expandable Content */}
              {isExpanded && citation.content_snippet && (
                <div className="px-4 pb-4 border-t border-border">
                  <div className="pt-4 pl-14">
                    <div className="text-xs font-medium text-text-muted uppercase tracking-wide mb-2">
                      Kutipan Teks
                    </div>
                    <div className="p-4 bg-bg-secondary rounded-lg border border-border">
                      <p className="text-sm text-text-secondary leading-relaxed whitespace-pre-wrap">
                        {citation.content_snippet}
                      </p>
                    </div>

                    {/* Relevance Bar */}
                    <div className="mt-3 flex items-center gap-3">
                      <span className="text-xs text-text-muted">Relevansi:</span>
                      <div className="flex-1 h-2 bg-bg-tertiary rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${scorePercent >= 70 ? 'bg-emerald-500' :
                              scorePercent >= 40 ? 'bg-amber-500' : 'bg-text-muted'
                            }`}
                          style={{ width: `${scorePercent}%` }}
                        />
                      </div>
                      <span className="text-xs font-medium text-text-secondary">{scorePercent}%</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
