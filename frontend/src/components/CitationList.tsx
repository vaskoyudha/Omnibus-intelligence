'use client';

import { Citation } from '@/lib/api';

interface CitationListProps {
  citations: Citation[];
}

export default function CitationList({ citations }: CitationListProps) {
  if (!citations || citations.length === 0) {
    return null;
  }

  const getTypeColor = (type: string) => {
    const typeUpper = type?.toUpperCase() ?? '';
    if (typeUpper.includes('UU') || typeUpper.includes('UNDANG')) {
      return 'bg-blue-100 text-blue-800 border-blue-200';
    }
    if (typeUpper.includes('PP') || typeUpper.includes('PEMERINTAH')) {
      return 'bg-green-100 text-green-800 border-green-200';
    }
    if (typeUpper.includes('PERPRES') || typeUpper.includes('PRESIDEN')) {
      return 'bg-purple-100 text-purple-800 border-purple-200';
    }
    if (typeUpper.includes('PERMEN') || typeUpper.includes('MENTERI')) {
      return 'bg-orange-100 text-orange-800 border-orange-200';
    }
    if (typeUpper.includes('PERDA') || typeUpper.includes('DAERAH')) {
      return 'bg-teal-100 text-teal-800 border-teal-200';
    }
    return 'bg-slate-100 text-slate-800 border-slate-200';
  };

  const formatScore = (score: number) => {
    return Math.round(score * 100);
  };

  // Extract document type from citation string or metadata
  const getDocumentType = (citation: Citation): string => {
    // Try metadata first
    if (citation.metadata?.jenis_dokumen) {
      return String(citation.metadata.jenis_dokumen);
    }
    // Legacy field
    if (citation.document_type) {
      return citation.document_type;
    }
    // Parse from citation string (e.g., "UU No. 11 Tahun 2020")
    const match = citation.citation?.match(/^(UU|PP|Perpres|Perda|Permen|Kepmen|Pergub|Perbup|Perwal)/i);
    return match ? match[1].toUpperCase() : 'Dokumen';
  };

  // Extract title from citation string or metadata
  const getDocumentTitle = (citation: Citation): string => {
    if (citation.metadata?.judul) {
      return String(citation.metadata.judul);
    }
    if (citation.document_title) {
      return citation.document_title;
    }
    return citation.citation || 'Dokumen Hukum';
  };

  // Get relevance score
  const getRelevanceScore = (citation: Citation): number => {
    return citation.score ?? citation.relevance_score ?? 0;
  };

  return (
    <div className="mt-6">
      <h3 className="text-lg font-semibold text-slate-800 mb-3 flex items-center gap-2">
        <svg className="w-5 h-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        Sumber Referensi ({citations.length})
      </h3>
      <div className="space-y-3">
        {citations.map((citation, index) => {
          const docType = getDocumentType(citation);
          const docTitle = getDocumentTitle(citation);
          const relevanceScore = getRelevanceScore(citation);
          const citationNumber = citation.number ?? index + 1;

          return (
            <div
              key={citation.citation_id || index}
              className="p-4 bg-white border border-slate-200 rounded-lg hover:border-blue-300 hover:shadow-sm transition-all duration-200"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span className={`px-2 py-0.5 text-xs font-medium rounded border ${getTypeColor(docType)}`}>
                      {docType}
                    </span>
                    <span className="text-xs text-slate-400">
                      Relevansi: {formatScore(relevanceScore)}%
                    </span>
                  </div>
                  <h4 className="font-medium text-slate-800 text-sm">
                    [{citationNumber}] {docTitle}
                  </h4>
                  {citation.citation && citation.citation !== docTitle && (
                    <p className="text-sm text-blue-600 font-medium mt-1">
                      {citation.citation}
                    </p>
                  )}
                  {typeof citation.metadata?.ringkasan === 'string' && citation.metadata.ringkasan && (
                    <p className="text-sm text-slate-600 mt-2 line-clamp-3">
                      {citation.metadata.ringkasan}
                    </p>
                  )}
                  {!(typeof citation.metadata?.ringkasan === 'string' && citation.metadata.ringkasan) && citation.content_snippet && (
                    <p className="text-sm text-slate-600 mt-2 line-clamp-3">
                      {citation.content_snippet}
                    </p>
                  )}
                </div>
                <div className="flex-shrink-0">
                  <div className="w-10 h-10 rounded-full bg-blue-50 flex items-center justify-center text-blue-700 font-bold text-sm">
                    {citationNumber}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
