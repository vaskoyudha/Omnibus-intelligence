'use client';

import { AskResponse } from '@/lib/api';
import CitationList from './CitationList';

interface AnswerCardProps {
  response: AskResponse;
}

export default function AnswerCard({ response }: AnswerCardProps) {
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.7) return 'text-green-600 bg-green-50';
    if (confidence >= 0.4) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.7) return 'Tinggi';
    if (confidence >= 0.4) return 'Sedang';
    return 'Rendah';
  };

  return (
    <div className="w-full max-w-4xl mx-auto mt-8">
      {/* Answer Section */}
      <div className="bg-white rounded-xl shadow-md border border-slate-200 overflow-hidden">
        <div className="bg-gradient-to-r from-blue-700 to-blue-800 px-6 py-4">
          <h2 className="text-white font-semibold text-lg flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            Jawaban
          </h2>
        </div>
        
        <div className="p-6">
          {/* Metadata */}
          <div className="flex items-center gap-4 mb-4 text-sm">
            <span className={`px-3 py-1 rounded-full font-medium ${getConfidenceColor(response.confidence)}`}>
              Keyakinan: {getConfidenceLabel(response.confidence)} ({Math.round(response.confidence * 100)}%)
            </span>
            <span className="text-slate-400">
              Waktu proses: {response.processing_time.toFixed(2)}s
            </span>
          </div>

          {/* Answer Text */}
          <div className="prose prose-slate max-w-none">
            <div className="text-slate-700 leading-relaxed whitespace-pre-wrap">
              {response.answer}
            </div>
          </div>

          {/* Citations */}
          <CitationList citations={response.citations} />
        </div>
      </div>

      {/* Disclaimer */}
      <div className="mt-4 p-4 bg-amber-50 border border-amber-200 rounded-lg">
        <p className="text-sm text-amber-800 flex items-start gap-2">
          <svg className="w-5 h-5 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span>
            <strong>Peringatan:</strong> Informasi ini bersifat informatif dan bukan merupakan nasihat hukum resmi. 
            Untuk keperluan hukum yang mengikat, silakan konsultasikan dengan profesional hukum yang berkualifikasi.
          </span>
        </p>
      </div>
    </div>
  );
}
