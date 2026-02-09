'use client';

import { useState } from 'react';
import Link from 'next/link';
import LoadingSpinner from '@/components/LoadingSpinner';
import { getGuidance, GuidanceResponse } from '@/lib/api';

const BUSINESS_TYPES = [
  { value: 'PT', label: 'PT (Perseroan Terbatas)', description: 'Badan hukum dengan modal terbagi atas saham' },
  { value: 'CV', label: 'CV (Commanditaire Vennootschap)', description: 'Persekutuan komanditer dengan sekutu aktif dan pasif' },
  { value: 'UD', label: 'UD (Usaha Dagang)', description: 'Usaha perorangan tanpa badan hukum' },
  { value: 'Koperasi', label: 'Koperasi', description: 'Badan usaha dengan prinsip kekeluargaan' },
  { value: 'Yayasan', label: 'Yayasan', description: 'Badan hukum nirlaba untuk tujuan sosial' },
];

export default function GuidancePage() {
  const [businessType, setBusinessType] = useState('');
  const [location, setLocation] = useState('');
  const [industry, setIndustry] = useState('');
  const [response, setResponse] = useState<GuidanceResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!businessType) {
      setError('Silakan pilih jenis badan usaha');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      const result = await getGuidance({
        business_type: businessType,
        location: location || undefined,
        industry: industry || undefined,
      });
      setResponse(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Terjadi kesalahan saat mendapatkan panduan');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-700 to-emerald-800 text-white py-12 px-4">
        <div className="max-w-4xl mx-auto">
          <Link href="/" className="inline-flex items-center gap-2 text-emerald-200 hover:text-white mb-4 transition-colors">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Kembali ke Beranda
          </Link>
          <h1 className="text-4xl font-bold mb-2">Panduan Pendirian Usaha</h1>
          <p className="text-emerald-200 text-lg">Business Formation Guidance</p>
          <p className="text-emerald-300 text-sm mt-2">
            Dapatkan panduan langkah demi langkah untuk mendirikan badan usaha di Indonesia
          </p>
        </div>
      </div>

      {/* Form Section */}
      <div className="max-w-4xl mx-auto px-4 -mt-6">
        <div className="bg-white rounded-xl shadow-lg p-6">
          <form onSubmit={handleSubmit}>
            {/* Business Type Selection */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-700 mb-3">
                Jenis Badan Usaha <span className="text-red-500">*</span>
              </label>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {BUSINESS_TYPES.map((type) => (
                  <label
                    key={type.value}
                    className={`relative flex flex-col p-4 cursor-pointer rounded-lg border-2 transition-all ${
                      businessType === type.value
                        ? 'border-emerald-500 bg-emerald-50'
                        : 'border-slate-200 hover:border-emerald-300'
                    }`}
                  >
                    <input
                      type="radio"
                      name="businessType"
                      value={type.value}
                      checked={businessType === type.value}
                      onChange={(e) => setBusinessType(e.target.value)}
                      className="sr-only"
                    />
                    <span className="font-semibold text-slate-800">{type.label}</span>
                    <span className="text-xs text-slate-500 mt-1">{type.description}</span>
                    {businessType === type.value && (
                      <div className="absolute top-2 right-2">
                        <svg className="w-5 h-5 text-emerald-600" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                        </svg>
                      </div>
                    )}
                  </label>
                ))}
              </div>
            </div>

            {/* Optional Fields */}
            <div className="grid gap-4 sm:grid-cols-2 mb-6">
              <div>
                <label htmlFor="location" className="block text-sm font-medium text-slate-700 mb-2">
                  Lokasi (Opsional)
                </label>
                <input
                  id="location"
                  type="text"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  placeholder="Contoh: Jakarta, Surabaya, dll."
                  className="w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 text-slate-800 placeholder-slate-400"
                />
              </div>
              <div>
                <label htmlFor="industry" className="block text-sm font-medium text-slate-700 mb-2">
                  Bidang Usaha (Opsional)
                </label>
                <input
                  id="industry"
                  type="text"
                  value={industry}
                  onChange={(e) => setIndustry(e.target.value)}
                  placeholder="Contoh: Teknologi, Perdagangan, dll."
                  className="w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 text-slate-800 placeholder-slate-400"
                />
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading || !businessType}
              className="w-full py-3 px-4 bg-emerald-600 text-white font-semibold rounded-lg hover:bg-emerald-700 focus:ring-4 focus:ring-emerald-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Memproses Panduan...
                </>
              ) : (
                <>
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                  </svg>
                  Dapatkan Panduan
                </>
              )}
            </button>
          </form>
        </div>
      </div>

      {/* Results Section */}
      <div className="max-w-4xl mx-auto px-4 py-8">
        {isLoading && <LoadingSpinner />}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-6 py-4 rounded-lg">
            <div className="flex items-start gap-3">
              <svg className="w-5 h-5 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <div>
                <h3 className="font-medium">Terjadi Kesalahan</h3>
                <p className="text-sm mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}

        {response && !isLoading && (
          <div className="space-y-6">
            {/* Summary Card */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-center gap-4 mb-4">
                <div className="w-14 h-14 rounded-full bg-emerald-100 flex items-center justify-center">
                  <svg className="w-8 h-8 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-xl font-bold text-slate-800">{response.business_type_name}</h2>
                  <p className="text-slate-500 text-sm">Jenis Badan Usaha: {response.business_type}</p>
                </div>
              </div>

              <div className="bg-slate-50 rounded-lg p-4">
                <h3 className="font-semibold text-slate-700 mb-2 flex items-center gap-2">
                  <svg className="w-5 h-5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Ringkasan
                </h3>
                <p className="text-slate-600 whitespace-pre-wrap">{response.summary}</p>
              </div>

              {/* Key Info */}
              <div className="grid gap-4 sm:grid-cols-2 mt-4">
                <div className="bg-emerald-50 rounded-lg p-4 border border-emerald-200">
                  <div className="flex items-center gap-2 text-emerald-700 font-semibold mb-1">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Estimasi Waktu
                  </div>
                  <p className="text-emerald-800">{response.total_estimated_time}</p>
                </div>
                <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                  <div className="flex items-center gap-2 text-blue-700 font-semibold mb-1">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
                    </svg>
                    Jumlah Langkah
                  </div>
                  <p className="text-blue-800">{response.steps.length} Langkah</p>
                </div>
              </div>

              <p className="text-xs text-slate-400 mt-4 text-right">
                Waktu pemrosesan: {(response.processing_time_ms / 1000).toFixed(2)} detik
              </p>
            </div>

            {/* Required Permits */}
            {response.required_permits && response.required_permits.length > 0 && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
                  <svg className="w-5 h-5 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                  </svg>
                  Izin yang Diperlukan
                </h3>
                <div className="flex flex-wrap gap-2">
                  {response.required_permits.map((permit, index) => (
                    <span
                      key={index}
                      className="px-3 py-1 bg-amber-50 text-amber-700 border border-amber-200 rounded-full text-sm font-medium"
                    >
                      {permit}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Steps Timeline */}
            {response.steps && response.steps.length > 0 && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-semibold text-slate-800 mb-6 flex items-center gap-2">
                  <svg className="w-5 h-5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                  Langkah-langkah Pendirian
                </h3>
                <div className="space-y-6">
                  {response.steps.map((step, index) => (
                    <div key={index} className="relative pl-8">
                      {/* Timeline connector */}
                      {index < response.steps.length - 1 && (
                        <div className="absolute left-3 top-8 w-0.5 h-full bg-emerald-200" />
                      )}
                      {/* Step number circle */}
                      <div className="absolute left-0 top-0 w-6 h-6 rounded-full bg-emerald-600 text-white flex items-center justify-center text-sm font-bold">
                        {step.step_number}
                      </div>
                      
                      <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                        <h4 className="font-semibold text-slate-800 mb-2">{step.title}</h4>
                        <p className="text-slate-600 text-sm mb-3">{step.description}</p>
                        
                        <div className="grid gap-3 sm:grid-cols-2">
                          {/* Estimated Time */}
                          <div className="flex items-start gap-2 text-sm">
                            <svg className="w-4 h-4 text-slate-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            <div>
                              <span className="text-slate-500">Waktu:</span>{' '}
                              <span className="text-slate-700">{step.estimated_time}</span>
                            </div>
                          </div>
                          
                          {/* Responsible Agency */}
                          <div className="flex items-start gap-2 text-sm">
                            <svg className="w-4 h-4 text-slate-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                            </svg>
                            <div>
                              <span className="text-slate-500">Instansi:</span>{' '}
                              <span className="text-slate-700">{step.responsible_agency}</span>
                            </div>
                          </div>
                        </div>

                        {/* Required Documents */}
                        {step.required_documents && step.required_documents.length > 0 && (
                          <div className="mt-3">
                            <p className="text-xs font-semibold text-slate-500 uppercase mb-2">Dokumen Diperlukan:</p>
                            <ul className="space-y-1">
                              {step.required_documents.map((doc, docIndex) => (
                                <li key={docIndex} className="flex items-center gap-2 text-sm text-slate-600">
                                  <svg className="w-4 h-4 text-emerald-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                  </svg>
                                  {doc}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Notes */}
                        {step.notes && (
                          <div className="mt-3 p-2 bg-amber-50 border border-amber-100 rounded text-xs text-amber-700">
                            <strong>Catatan:</strong> {step.notes}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Citations */}
            {response.citations && response.citations.length > 0 && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
                  <svg className="w-5 h-5 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                  </svg>
                  Referensi Peraturan ({response.citations.length})
                </h3>
                <div className="space-y-3">
                  {response.citations.map((citation, index) => (
                    <div
                      key={index}
                      className="p-4 bg-slate-50 rounded-lg border border-slate-200"
                    >
                      <div className="flex items-start gap-3">
                        <span className="w-6 h-6 rounded-full bg-slate-200 text-slate-600 flex items-center justify-center text-xs font-bold flex-shrink-0">
                          {citation.number}
                        </span>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm text-slate-700 font-medium">{citation.citation_id}</p>
                          <p className="text-sm text-slate-600 mt-1">{citation.citation}</p>
                          <div className="flex items-center gap-2 mt-2">
                            <span className="text-xs px-2 py-0.5 bg-emerald-100 text-emerald-700 rounded">
                              Relevansi: {(citation.score * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Empty State */}
        {!response && !isLoading && !error && (
          <div className="text-center py-12">
            <div className="w-16 h-16 mx-auto mb-4 bg-emerald-100 rounded-full flex items-center justify-center">
              <svg className="w-8 h-8 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              Panduan Pendirian Badan Usaha
            </h3>
            <p className="text-gray-500 max-w-md mx-auto">
              Pilih jenis badan usaha yang ingin Anda dirikan untuk mendapatkan panduan 
              langkah demi langkah berdasarkan peraturan perundang-undangan Indonesia.
            </p>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="border-t border-gray-200 mt-auto">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <p className="text-center text-sm text-gray-500">
            Omnibus Legal Compass - Panduan Pendirian Usaha
          </p>
          <p className="text-center text-xs text-gray-400 mt-1">
            Panduan bersifat informatif dan bukan merupakan nasihat hukum resmi
          </p>
        </div>
      </footer>
    </main>
  );
}
