'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import SkeletonLoader from '@/components/SkeletonLoader';
import { getGuidance, GuidanceResponse } from '@/lib/api';

const BUSINESS_TYPES = [
  { value: 'PT', label: 'PT (Perseroan Terbatas)', description: 'Badan hukum dengan modal terbagi atas saham' },
  { value: 'CV', label: 'CV (Commanditaire Vennootschap)', description: 'Persekutuan komanditer dengan sekutu aktif dan pasif' },
  { value: 'UD', label: 'UD (Usaha Dagang)', description: 'Usaha perorangan tanpa badan hukum' },
  { value: 'Koperasi', label: 'Koperasi', description: 'Badan usaha dengan prinsip kekeluargaan' },
  { value: 'Yayasan', label: 'Yayasan', description: 'Badan hukum nirlaba untuk tujuan sosial' },
];

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1, delayChildren: 0.1 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 16 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] as [number, number, number, number] } },
};

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
    <div className="min-h-screen">
      {/* Hero Section */}
      <motion.div
        className="py-16 px-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <div className="max-w-4xl mx-auto text-center">
          <motion.h1 className="text-hero text-gradient mb-3" variants={itemVariants}>
            Panduan Pendirian Usaha
          </motion.h1>
          <motion.p className="text-lg text-text-secondary mb-1" variants={itemVariants}>
            Business Formation Guidance
          </motion.p>
          <motion.p className="text-sm text-text-muted" variants={itemVariants}>
            Dapatkan panduan langkah demi langkah untuk mendirikan badan usaha di Indonesia
          </motion.p>
        </div>
      </motion.div>

      {/* Form Section */}
      <motion.div
        className="max-w-4xl mx-auto px-4 -mt-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3, duration: 0.5, ease: [0.22, 1, 0.36, 1] as [number, number, number, number] }}
      >
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Business Type Selection */}
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-3">
              Jenis Badan Usaha <span className="text-error">*</span>
            </label>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {BUSINESS_TYPES.map((type) => (
                <motion.label
                  key={type.value}
                  className={`relative flex flex-col p-5 cursor-pointer glass rounded-2xl border-2 transition-all ${
                    businessType === type.value
                      ? 'border-accent bg-accent-lighter shadow-lg'
                      : 'border-transparent hover:border-accent-light hover:shadow-md'
                  }`}
                  whileHover={{ scale: 1.02, y: -2 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <input
                    type="radio"
                    name="businessType"
                    value={type.value}
                    checked={businessType === type.value}
                    onChange={(e) => setBusinessType(e.target.value)}
                    className="sr-only"
                  />
                  <span className="font-semibold text-text-primary">{type.label}</span>
                  <span className="text-xs text-text-muted mt-1.5 leading-relaxed">{type.description}</span>
                  {businessType === type.value && (
                    <motion.div
                      className="absolute top-3 right-3"
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: 'spring', stiffness: 500, damping: 15 }}
                    >
                      <svg className="w-5 h-5 text-accent" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                    </motion.div>
                  )}
                </motion.label>
              ))}
            </div>
          </div>

          {/* Optional Fields */}
          <div className="glass-strong rounded-2xl p-6">
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <label htmlFor="location" className="block text-sm font-medium text-text-secondary mb-2">
                  Lokasi (Opsional)
                </label>
                <input
                  id="location"
                  type="text"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  placeholder="Contoh: Jakarta, Surabaya, dll."
                  className="w-full px-4 py-3 glass rounded-xl border border-border focus:border-accent focus:shadow-glow transition-all text-text-primary placeholder-text-muted"
                />
              </div>
              <div>
                <label htmlFor="industry" className="block text-sm font-medium text-text-secondary mb-2">
                  Bidang Usaha (Opsional)
                </label>
                <input
                  id="industry"
                  type="text"
                  value={industry}
                  onChange={(e) => setIndustry(e.target.value)}
                  placeholder="Contoh: Teknologi, Perdagangan, dll."
                  className="w-full px-4 py-3 glass rounded-xl border border-border focus:border-accent focus:shadow-glow transition-all text-text-primary placeholder-text-muted"
                />
              </div>
            </div>
          </div>

          {/* Submit Button */}
          <motion.button
            type="submit"
            disabled={isLoading || !businessType}
            className="w-full py-3.5 px-6 bg-gradient-to-r from-accent to-accent-dark text-white font-semibold rounded-xl hover:shadow-lg hover:shadow-accent/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
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
          </motion.button>
        </form>
      </motion.div>

      {/* Results Section */}
      <div className="max-w-4xl mx-auto px-4 py-8">
        {isLoading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <SkeletonLoader variant="card" lines={5} />
          </motion.div>
        )}

        {error && (
          <motion.div
            className="glass-strong rounded-2xl border border-error/20 px-6 py-5"
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <div className="flex items-start gap-3">
              <div className="w-10 h-10 rounded-xl bg-red-50 flex items-center justify-center flex-shrink-0">
                <svg className="w-5 h-5 text-error" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div>
                <h3 className="font-semibold text-text-primary">Terjadi Kesalahan</h3>
                <p className="text-sm text-text-secondary mt-1">{error}</p>
              </div>
            </div>
          </motion.div>
        )}

        {response && !isLoading && (
          <motion.div
            className="space-y-6"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
          >
            {/* Summary Card */}
            <motion.div className="glass-strong rounded-2xl shadow-lg p-6" variants={itemVariants}>
              <div className="flex items-center gap-4 mb-4">
                <div className="w-14 h-14 rounded-2xl bg-accent-lighter flex items-center justify-center">
                  <svg className="w-8 h-8 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-xl font-bold text-text-primary">{response.business_type_name}</h2>
                  <p className="text-text-muted text-sm">Jenis Badan Usaha: {response.business_type}</p>
                </div>
              </div>

              <div className="bg-bg-secondary rounded-xl p-5">
                <h3 className="font-semibold text-text-primary mb-2 flex items-center gap-2">
                  <svg className="w-5 h-5 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Ringkasan
                </h3>
                <p className="text-text-secondary whitespace-pre-wrap leading-relaxed">{response.summary}</p>
              </div>

              {/* Key Info */}
              <div className="grid gap-4 sm:grid-cols-2 mt-4">
                <div className="bg-accent-lighter rounded-xl p-4 border border-accent-light">
                  <div className="flex items-center gap-2 text-accent font-semibold mb-1">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Estimasi Waktu
                  </div>
                  <p className="text-accent-dark font-medium">{response.total_estimated_time}</p>
                </div>
                <div className="bg-accent-lighter rounded-xl p-4 border border-accent-light">
                  <div className="flex items-center gap-2 text-accent font-semibold mb-1">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
                    </svg>
                    Jumlah Langkah
                  </div>
                  <p className="text-accent-dark font-medium">{response.steps.length} Langkah</p>
                </div>
              </div>

              <p className="text-xs text-text-muted mt-4 text-right">
                Waktu pemrosesan: {(response.processing_time_ms / 1000).toFixed(2)} detik
              </p>
            </motion.div>

            {/* Required Permits */}
            {response.required_permits && response.required_permits.length > 0 && (
              <motion.div className="glass-strong rounded-2xl shadow-lg p-6" variants={itemVariants}>
                <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
                  <svg className="w-5 h-5 text-warning" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                  </svg>
                  Izin yang Diperlukan
                </h3>
                <div className="flex flex-wrap gap-2">
                  {response.required_permits.map((permit, index) => (
                    <motion.span
                      key={index}
                      className="glass rounded-full px-4 py-1.5 text-sm font-medium text-accent border border-border-accent"
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: index * 0.05 }}
                    >
                      {permit}
                    </motion.span>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Steps Timeline */}
            {response.steps && response.steps.length > 0 && (
              <motion.div className="glass-strong rounded-2xl shadow-lg p-6" variants={itemVariants}>
                <h3 className="text-lg font-semibold text-text-primary mb-6 flex items-center gap-2">
                  <svg className="w-5 h-5 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                  Langkah-langkah Pendirian
                </h3>
                <div className="space-y-6">
                  {response.steps.map((step, index) => (
                    <motion.div
                      key={index}
                      className="relative pl-10"
                      initial={{ opacity: 0, x: -12 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.2 + index * 0.1 }}
                    >
                      {/* Timeline connector */}
                      {index < response.steps.length - 1 && (
                        <div className="absolute left-[14px] top-10 w-0.5 h-full bg-gradient-to-b from-accent to-accent-light" />
                      )}
                      {/* Step number circle */}
                      <div className="absolute left-0 top-0 w-7 h-7 rounded-full bg-gradient-to-br from-accent to-accent-dark text-white flex items-center justify-center text-sm font-bold shadow-md">
                        {step.step_number}
                      </div>
                      
                      <div className="glass rounded-2xl p-5 border border-border hover:shadow-md transition-shadow">
                        <h4 className="font-semibold text-text-primary mb-2">{step.title}</h4>
                        <p className="text-text-secondary text-sm mb-3 leading-relaxed">{step.description}</p>
                        
                        <div className="grid gap-3 sm:grid-cols-2">
                          <div className="flex items-start gap-2 text-sm">
                            <svg className="w-4 h-4 text-text-muted mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            <div>
                              <span className="text-text-muted">Waktu:</span>{' '}
                              <span className="text-text-primary font-medium">{step.estimated_time}</span>
                            </div>
                          </div>
                          <div className="flex items-start gap-2 text-sm">
                            <svg className="w-4 h-4 text-text-muted mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                            </svg>
                            <div>
                              <span className="text-text-muted">Instansi:</span>{' '}
                              <span className="text-text-primary font-medium">{step.responsible_agency}</span>
                            </div>
                          </div>
                        </div>

                        {/* Required Documents */}
                        {step.required_documents && step.required_documents.length > 0 && (
                          <div className="mt-3">
                            <p className="text-xs font-semibold text-text-muted uppercase mb-2">Dokumen Diperlukan:</p>
                            <ul className="space-y-1">
                              {step.required_documents.map((doc, docIndex) => (
                                <li key={docIndex} className="flex items-center gap-2 text-sm text-text-secondary">
                                  <svg className="w-4 h-4 text-success flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
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
                          <div className="mt-3 p-3 bg-amber-50 border border-amber-100 rounded-xl text-xs text-amber-700">
                            <strong>Catatan:</strong> {step.notes}
                          </div>
                        )}
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Citations */}
            {response.citations && response.citations.length > 0 && (
              <motion.div className="glass-strong rounded-2xl shadow-lg p-6" variants={itemVariants}>
                <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
                  <svg className="w-5 h-5 text-text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                  </svg>
                  Referensi Peraturan ({response.citations.length})
                </h3>
                <div className="space-y-3">
                  {response.citations.map((citation, index) => (
                    <div
                      key={index}
                      className="glass rounded-xl p-4 border border-border"
                    >
                      <div className="flex items-start gap-3">
                        <span className="w-6 h-6 rounded-full bg-bg-tertiary text-text-secondary flex items-center justify-center text-xs font-bold flex-shrink-0">
                          {citation.number}
                        </span>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm text-text-primary font-medium">{citation.citation_id}</p>
                          <p className="text-sm text-text-secondary mt-1">{citation.citation}</p>
                          <div className="flex items-center gap-2 mt-2">
                            <span className="text-xs px-2 py-0.5 bg-accent-lighter text-accent rounded font-medium">
                              Relevansi: {(citation.score * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </motion.div>
        )}

        {/* Empty State */}
        {!response && !isLoading && !error && (
          <motion.div
            className="text-center py-16"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <div className="w-20 h-20 mx-auto mb-5 glass-strong rounded-2xl flex items-center justify-center shadow-md">
              <svg className="w-10 h-10 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-text-primary mb-2">
              Panduan Pendirian Badan Usaha
            </h3>
            <p className="text-text-secondary max-w-md mx-auto leading-relaxed">
              Pilih jenis badan usaha yang ingin Anda dirikan untuk mendapatkan panduan 
              langkah demi langkah berdasarkan peraturan perundang-undangan Indonesia.
            </p>
          </motion.div>
        )}
      </div>

      {/* Footer */}
      <footer className="border-t border-border mt-auto">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <p className="text-center text-sm text-text-muted">
            Omnibus Legal Compass - Panduan Pendirian Usaha
          </p>
          <p className="text-center text-xs text-text-muted/60 mt-1">
            Panduan bersifat informatif dan bukan merupakan nasihat hukum resmi
          </p>
        </div>
      </footer>
    </div>
  );
}
