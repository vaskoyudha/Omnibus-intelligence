'use client';

import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import SkeletonLoader from '@/components/SkeletonLoader';
import CitationList from '@/components/CitationList';
import { checkCompliance, ComplianceResponse } from '@/lib/api';

type InputType = 'text' | 'pdf';

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

export default function CompliancePage() {
  const [inputType, setInputType] = useState<InputType>('text');
  const [businessDescription, setBusinessDescription] = useState('');
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [response, setResponse] = useState<ComplianceResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      if (inputType === 'pdf' && pdfFile) {
        const result = await checkCompliance('', pdfFile);
        setResponse(result);
      } else if (inputType === 'text' && businessDescription.trim()) {
        const result = await checkCompliance(businessDescription);
        setResponse(result);
      } else {
        throw new Error(inputType === 'pdf' 
          ? 'Silakan pilih file PDF terlebih dahulu' 
          : 'Silakan masukkan deskripsi kegiatan usaha');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Terjadi kesalahan saat memeriksa kepatuhan');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.type !== 'application/pdf') {
        setError('Hanya file PDF yang diperbolehkan');
        return;
      }
      setPdfFile(file);
      setError(null);
    }
  };

  const getRiskBadge = (riskLevel: string) => {
    const styles: Record<string, string> = {
      tinggi: 'bg-red-50 text-red-700 border-red-200',
      sedang: 'bg-amber-50 text-amber-700 border-amber-200',
      rendah: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    };
    const labels: Record<string, string> = {
      tinggi: 'Risiko Tinggi',
      sedang: 'Risiko Sedang',
      rendah: 'Risiko Rendah',
    };
    return (
      <span className={`px-3 py-1 text-sm font-semibold rounded-full border ${styles[riskLevel] || styles.sedang}`}>
        {labels[riskLevel] || riskLevel}
      </span>
    );
  };

  const getSeverityColor = (severity: string) => {
    const colors: Record<string, string> = {
      tinggi: 'text-red-700 bg-red-50 border-red-200',
      sedang: 'text-amber-700 bg-amber-50 border-amber-200',
      rendah: 'text-emerald-700 bg-emerald-50 border-emerald-200',
    };
    return colors[severity.toLowerCase()] || 'text-text-secondary bg-bg-secondary border-border';
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
            Pemeriksa Kepatuhan
          </motion.h1>
          <motion.p className="text-lg text-text-secondary mb-1" variants={itemVariants}>
            Compliance Checker untuk Kegiatan Usaha
          </motion.p>
          <motion.p className="text-sm text-text-muted" variants={itemVariants}>
            Analisis kepatuhan usaha Anda terhadap peraturan perundang-undangan Indonesia
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
        <div className="glass-strong rounded-2xl shadow-lg p-8">
          <form onSubmit={handleSubmit}>
            {/* Segmented Control */}
            <div className="flex justify-center mb-8">
              <div className="glass rounded-xl p-1 inline-flex">
                {(['text', 'pdf'] as const).map((type) => (
                  <button
                    key={type}
                    type="button"
                    onClick={() => setInputType(type)}
                    className={`relative px-6 py-2.5 rounded-lg text-sm font-medium transition-all ${
                      inputType === type
                        ? 'text-accent'
                        : 'text-text-secondary hover:text-text-primary'
                    }`}
                  >
                    {inputType === type && (
                      <motion.div
                        layoutId="compliance-tab"
                        className="absolute inset-0 bg-white shadow-md rounded-lg -z-10"
                        transition={{ type: 'spring', bounce: 0.15, duration: 0.5 }}
                      />
                    )}
                    <span className="flex items-center gap-2">
                      {type === 'text' ? (
                        <>
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                          Deskripsi Teks
                        </>
                      ) : (
                        <>
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                          </svg>
                          Upload PDF
                        </>
                      )}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Text Input */}
            <AnimatePresence mode="wait">
              {inputType === 'text' && (
                <motion.div
                  key="text"
                  className="mb-6"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.3 }}
                >
                  <label htmlFor="businessDescription" className="block text-sm font-medium text-text-secondary mb-2">
                    Deskripsi Kegiatan Usaha
                  </label>
                  <textarea
                    id="businessDescription"
                    value={businessDescription}
                    onChange={(e) => setBusinessDescription(e.target.value)}
                    placeholder="Jelaskan kegiatan usaha Anda secara detail, termasuk jenis usaha, struktur perusahaan, jumlah karyawan, dan aspek operasional lainnya..."
                    rows={6}
                    className="w-full px-4 py-3 glass rounded-xl border border-border focus:border-accent focus:shadow-glow transition-all resize-none text-text-primary placeholder-text-muted"
                  />
                </motion.div>
              )}

              {inputType === 'pdf' && (
                <motion.div
                  key="pdf"
                  className="mb-6"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.3 }}
                >
                  <label className="block text-sm font-medium text-text-secondary mb-2">
                    Upload Dokumen PDF
                  </label>
                  <motion.div 
                    className="glass rounded-xl border-2 border-dashed border-border p-10 text-center hover:border-accent transition-all cursor-pointer"
                    onClick={() => fileInputRef.current?.click()}
                    whileHover={{ scale: 1.005 }}
                    whileTap={{ scale: 0.995 }}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".pdf"
                      onChange={handleFileChange}
                      className="hidden"
                    />
                    <svg className="w-12 h-12 mx-auto text-text-muted mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    {pdfFile ? (
                      <div>
                        <div className="flex items-center justify-center gap-2">
                          <svg className="w-5 h-5 text-accent" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                          </svg>
                          <p className="text-accent font-medium">{pdfFile.name}</p>
                        </div>
                        <p className="text-sm text-text-muted mt-1">
                          {(pdfFile.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      </div>
                    ) : (
                      <div>
                        <p className="text-text-secondary font-medium">Klik untuk memilih file PDF</p>
                        <p className="text-sm text-text-muted mt-1">atau drag and drop di sini</p>
                      </div>
                    )}
                  </motion.div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Submit Button */}
            <motion.button
              type="submit"
              disabled={isLoading || (inputType === 'text' && !businessDescription.trim()) || (inputType === 'pdf' && !pdfFile)}
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
                  Memeriksa Kepatuhan...
                </>
              ) : (
                <>
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Periksa Kepatuhan
                </>
              )}
            </motion.button>
          </form>
        </div>
      </motion.div>

      {/* Results Section */}
      <div className="max-w-4xl mx-auto px-4 py-8">
        {isLoading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <SkeletonLoader variant="card" lines={4} />
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
            {/* Compliance Status Card */}
            <motion.div className="glass-strong rounded-2xl shadow-lg p-6" variants={itemVariants}>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-4">
                  {response.compliant ? (
                    <div className="w-14 h-14 rounded-2xl bg-emerald-50 flex items-center justify-center">
                      <svg className="w-8 h-8 text-success" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                  ) : (
                    <div className="w-14 h-14 rounded-2xl bg-red-50 flex items-center justify-center">
                      <svg className="w-8 h-8 text-error" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </div>
                  )}
                  <div>
                    <h2 className="text-xl font-bold text-text-primary">
                      {response.compliant ? 'Patuh' : 'Tidak Patuh'}
                    </h2>
                    <p className="text-text-muted text-sm">Status Kepatuhan</p>
                  </div>
                </div>
                {getRiskBadge(response.risk_level)}
              </div>

              {/* Summary */}
              <div className="bg-bg-secondary rounded-xl p-5 mt-4">
                <h3 className="font-semibold text-text-primary mb-2 flex items-center gap-2">
                  <svg className="w-5 h-5 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Ringkasan
                </h3>
                <p className="text-text-secondary whitespace-pre-wrap leading-relaxed">{response.summary}</p>
              </div>

              <p className="text-xs text-text-muted mt-4 text-right">
                Waktu pemrosesan: {(response.processing_time_ms / 1000).toFixed(2)} detik
              </p>
            </motion.div>

            {/* Issues Card */}
            {response.issues && response.issues.length > 0 && (
              <motion.div className="glass-strong rounded-2xl shadow-lg p-6" variants={itemVariants}>
                <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
                  <svg className="w-5 h-5 text-error" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  Masalah Ditemukan ({response.issues.length})
                </h3>
                <div className="space-y-3">
                  {response.issues.map((issue, index) => (
                    <motion.div
                      key={index}
                      className={`glass rounded-xl p-4 border ${getSeverityColor(issue.severity)}`}
                      initial={{ opacity: 0, x: -12 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.08 }}
                    >
                      <div className="flex items-start gap-3">
                        <span className="px-2 py-0.5 text-xs font-semibold rounded uppercase">
                          {issue.severity}
                        </span>
                        <div className="flex-1">
                          <p className="font-medium">{issue.issue}</p>
                          <p className="text-sm opacity-80 mt-1">
                            Regulasi: {issue.regulation}
                          </p>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Recommendations Card */}
            {response.recommendations && response.recommendations.length > 0 && (
              <motion.div className="glass-strong rounded-2xl shadow-lg p-6" variants={itemVariants}>
                <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
                  <svg className="w-5 h-5 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                  Rekomendasi ({response.recommendations.length})
                </h3>
                <ul className="space-y-3">
                  {response.recommendations.map((rec, index) => (
                    <motion.li
                      key={index}
                      className="flex items-start gap-3 text-text-secondary"
                      initial={{ opacity: 0, x: -12 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.08 }}
                    >
                      <span className="w-7 h-7 rounded-full bg-accent-light text-accent flex items-center justify-center text-sm font-semibold flex-shrink-0">
                        {index + 1}
                      </span>
                      <span className="pt-0.5">{rec}</span>
                    </motion.li>
                  ))}
                </ul>
              </motion.div>
            )}

            {/* Citations */}
            {response.citations && response.citations.length > 0 && (
              <motion.div className="glass-strong rounded-2xl shadow-lg overflow-hidden" variants={itemVariants}>
                <CitationList citations={response.citations} />
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
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-text-primary mb-2">
              Periksa Kepatuhan Usaha Anda
            </h3>
            <p className="text-text-secondary max-w-md mx-auto leading-relaxed">
              Masukkan deskripsi kegiatan usaha atau upload dokumen PDF untuk memeriksa 
              kepatuhan terhadap peraturan perundang-undangan Indonesia.
            </p>
          </motion.div>
        )}
      </div>

      {/* Footer */}
      <footer className="border-t border-border mt-auto">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <p className="text-center text-sm text-text-muted">
            Omnibus Legal Compass - Pemeriksa Kepatuhan Usaha
          </p>
          <p className="text-center text-xs text-text-muted/60 mt-1">
            Hasil pemeriksaan bersifat informatif dan bukan merupakan nasihat hukum resmi
          </p>
        </div>
      </footer>
    </div>
  );
}
