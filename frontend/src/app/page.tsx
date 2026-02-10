'use client';

import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import SearchBar from '@/components/SearchBar';
import AnswerCard from '@/components/AnswerCard';
import StreamingAnswerCard from '@/components/StreamingAnswerCard';
import SkeletonLoader from '@/components/SkeletonLoader';
import { 
  askQuestion, 
  askQuestionStream, 
  AskResponse, 
  CitationInfo, 
  ConfidenceScore, 
  ValidationResult 
} from '@/lib/api';

const exampleQuestions = [
  'Apa syarat pendirian PT?',
  'Bagaimana ketentuan PHK karyawan?',
  'Apa itu RUPS?',
  'Apa hak pekerja menurut UU Cipta Kerja?',
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

export default function Home() {
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Streaming state
  const [useStreaming, setUseStreaming] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingAnswer, setStreamingAnswer] = useState('');
  const [streamingCitations, setStreamingCitations] = useState<CitationInfo[]>([]);
  const [streamingConfidence, setStreamingConfidence] = useState<ConfidenceScore | null>(null);
  const [streamingValidation, setStreamingValidation] = useState<ValidationResult | null>(null);
  const [streamingProcessingTime, setStreamingProcessingTime] = useState(0);

  const handleSearch = useCallback(async (query: string) => {
    setError(null);
    setResponse(null);
    
    setStreamingAnswer('');
    setStreamingCitations([]);
    setStreamingConfidence(null);
    setStreamingValidation(null);
    setStreamingProcessingTime(0);
    
    if (useStreaming) {
      setIsStreaming(true);
      setIsLoading(false);
      
      try {
        await askQuestionStream(query, {
          onMetadata: (metadata) => {
            setStreamingCitations(metadata.citations);
            setStreamingConfidence(metadata.confidence_score);
          },
          onChunk: (text) => {
            setStreamingAnswer(prev => prev + text);
          },
          onDone: (data) => {
            setStreamingValidation(data.validation);
            setStreamingProcessingTime(data.processing_time_ms);
            setIsStreaming(false);
          },
          onError: (err) => {
            setError(err.message);
            setIsStreaming(false);
          },
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Terjadi kesalahan saat memproses pertanyaan Anda');
        setIsStreaming(false);
      }
    } else {
      setIsLoading(true);
      
      try {
        const result = await askQuestion(query);
        setResponse(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Terjadi kesalahan saat memproses pertanyaan Anda');
      } finally {
        setIsLoading(false);
      }
    }
  }, [useStreaming]);

  const showStreamingResult = useStreaming && (isStreaming || streamingAnswer);
  const showRegularResult = !useStreaming && response && !isLoading;

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
          <motion.h1
            className="text-hero text-gradient mb-3"
            variants={itemVariants}
          >
            Tanya Jawab Hukum
          </motion.h1>
          <motion.p
            className="text-lg text-text-secondary mb-1"
            variants={itemVariants}
          >
            Sistem Tanya Jawab Hukum Indonesia
          </motion.p>
          <motion.p
            className="text-sm text-text-muted"
            variants={itemVariants}
          >
            Didukung oleh AI untuk membantu Anda memahami peraturan perundang-undangan
          </motion.p>
        </div>
      </motion.div>

      {/* Search Section */}
      <motion.div
        className="max-w-4xl mx-auto px-4 -mt-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3, duration: 0.5, ease: [0.22, 1, 0.36, 1] as [number, number, number, number] }}
      >
        <div className="glass-strong rounded-2xl shadow-lg p-6">
          <SearchBar onSearch={handleSearch} isLoading={isLoading || isStreaming} />
          
          {/* Options Row */}
          <div className="mt-4 pt-4 border-t border-border flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div className="flex-1">
              <p className="text-sm text-text-muted mb-2.5">Contoh pertanyaan:</p>
              <div className="flex flex-wrap gap-2">
                {exampleQuestions.map((question, i) => (
                  <motion.button
                    key={question}
                    onClick={() => handleSearch(question)}
                    disabled={isLoading || isStreaming}
                    className="text-sm px-4 py-2 glass rounded-full text-text-secondary hover:text-accent hover:border-border-accent border border-transparent transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 + i * 0.05, duration: 0.4 }}
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                  >
                    {question}
                  </motion.button>
                ))}
              </div>
            </div>
            
            {/* Streaming Toggle */}
            <div className="flex items-center gap-2.5 no-print flex-shrink-0">
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={useStreaming}
                  onChange={(e) => setUseStreaming(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-10 h-5.5 bg-bg-tertiary peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-accent-light rounded-full peer peer-checked:after:translate-x-[18px] after:content-[''] after:absolute after:top-[3px] after:start-[3px] after:bg-white after:shadow-sm after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-accent" />
              </label>
              <span className="text-xs font-medium text-text-muted">
                Streaming {useStreaming ? 'On' : 'Off'}
              </span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Results Section */}
      <div className="max-w-4xl mx-auto px-4 py-8">
        {isLoading && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <SkeletonLoader variant="card" lines={5} />
          </motion.div>
        )}
        
        {error && (
          <motion.div
            className="glass-strong rounded-2xl border border-error/20 px-6 py-5"
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ type: 'spring', stiffness: 300, damping: 20 }}
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
        
        {/* Streaming Result */}
        {showStreamingResult && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <StreamingAnswerCard
              answer={streamingAnswer}
              citations={streamingCitations}
              confidenceScore={streamingConfidence}
              validation={streamingValidation}
              processingTimeMs={streamingProcessingTime}
              isStreaming={isStreaming}
            />
          </motion.div>
        )}
        
        {/* Regular Result */}
        {showRegularResult && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <AnswerCard response={response} />
          </motion.div>
        )}
        
        {/* Empty State */}
        {!response && !isLoading && !error && !showStreamingResult && (
          <motion.div
            className="text-center py-16"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.5 }}
          >
            <div className="w-20 h-20 mx-auto mb-5 glass-strong rounded-2xl flex items-center justify-center shadow-md">
              <svg className="w-10 h-10 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-text-primary mb-2">
              Tanyakan Pertanyaan Hukum Anda
            </h3>
            <p className="text-text-secondary max-w-md mx-auto leading-relaxed">
              Ketik pertanyaan tentang peraturan perundang-undangan Indonesia, 
              dan sistem akan mencari jawaban dari dokumen hukum yang relevan.
            </p>
          </motion.div>
        )}
      </div>

      {/* Footer */}
      <footer className="border-t border-border mt-auto">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <p className="text-center text-sm text-text-muted">
            Omnibus Legal Compass - Sistem RAG untuk Dokumen Hukum Indonesia
          </p>
          <p className="text-center text-xs text-text-muted/60 mt-1">
            Jawaban yang diberikan bersifat informatif dan bukan merupakan nasihat hukum resmi
          </p>
        </div>
      </footer>
    </div>
  );
}
