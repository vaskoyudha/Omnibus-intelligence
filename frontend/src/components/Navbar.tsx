'use client';

import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { motion } from 'framer-motion';

const navLinks = [
  { href: '/', label: 'Tanya Jawab', icon: 'M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z' },
  { href: '/compliance', label: 'Kepatuhan', icon: 'M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z' },
  { href: '/guidance', label: 'Panduan Usaha', icon: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01' },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <motion.header
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      className="fixed top-0 left-0 right-0 z-50 glass-strong no-print"
    >
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <motion.div
            className="w-9 h-9 rounded-xl bg-gradient-to-br from-[#AAFF00] to-[#88CC00] flex items-center justify-center shadow-md shadow-[#AAFF00]/20"
            whileHover={{ scale: 1.05, rotate: -3 }}
            whileTap={{ scale: 0.95 }}
            transition={{ type: 'spring', stiffness: 400, damping: 15 }}
          >
            <svg className="w-5 h-5 text-[#0A0A0F]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
          </motion.div>
          <span className="font-bold text-lg text-text-primary tracking-tight hidden sm:block">
            Legal Compass
          </span>
        </Link>

        {/* Navigation Links */}
        <nav className="flex items-center gap-1">
          {navLinks.map((link) => {
            const isActive = pathname === link.href;
            return (
              <Link key={link.href} href={link.href}>
                <motion.div
                  className={`relative px-4 py-2 rounded-xl text-sm font-medium transition-colors flex items-center gap-2 ${isActive
                      ? 'text-[#AAFF00]'
                      : 'text-text-secondary hover:text-text-primary'
                    }`}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <svg
                    className={`w-4 h-4 hidden sm:block ${isActive ? 'text-[#AAFF00]' : 'text-text-muted'}`}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={link.icon} />
                  </svg>
                  {link.label}
                  {isActive && (
                    <motion.div
                      layoutId="nav-indicator"
                      className="absolute inset-0 bg-[#AAFF00]/10 border border-[#AAFF00]/20 rounded-xl -z-10"
                      transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
                    />
                  )}
                </motion.div>
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Bottom border with subtle gradient */}
      <div className="h-px bg-gradient-to-r from-transparent via-[rgba(255,255,255,0.08)] to-transparent" />
    </motion.header>
  );
}
