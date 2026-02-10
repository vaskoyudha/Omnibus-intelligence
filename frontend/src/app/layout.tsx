import type { Metadata } from "next";
import { Geist_Mono } from "next/font/google";
import "@fontsource-variable/plus-jakarta-sans";
import "./globals.css";
import Navbar from "@/components/Navbar";
import PageTransition from "@/components/PageTransition";
import AmbientBackground from "@/components/AmbientBackground";
import { Toaster } from "sonner";

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Omnibus Legal Compass - Sistem Tanya Jawab Hukum Indonesia",
  description: "Sistem RAG (Retrieval-Augmented Generation) untuk menjawab pertanyaan tentang peraturan perundang-undangan Indonesia dengan kutipan sumber yang akurat.",
  keywords: ["hukum indonesia", "peraturan", "undang-undang", "legal", "omnibus", "tanya jawab hukum"],
  authors: [{ name: "Omnibus Legal Compass Team" }],
  openGraph: {
    title: "Omnibus Legal Compass",
    description: "Sistem Tanya Jawab Hukum Indonesia dengan AI",
    type: "website",
    locale: "id_ID",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="id" suppressHydrationWarning>
      <body className={`${geistMono.variable} antialiased`}>
        <AmbientBackground />
        <Navbar />
        <main className="pt-16 min-h-screen">
          <PageTransition>
            {children}
          </PageTransition>
        </main>
        <Toaster
          position="bottom-right"
          toastOptions={{
            style: {
              background: 'rgba(255, 255, 255, 0.92)',
              backdropFilter: 'blur(20px)',
              border: '1px solid rgba(255, 255, 255, 0.5)',
              boxShadow: '0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.04)',
            },
          }}
          richColors
          closeButton
        />
      </body>
    </html>
  );
}
