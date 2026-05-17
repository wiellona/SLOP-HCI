// app/layout.tsx
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './global.css';


const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'SLOP - Sign Language Output Provider',
  description: 'AI-powered sign language translation system for cafe service',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="id">
      <body className={inter.className}>
        <div className="flex flex-col h-screen">
          <div className="flex-1 overflow-hidden">{children}</div>
        </div>
      </body>
    </html>
  );
}