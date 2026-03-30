import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'SAVAGE22 Terminal',
  description: 'Crypto Trading Analysis Platform - Numerology, Astrology, Technical Analysis & Tweet Decoding',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gradient-dark grid-pattern antialiased">
        {children}
      </body>
    </html>
  );
}
