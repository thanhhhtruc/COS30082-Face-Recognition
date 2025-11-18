import type { Metadata } from 'next'
import { Analytics } from '@vercel/analytics/next'
import './globals.css'
import { Poppins } from 'next/font/google'

const poppins = Poppins({
  subsets: ['latin'],
  weight: ['400', '700'],
  variable: '--font-sans',
})

export const metadata: Metadata = {
  title: 'Facial Recognition with Emotion & Liveness',
  description: 'COS30082 - Applied Machine Learning Project',
  authors: [{ name: 'Tran Pham Thanh Truc - 104813707', url: 'https://github.com/thanhhhtruc/COS30082-Face-Recognition.git' }]
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={poppins.variable}>
      <body className="font-sans antialiased">
        {children}
        <Analytics />
      </body>
    </html>
  )
}
