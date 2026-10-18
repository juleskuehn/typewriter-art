import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Typewriter Art",
  description:
    "Convert photographs into typewriter art with multi-layer ASCII optimization",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.className} h-full antialiased`}>
      <body className="h-full bg-gray-100 text-gray-800">
        <TooltipProvider>{children}</TooltipProvider>
      </body>
    </html>
  );
}
