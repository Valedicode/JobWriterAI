import type { Metadata } from "next";
import { Libre_Franklin } from "next/font/google";
import { Libre_Caslon_Text } from "next/font/google";
import { GeistMono } from "geist/font/mono";
import "./globals.css";

const sans = Libre_Franklin({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-sans-user",
});

const serif = Libre_Caslon_Text({
  subsets: ["latin"],
  weight: ["400", "700"],
  style: ["normal", "italic"],
  display: "swap",
  variable: "--font-serif-user",
});

export const metadata: Metadata = {
  title: "JobWriterAI",
  description:
    "Turn a generic resume and a job posting into a tailored resume and cover letter — reviewed and approved step by step.",
};

// Resolve the theme before first paint so the ink/paper palette never flashes.
const themeScript = `(function(){try{var t=localStorage.getItem('theme');var d=t==='dark'||(!t&&window.matchMedia('(prefers-color-scheme: dark)').matches);var c=document.documentElement.classList;c.toggle('dark',d);c.toggle('light',!d);}catch(e){}})();`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body
        className={`${sans.variable} ${serif.variable} ${GeistMono.variable} antialiased`}
      >
        {/*
          IMPECCABLE DIRECTION CONTRACT — JobWriterAI redesign · surface: landing / entry
          Source: user-pinned Figma "Concept B — split showcase" (file 0gOzPfmywb5W8tFJ11Z0BP). No concept roll; the user authored and chose the comp.
          THESIS: Prove the human-in-the-loop gate by showing one working in the hero, not by describing a "tailor my resume" button. Refuses the centered-hero-plus-icon-card-grid entry.
          OWN-WORLD: Warm editorial — paper/sand grounds, warm near-black ink, one terracotta accent that carries CTAs, links and the second half of the headline. Libre Caslon Text display + wordmark, Libre Franklin for body/UI. Soft 16px cards, hairline warm borders, one low warm shadow + slight rotation on the layered gate card. Drawn line icons only.
          STORY: A student lands, reads "you see every change before it's yours", sees a real compatibility gate (0.78, Approve/Edit/Reject), and either starts with their CV or picks one of three named flows.
          FIRST VIEWPORT: Split — serif headline + subcopy + one terracotta pill CTA left (~55%); layered gate-preview card + "nothing written until you say so" chip right. "Pre-launch" tag in the nav, top right.
          FORM: split showcase (user's pick from their own comp set). Seed key: n/a (pinned direction).
          FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
        */}
        {children}
      </body>
    </html>
  );
}
