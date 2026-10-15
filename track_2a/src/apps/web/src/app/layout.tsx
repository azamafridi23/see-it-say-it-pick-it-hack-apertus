import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Pick Lab · Apertus Robot Playground",
  description: "See the scene. Say the task. Watch the robot act. A 2D Apertus playground.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
