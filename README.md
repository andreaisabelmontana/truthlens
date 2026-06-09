# TruthLens — Interactive Showcase

A from-scratch static showcase site for **TruthLens**, a BCSAI hackathon project:
an AI-powered note-taking workspace that **fact-checks study notes sentence-by-sentence**,
returning corrections with reasoning and sources.

🔗 **Live site:** https://andreaisabelmontana.github.io/truthlens/

## What the original project does
- **Sentence-level fact checking** — inspects every sentence and flags weak or incorrect claims.
- **AI-powered** — uses local LLMs via **Ollama** (gpt-oss:20b) for grounded corrections.
- **Source citations** — every correction links to trustworthy references.
- **Learning workspace** — dashboard to manage documents, view analysis, and track corrections.
- **Secure** — user authentication with per-user workspaces.

**Stack:** Next.js 16 / React 19 / TypeScript / Tailwind (frontend) · Django 5 + DRF / PostgreSQL (backend) · Ollama + PyTorch + Transformers (AI) · Docker Compose (devops).

Submitted to the **Automate Learning** hackathon track (+ GitBook Docs & Cline CLI bonus tracks).

## About this repo
An **original, hand-built static site** (single `index.html`, no framework, no fork) presenting the
project, with a scripted interactive fact-check demo. It is not a reimplementation of TruthLens itself.

Original source code: [sofiagzzloz/TruthLens](https://github.com/sofiagzzloz/TruthLens).
