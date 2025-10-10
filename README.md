---
title: TDS Project API Builder
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: static
sdk_version: "0.0.1"
app_file: main.py
pinned: false
---

# TDS Project - LLM-Assisted GitHub Pages Builder

## Summary
This project is a FastAPI application that automates building, updating, and deploying static web apps to GitHub Pages.  
It uses an LLM (via AiPipe/OpenAI-compatible API) to generate HTML/JS code based on a JSON request specifying the app brief, required features, and attachments.  

It is designed to handle multiple rounds of requests:
- **Round 1:** Create the repository, generate the app, deploy to GitHub Pages, and notify evaluation endpoint.  
- **Round 2+:** Update the existing app, regenerate code based on new brief, redeploy, and notify evaluation endpoint.

---

## Setup

1. **Clone the repository** (already pushed to GitHub and Hugging Face):
```bash
git clone https://github.com/Jishan-Kapoor/tds_project.git
cd tds_project
