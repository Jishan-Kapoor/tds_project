import os
import base64
import json
import time
import requests
import asyncio
from github import Github, GithubException, Auth
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from retry import retry
import logging
from concurrent.futures import ThreadPoolExecutor
import uvicorn
import threading
from typing import Union, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Env vars
MY_SECRET = os.getenv("MY_SECRET")
GITHUB_PAT = os.getenv("GITHUB_PAT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")

if not all([MY_SECRET, GITHUB_PAT, OPENAI_API_KEY, GITHUB_USERNAME]):
    missing = [k for k, v in {"MY_SECRET": MY_SECRET, "GITHUB_PAT": GITHUB_PAT,
                              "OPENAI_API_KEY": OPENAI_API_KEY, "GITHUB_USERNAME": GITHUB_USERNAME}.items() if not v]
    logger.error(f"Missing required environment variables: {', '.join(missing)}")
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

logger.info(f"GITHUB_USERNAME: {GITHUB_USERNAME}")

llm_client = OpenAI(api_key=OPENAI_API_KEY)

MIT_LICENSE = """
MIT License

Copyright (c) {year} {username}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""".strip()

class RequestBody(BaseModel):
    email: str
    secret: str
    task: str
    round: int
    nonce: str
    brief: str
    checks: list[Union[str, Dict[str, str]]]
    evaluation_url: str
    attachments: list[dict]

@retry(tries=3, delay=2, backoff=2, max_delay=10)
async def generate_with_llm(prompt: str) -> str:
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Output ONLY the complete code/content. NO explanations."},
                    {"role": "user", "content": prompt}
                ],
                timeout=120
            )
        )
        content = response.choices[0].message.content
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
        return content.strip()
    except Exception as e:
        logger.error(f"LLM error: {str(e)}")
        raise

def decode_attachment(attachment: dict) -> tuple[str, bytes]:
    name = attachment["name"]
    data_uri = attachment["url"]
    if not data_uri.startswith("data:"):
        raise ValueError("Invalid data URI")
    header, encoded = data_uri.split(",", 1)
    data = base64.b64decode(encoded)
    logger.info(f"Decoded attachment: {name}")
    return name, data

def _detect_task_type(brief: str, attachment_names: list[str]) -> str:
    brief_lower = brief.lower()
    if any(keyword in brief_lower for keyword in ['captcha', 'decode', 'ocr']):
        return 'captcha'
    elif any('.md' in name for name in attachment_names):
        return 'markdown'
    elif any(keyword in brief_lower for keyword in ['csv', 'sales', 'sum']):
        return 'csv'
    elif 'github' in brief_lower and 'user' in brief_lower:
        return 'github'
    return 'generic'

def _build_code_prompt(brief: str, checks: list[str], attachment_names: list[str], round: int) -> str:
    task_type = _detect_task_type(brief, attachment_names)
    image_name = next((name for name in attachment_names if name.endswith(('.png', '.jpg', '.jpeg'))), None)
    md_name = next((name for name in attachment_names if name.endswith('.md')), None)
    
    base_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>body{{font-family:Arial;margin:20px;}}</style>
</head>
<body>"""

    if task_type == 'captcha':
        prompt = f"""Generate COMPLETE index.html for CAPTCHA OCR: {brief}

MUST PASS CHECKS:
{chr(10).join(f'- {check}' for check in checks)}

EXACT REQUIREMENTS:
1. CDN: https://cdn.jsdelivr.net/npm/tesseract.js@6/dist/tesseract.min.js
2. Image: src="{image_name}"
3. Result: #result div
4. Global: window.decodedText

USE THIS EXACT HTML:
{base_html.format(title='CAPTCHA Decoder')}
<h1>CAPTCHA Decoder</h1>
<img src="{image_name}" style="max-width:300px;">
<div id="status">Loading...</div>
<div id="result"></div>
<script src="https://cdn.jsdelivr.net/npm/tesseract.js@6/dist/tesseract.min.js"></script>
<script>
window.addEventListener('load', async () => {{
  document.getElementById('status').textContent = 'Decoding...';
  const worker = await Tesseract.createWorker('eng');
  try {{
    const {{ data: {{ text }} }} = await worker.recognize('{image_name}');
    window.decodedText = text.trim().replace(/\\s/g, '');
    document.getElementById('result').innerHTML = `<h2>Decoded: ${{window.decodedText}}</h2>`;
  }} catch (e) {{
    document.getElementById('result').innerHTML = 'Error: ' + e.message;
  }} finally {{
    await worker.terminate();
    document.getElementById('status').textContent = 'Done';
  }}
}});
</script>
</body>
</html>

Output ONLY this complete HTML."""

    elif task_type == 'markdown':
        prompt = f"""Generate COMPLETE index.html for Markdown to HTML: {brief}

MUST PASS CHECKS:
{chr(10).join(f'- {check}' for check in checks)}

EXACT REQUIREMENTS:
1. CDN marked: https://cdn.jsdelivr.net/npm/marked/marked.min.js
2. CDN highlight.js: https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/lib/highlight.min.js
3. CDN highlight CSS: https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/styles/default.min.css
4. Load: fetch('{md_name}')
5. Render: #markdown-output

USE THIS EXACT HTML:
{base_html.format(title='Markdown Viewer')}
<h1>Markdown Viewer</h1>
<div id="markdown-output"></div>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/styles/default.min.css">
<script src="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/lib/highlight.min.js"></script>
<script>
hljs.highlightAll();
window.addEventListener('load', async () => {{
  const response = await fetch('{md_name}');
  const md = await response.text();
  document.getElementById('markdown-output').innerHTML = marked.parse(md);
  hljs.highlightAll();
}});
</script>
</body>
</html>

Output ONLY this complete HTML."""

    else:
        prompt = f"""Generate COMPLETE index.html for: {brief}

MUST PASS ALL CHECKS:
{chr(10).join(f'- {check}' for check in checks)}

Use attachments: {', '.join(attachment_names)}
Complete working app with inline CSS/JS + CDN libraries.
Auto-run on page load.

Output ONLY complete HTML."""

    return prompt

def _build_update_prompt(existing_code: str, brief: str, checks: list[str], task: str, attachment_names: list[str]) -> str:
    task_type = _detect_task_type(brief, attachment_names)
    if task_type == 'captcha':
        return f"""Update this CAPTCHA code:
{existing_code}

Keep Tesseract.js OCR for {attachment_names[0]}. Set window.decodedText.
Output ONLY complete HTML."""
    elif task_type == 'markdown':
        return f"""Update this Markdown code:
{existing_code}

Keep marked + highlight.js for {attachment_names[0]}. Render to #markdown-output.
Output ONLY complete HTML."""
    return f"""Update code for: {brief}
{existing_code}
Pass checks: {', '.join(checks)}
Output ONLY complete HTML."""

def _build_readme_prompt(brief: str, round: int) -> str:
    task_type = _detect_task_type(brief, [])
    if 'captcha' in brief.lower():
        return """# CAPTCHA Decoder

## Summary
Browser-based OCR to decode CAPTCHA images using Tesseract.js.

## Setup
1. Fork repo
2. Enable GitHub Pages

## Usage
Visit page - auto-decodes CAPTCHA

## Code Explanation
- Tesseract.js v6 for OCR
- Auto-runs on load
- Cleans whitespace

## License
MIT"""
    elif 'markdown' in brief.lower():
        return """# Markdown Viewer

## Summary
Converts input.md to HTML with syntax highlighting.

## Setup
1. Fork repo
2. Enable GitHub Pages

## Usage
Visit page - auto-renders markdown

## Code Explanation
- marked.js for Markdown
- highlight.js for code blocks
- Auto-fetch and render

## License
MIT"""
    return f"""# {brief.replace(' ', ' ').title()}

## Summary
Static web app for {brief}

## Setup
1. Fork repo
2. Enable GitHub Pages

## Usage
Visit the page

## Code Explanation
Complete HTML/JS implementation

## License
MIT"""

async def create_or_update_repo(task: str, round: int, brief: str, checks: list[Union[str, Dict[str, str]]], attachments: list[dict]) -> dict:
    normalized_checks = [check if isinstance(check, str) else check.get("js", "") for check in checks]
    
    g = Github(auth=Auth.Token(GITHUB_PAT))
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        user = await loop.run_in_executor(executor, g.get_user)
        repo_name = task.replace("/", "-")  # Safe repo name
        
        # Create/Update repo
        try:
            if round == 1:
                try:
                    existing_repo = await loop.run_in_executor(executor, user.get_repo, repo_name)
                    await loop.run_in_executor(executor, existing_repo.delete)
                    await asyncio.sleep(3)
                except: pass
            repo = await loop.run_in_executor(executor, lambda: user.create_repo(repo_name, private=False))
        except GithubException:
            repo = await loop.run_in_executor(executor, lambda: g.get_repo(f"{GITHUB_USERNAME}/{repo_name}"))

        # Process attachments
        attachment_files = {}
        for att in attachments:
            name, content = decode_attachment(att)
            attachment_files[name] = content
        attachment_names = list(attachment_files.keys())

        # Generate code
        llm_prompt = _build_code_prompt(brief, normalized_checks, attachment_names, round)
        generated_code = await generate_with_llm(llm_prompt)
        generated_readme = await generate_with_llm(_build_readme_prompt(brief, round))

        commit_msg = f"Round {round}"
        branch = "main"

        files_to_upload = {
            "index.html": generated_code.encode('utf-8'),
            "README.md": generated_readme.encode('utf-8'),
            "LICENSE": MIT_LICENSE.format(year=time.strftime("%Y"), username=GITHUB_USERNAME).encode('utf-8')
        }
        files_to_upload.update(attachment_files)

        # Upload files
        for path, content in files_to_upload.items():
            try:
                existing = await loop.run_in_executor(executor, lambda p=path: repo.get_contents(p, ref=branch))
                await loop.run_in_executor(executor, lambda: repo.update_file(path, commit_msg, content, existing.sha, branch=branch))
            except:
                await loop.run_in_executor(executor, lambda: repo.create_file(path, commit_msg, content, branch=branch))

        # Enable GitHub Pages
        try:
            await loop.run_in_executor(executor, lambda: requests.post(
                f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/pages",
                headers={"Authorization": f"token {GITHUB_PAT}"},
                json={"source": {"branch": branch, "path": "/"}}
            ))
        except: pass

        pages_url = f"https://{GITHUB_USERNAME}.github.io/{repo_name}/"
        
        # Wait for pages
        for i in range(20):
            try:
                resp = await loop.run_in_executor(executor, lambda: requests.get(pages_url, timeout=5))
                if resp.status_code == 200: break
            except: pass
            await asyncio.sleep(3)

        commit_sha = (await loop.run_in_executor(executor, lambda: repo.get_branch(branch))).commit.sha

        return {
            "repo_url": repo.html_url,
            "commit_sha": commit_sha, 
            "pages_url": pages_url
        }

async def notify_evaluation(evaluation_url: str, payload: dict):
    for i in range(5):
        try:
            resp = requests.post(evaluation_url, json=payload, timeout=10)
            if resp.status_code in (200, 201): return
        except: pass
        await asyncio.sleep(2)
    logger.error(f"Failed to notify: {evaluation_url}")

async def process_background(req: RequestBody, start_time: float):
    try:
        repo_details = await create_or_update_repo(req.task, req.round, req.brief, req.checks, req.attachments)
        payload = {
            "email": req.email, "task": req.task, "round": req.round, "nonce": req.nonce,
            "repo_url": repo_details["repo_url"], "commit_sha": repo_details["commit_sha"],
            "pages_url": repo_details["pages_url"]
        }
        await notify_evaluation(req.evaluation_url, payload)
        logger.info(f"SUCCESS in {time.time() - start_time:.1f}s")
    except Exception as e:
        logger.error(f"FAILED: {e}")
        raise

used_nonces = set()
nonce_lock = threading.Lock()

@app.post("/api-endpoint")
async def process_request(request: Request, background_tasks: BackgroundTasks):
    start_time = time.time()
    try:
        body = await request.json()
        req = RequestBody(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if req.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    with nonce_lock:
        if req.nonce in used_nonces:
            raise HTTPException(status_code=400, detail="Duplicate")
        used_nonces.add(req.nonce)

    background_tasks.add_task(process_background, req, start_time)
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
