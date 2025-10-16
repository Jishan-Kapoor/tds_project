import os
import base64
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

load_dotenv()  # Load .env file

app = FastAPI()

# Env vars
MY_SECRET = os.getenv("MY_SECRET")
GITHUB_PAT = os.getenv("GITHUB_PAT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")

# Validate environment variables
if not all([MY_SECRET, GITHUB_PAT, OPENAI_API_KEY, GITHUB_USERNAME]):
    missing = [k for k, v in {"MY_SECRET": MY_SECRET, "GITHUB_PAT": GITHUB_PAT,
                              "OPENAI_API_KEY": OPENAI_API_KEY, "GITHUB_USERNAME": GITHUB_USERNAME}.items() if not v]
    logger.error(f"Missing required environment variables: {', '.join(missing)}")
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

logger.info(f"MY_SECRET: {MY_SECRET}")
logger.info(f"GITHUB_PAT loaded: {bool(GITHUB_PAT)}")
logger.info(f"OPENAI_API_KEY loaded: {bool(OPENAI_API_KEY)}")
logger.info(f"GITHUB_USERNAME: {GITHUB_USERNAME}")

# LLM client
llm_client = OpenAI(
    api_key=OPENAI_API_KEY
)

# MIT License text
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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
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
                    {"role": "system",
                     "content": "You are a code generator for simple static web apps. Output only the code or content requested, without extra explanations. Follow all requirements strictly."},
                    {"role": "user", "content": prompt}
                ],
                timeout=120
            )
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])
        logger.info(f"Generated content: {content[:100]}...")
        return content
    except Exception as e:
        logger.error(f"LLM error: {str(e)}")
        raise


def decode_attachment(attachment: dict) -> tuple[str, bytes]:
    name = attachment["name"]
    data_uri = attachment["url"]
    if not data_uri.startswith("data:"):
        logger.error(f"Invalid data URI for attachment: {name}")
        raise ValueError("Invalid data URI")
    try:
        header, encoded = data_uri.split(",", 1)
        data = base64.b64decode(encoded)
        logger.info(f"Decoded attachment: {name}")
        return name, data
    except Exception as e:
        logger.error(f"Error decoding attachment {name}: {str(e)}")
        raise ValueError(f"Error decoding attachment: {str(e)}")


def _build_code_prompt(brief: str, checks: list[str], attachment_names: list[str], round: int) -> str:
    is_csv_task = any(keyword in brief.lower() for keyword in ['csv', 'sales', 'sum'])
    is_github_task = 'github' in brief.lower() and 'user' in brief.lower()
    has_markdown = any(name.lower().endswith('.md') for name in attachment_names)
    is_ocr_task = any(keyword in brief.lower() for keyword in ['ocr', 'captcha', 'decode', 'recognize text'])

    prompt = f"""Generate a complete, functional index.html file for GitHub Pages that implements: {brief}

CRITICAL REQUIREMENTS:
1. Include all necessary inline CSS and JavaScript - no external JS files
2. Use CDN links for libraries (e.g., marked, highlight.js, tesseract.js) from jsdelivr
3. Implement ALL features mentioned in the brief
4. Handle attachments by referencing them in the same directory (e.g., fetch('input.md'), src='captcha.png')
5. Make it a complete, working application, not a placeholder
6. Ensure all JavaScript runs on window load or DOMContentLoaded
7. Do not hardcode any data; always fetch or load from files
8. Output ONLY the complete HTML code, no explanations or wrappers

MUST PASS THESE CHECKS:
{'\n'.join(f'- {check}' for check in checks)}
"""

    if is_csv_task:
        prompt += """
CSV PARSING - CRITICAL:
- Use robust parsing: trim whitespace, handle empty lines, check column existence
- Parse each value with parseFloat() and check with isNaN() before adding
- Handle trailing newlines, empty rows, missing columns gracefully
- Ensure final total is a valid number, not NaN
"""

    if is_github_task:
        prompt += """
GITHUB API HANDLING:
- Use fetch() with error handling
- Support optional ?token= query parameter for authentication
- Display dates in YYYY-MM-DD UTC format
- Parse created_at from API response correctly
- Handle API errors gracefully with user-friendly messages
"""

    if has_markdown:
        md_file = next((name for name in attachment_names if name.lower().endswith('.md')), 'input.md')
        prompt += f"""
MARKDOWN HANDLING - CRITICAL:
- Load marked from CDN: <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
- Load highlight.js from CDN: <script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@latest/build/highlight.min.js"></script>
- Load highlight.js CSS: <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@latest/build/styles/default.min.css">
- Fetch markdown: fetch('{md_file}').then(res => res.text()).then(text => {{
  const html = marked.parse(text);
  document.getElementById('markdown-output').innerHTML = html;
  hljs.highlightAll();
}})
- Run on DOMContentLoaded
- Ensure #markdown-output exists and renders the HTML including headers and code blocks with highlighting
"""

    if is_ocr_task:
        image_file = next((name for name in attachment_names if name.lower().endswith(('.png', '.jpg', '.jpeg'))), 'captcha.png')
        prompt += f"""
OCR/CAPTCHA HANDLING - CRITICAL:
- Load Tesseract.js from CDN: <script src="https://cdn.jsdelivr.net/npm/tesseract.js@6/dist/tesseract.min.js"></script>
- Use this exact code pattern:
async function runOCR() {{
  const worker = await Tesseract.createWorker('eng');
  try {{
    const {{ data: {{ text }} }} = await worker.recognize('{image_file}');
    window.decodedText = text.trim().replace(/\\s+/g, '');
    document.getElementById('result').textContent = `Decoded: ${{window.decodedText}}`;
  }} catch (err) {{
    document.getElementById('result').textContent = 'Error: ' + err.message;
  }} finally {{
    await worker.terminate();
  }}
}}
document.addEventListener('DOMContentLoaded', runOCR);
- Include <img src="{image_file}" alt="Image">
- Include <div id="result">Loading...</div>
- Ensure typeof window.decodedText === 'string' && window.decodedText.length > 0 if successful
"""

    prompt += "\nOutput ONLY the complete HTML code."

    return prompt


def _build_update_prompt(existing_code: str, brief: str, checks: list[str], task: str, attachment_names: list[str]) -> str:
    is_csv_task = any(keyword in brief.lower() for keyword in ['csv', 'sales', 'sum'])
    is_github_task = task.startswith("github-user-created")
    has_markdown = any(name.lower().endswith('.md') for name in attachment_names)
    is_ocr_task = any(keyword in brief.lower() for keyword in ['ocr', 'captcha', 'decode', 'recognize text'])

    prompt = f"""Here is the existing index.html code:

{existing_code}

UPDATE REQUIREMENTS:
1. Add/modify to implement this new brief: {brief}
2. KEEP all existing functionality working unless specified
3. Ensure it passes BOTH old checks AND new checks: {', '.join(checks)}
4. Make it a complete, working update
5. If new attachments are provided, reference them in the same directory and update the logic accordingly
6. Modify the existing code by adding the new features while preserving the existing logic and structure
7. Ensure all JavaScript runs on window load or DOMContentLoaded
8. Do not hardcode any data
9. Output ONLY the complete updated HTML code

CRITICAL REQUIREMENTS (same as initial):
- Include all necessary inline CSS and JavaScript - no external JS files
- Use CDN links for libraries
- Handle attachments by referencing them in the same directory
"""

    if is_csv_task:
        prompt += """
CSV PARSING - CRITICAL (if applicable):
- Use robust parsing with proper validation
- Check each value with parseFloat() and isNaN()
- Handle empty lines, missing columns, trailing newlines
- Ensure sums are valid numbers, not NaN
"""

    if is_github_task:
        prompt += """
GITHUB TASK - CRITICAL:
- Allow use of localStorage for caching if specified
- Maintain robust fetch() with error handling
- Support optional ?token= query parameter
- Ensure dates remain in YYYY-MM-DD UTC format
"""

    if has_markdown:
        prompt += """
MARKDOWN HANDLING (if applicable):
- Load marked from CDN if not already
- Use marked.parse() to convert markdown to HTML
- Load highlight.js and CSS if not already
- Call hljs.highlightAll() after setting HTML
- Render in #markdown-output
"""

    if is_ocr_task:
        prompt += """
OCR/CAPTCHA HANDLING - CRITICAL:
- Load Tesseract.js from CDN if not already
- Use createWorker and recognize as in initial prompt
- Set window.decodedText
- Run on load
"""

    return prompt


def _build_readme_prompt(brief: str, round: int) -> str:
    return f"""Write a professional, comprehensive README.md for a static web application that implements: {brief}

REQUIRED SECTIONS:
1. **Summary**: What the app does (2-3 sentences)
2. **Setup**: How to deploy on GitHub Pages (step-by-step)
3. **Usage**:
   - How to access the page
   - Any query parameters or configuration options
   - Key features and how to use them
4. **Code Explanation**:
   - Brief technical overview of the HTML/JS implementation
   - Key libraries used (if any)
   - Important algorithms or logic
5. **License**: State that it's MIT licensed

REQUIREMENTS:
- Professional tone and formatting
- Clear, concise language
- Use proper markdown formatting (headers, lists, code blocks)
- Include examples where helpful
- No placeholder text - make it specific to this application

Output ONLY the markdown content."""


async def create_or_update_repo(task: str, round: int, brief: str, checks: list[Union[str, Dict[str, str]]], attachments: list[dict]) -> dict:
    # Normalize checks
    normalized_checks = []
    for check in checks:
        if isinstance(check, str):
            normalized_checks.append(check)
        elif isinstance(check, dict) and "js" in check:
            normalized_checks.append(check["js"])
        else:
            raise ValueError("Invalid check format")

    g = Github(auth=Auth.Token(GITHUB_PAT))
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        try:
            user = await loop.run_in_executor(executor, g.get_user)
            repo_name = task
            repo = None

            if round == 1:
                try:
                    existing_repo = await loop.run_in_executor(executor, user.get_repo, repo_name)
                    await loop.run_in_executor(executor, existing_repo.delete)
                    await asyncio.sleep(5)
                    logger.info(f"Deleted existing repo: {repo_name}")
                except GithubException:
                    logger.info(f"No existing repo to delete")
                repo = await loop.run_in_executor(executor, lambda: user.create_repo(repo_name, private=False, auto_init=False))
                logger.info(f"Created repo: {repo_name}")
            else:
                repo = await loop.run_in_executor(executor, lambda: g.get_repo(f"{GITHUB_USERNAME}/{repo_name}"))
                logger.info(f"Found repo for update: {repo_name}")

            attachment_files = {}
            for att in attachments:
                name, content = decode_attachment(att)
                attachment_files[name] = content

            attachment_names = list(attachment_files.keys())

            if round == 1:
                llm_prompt = _build_code_prompt(brief, normalized_checks, attachment_names, round)
            else:
                existing_index = await loop.run_in_executor(executor, lambda: repo.get_contents("index.html"))
                existing_index_content = base64.b64decode(existing_index.content).decode('utf-8')
                llm_prompt = _build_update_prompt(existing_index_content, brief, normalized_checks, task, attachment_names)

            generated_code = await generate_with_llm(llm_prompt)

            readme_prompt = _build_readme_prompt(brief, round)
            if round > 1:
                try:
                    existing_readme = await loop.run_in_executor(executor, lambda: repo.get_contents("README.md"))
                    existing_readme_content = base64.b64decode(existing_readme.content).decode('utf-8')
                    readme_prompt = f"Existing README:\n\n{existing_readme_content}\n\nUpdate to include new features from brief: {brief}. Keep sections and update as needed. Output only updated markdown."
                except GithubException:
                    pass

            generated_readme = await generate_with_llm(readme_prompt)

            commit_msg = f"{'Initial build' if round == 1 else 'Update'} for round {round}"
            branch = "main"

            files_to_upload = {
                "index.html": generated_code.encode('utf-8'),
                "README.md": generated_readme.encode('utf-8'),
            }
            if round == 1:
                files_to_upload["LICENSE"] = MIT_LICENSE.format(year=time.strftime("%Y"), username=GITHUB_USERNAME).encode('utf-8')
            files_to_upload.update(attachment_files)

            for path, content in files_to_upload.items():
                try:
                    existing = await loop.run_in_executor(executor, lambda: repo.get_contents(path, ref=branch))
                    await loop.run_in_executor(executor, lambda: repo.update_file(path, commit_msg, content, existing.sha, branch=branch))
                    logger.info(f"Updated file: {path}")
                except GithubException:
                    await loop.run_in_executor(executor, lambda: repo.create_file(path, commit_msg, content, branch=branch))
                    logger.info(f"Created file: {path}")

            # Check if Pages enabled
            pages_check = await loop.run_in_executor(
                executor,
                lambda: requests.get(
                    f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/pages",
                    headers={"Authorization": f"token {GITHUB_PAT}", "Accept": "application/vnd.github.v3+json"}
                )
            )
            pages_url = f"https://{GITHUB_USERNAME}.github.io/{repo_name}/"
            if pages_check.status_code == 404:
                pages_response = await loop.run_in_executor(
                    executor,
                    lambda: requests.post(
                        f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/pages",
                        headers={"Authorization": f"token {GITHUB_PAT}", "Accept": "application/vnd.github.v3+json"},
                        json={"source": {"branch": branch, "path": "/"}}
                    )
                )
                if pages_response.status_code not in (201, 204):
                    raise Exception(f"Failed to enable Pages: {pages_response.text}")
                logger.info("Enabled GitHub Pages")
            else:
                pages_url = pages_check.json().get("html_url", pages_url)
                logger.info(f"GitHub Pages already enabled")

            # Wait for Pages to be ready
            max_retries = 15
            delay = 5
            for attempt in range(max_retries):
                try:
                    response = await loop.run_in_executor(executor, lambda: requests.get(pages_url, timeout=10))
                    if response.status_code == 200:
                        logger.info(f"Pages ready after {attempt + 1} attempts")
                        break
                except Exception as e:
                    logger.warning(f"Pages check error: {str(e)}")
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, 60)
            else:
                raise Exception("Pages failed to become ready")

            commit_sha = (await loop.run_in_executor(executor, lambda: repo.get_branch(branch))).commit.sha
            logger.info(f"Commit SHA: {commit_sha}")

            return {
                "repo_url": repo.html_url,
                "commit_sha": commit_sha,
                "pages_url": pages_url
            }
        except Exception as e:
            logger.error(f"Error in create_or_update_repo: {str(e)}")
            raise


async def notify_evaluation(evaluation_url: str, payload: dict):
    delay = 1
    max_attempts = 10
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        for attempt in range(max_attempts):
            try:
                response = await loop.run_in_executor(
                    executor,
                    lambda: requests.post(evaluation_url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
                )
                if response.status_code == 200:
                    logger.info(f"Notification successful")
                    return
                logger.warning(f"Notification failed: {response.status_code}")
            except Exception as e:
                logger.warning(f"Notification error: {str(e)}")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)
        raise Exception("Failed to notify evaluation URL")


async def process_background(req: RequestBody, start_time: float):
    try:
        async with asyncio.timeout(600):
            repo_details = await create_or_update_repo(req.task, req.round, req.brief, req.checks, req.attachments)
            payload = {
                "email": req.email,
                "task": req.task,
                "round": req.round,
                "nonce": req.nonce,
                "repo_url": repo_details["repo_url"],
                "commit_sha": repo_details["commit_sha"],
                "pages_url": repo_details["pages_url"]
            }
            await notify_evaluation(req.evaluation_url, payload)
            elapsed = time.time() - start_time
            logger.info(f"Request processed in {elapsed:.2f} seconds")
    except asyncio.TimeoutError:
        logger.error("Timeout exceeded")
        raise
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise


used_nonces = set()
nonce_lock = threading.Lock()


@app.post("/api-endpoint")
async def process_request(request: Request, background_tasks: BackgroundTasks):
    start_time = time.time()
    try:
        body = await request.json()
        req = RequestBody(**body)
        logger.info(f"Received request for task: {req.task}, round: {req.round}")
    except Exception as e:
        logger.error(f"Invalid JSON: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if req.secret != MY_SECRET:
        logger.error("Invalid secret")
        raise HTTPException(status_code=403, detail="Invalid secret")

    with nonce_lock:
        if req.nonce in used_nonces:
            logger.error(f"Duplicate nonce: {req.nonce}")
            raise HTTPException(status_code=400, detail="Duplicate request")
        used_nonces.add(req.nonce)

    background_tasks.add_task(process_background, req, start_time)
    return {"status": "success"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
