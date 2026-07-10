"""
Señal v2: morning AI signal for a practitioner-operator.

Pulls high-signal RSS feeds in parallel, skips already-seen links, optionally
enriches thin summaries with article text, scores via GLM for a personal profile,
and emails a tight HTML brief (day brief + top picks + why it matters).

Designed for local runs and GitHub Actions.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html as html_lib
import json
import logging
import os
import re
import smtplib
import sys
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional local dep
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


def load_local_env(path: Path) -> None:
    """Load local .env values for development, with no dependency required."""
    loaded = load_dotenv(path)
    if loaded or not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


load_local_env(Path(__file__).with_name(".env"))

# --- CONFIG ---

USER_AGENT = "Senal/2.0 (+https://github.com/gilerojas/Pulse; personal digest bot)"

# Curated for signal density. Dead / low-yield feeds removed after v1 health checks.
AI_FEEDS: list[tuple[str, str]] = [
    # Practitioners & operators
    ("https://simonwillison.net/atom/everything/", "Simon Willison"),
    ("https://www.oneusefulthing.org/feed", "One Useful Thing"),
    ("https://www.latent.space/feed", "Latent Space"),
    ("https://sebastianraschka.com/rss_feed.xml", "Sebastian Raschka"),
    ("https://lilianweng.github.io/index.xml", "Lilian Weng"),
    ("https://blog.eleuther.ai/index.xml", "EleutherAI"),
    # Labs & platforms
    ("https://openai.com/blog/rss.xml", "OpenAI"),
    ("https://deepmind.google/blog/rss.xml", "Google DeepMind"),
    ("https://blog.google/technology/ai/rss/", "Google AI"),
    ("https://huggingface.co/blog/feed.xml", "Hugging Face"),
    ("https://feeds.feedburner.com/blogspot/gJZg", "Google Research"),
    ("https://www.microsoft.com/en-us/research/feed/", "Microsoft Research"),
    # Curated newsletters / digests
    ("https://importai.substack.com/feed", "Import AI"),
    ("https://lastweekin.ai/feed", "Last Week in AI"),
    # News & forums (noisier — model filters hard)
    ("https://www.technologyreview.com/feed/", "MIT Technology Review"),
    ("https://hnrss.org/frontpage", "Hacker News"),
    ("https://www.reddit.com/r/MachineLearning/.rss", "r/MachineLearning"),
    ("https://www.reddit.com/r/LocalLLaMA/.rss", "r/LocalLLaMA"),
]

SYSTEM_PROMPT = """
You are Señal — a sharp content intelligence brief for Gilberto Rojas.

Profile (use this; do not invent more biography):
- COO in Santo Domingo, Dominican Republic, transitioning toward CEO
- Runs companies in paint / chemical manufacturing and operations
- MSc in Data Science & Deep Learning; deploys AI in real production systems, not demos
- Reads as a practitioner-operator: wants leverage, not hype or generic “AI is changing everything”

What he cares about most (in order):
1. Model / capability breakthroughs that change what is actually possible
2. AI in real operations — deployment, reliability, cost, people, process, failure modes
3. Tooling, agents, infrastructure, and APIs he could use or learn from
4. Industry dynamics, policy, and safety when they have practical consequences

Your job:
- From the candidate list, pick the TOP items worth his morning attention.
- Hard filter: only AI / ML / agents / model-infra / applied-AI-ops items. Pure general software (language runtimes, non-AI frameworks, unrelated startups) is out even if trending on HN.
- Prefer significance + practical leverage over clickbait, rumor, or pure gossip.
- Prefer primary sources and credible practitioners over SEO recaps when both exist.
- If several items are the same story, keep the best one only.
- Be honest when a summary is thin: say what is known, not what you invent.
- Write for a busy executive who already knows the AI basics.
- Bias toward things that change decisions: cost, capability, deployment risk, tooling he might adopt, or industry moves with operational consequences.

Scoring (0–10):
- 9–10: rare, must-read today for an AI-operating executive
- 7–8: high value for a practitioner-operator
- 5–6: solid but skippable if busy
- below 5: do not include

Return ONLY valid JSON matching the schema in the user message. No markdown fences, no preamble.
"""

MAX_ARTICLES = 50
DAYS_BACK = 3
FALLBACK_DAYS = 7
TOP_N = 5
ENRICH_CANDIDATES = 12
ENRICH_MIN_SUMMARY_CHARS = 280
ENRICH_MAX_CHARS = 1800
SEEN_RETENTION_DAYS = 45
DEFAULT_GLM_MODEL = "glm-5.2"
DEFAULT_GLM_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
DATA_DIR = Path(__file__).with_name("data")
SEEN_PATH = DATA_DIR / "seen_articles.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("senal")


# --- CLI / CONFIG ---


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and email the Señal AI digest.")
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Validate required configuration without calling feeds, GLM, or SMTP.",
    )
    parser.add_argument(
        "--send-test-email",
        action="store_true",
        help="Send a simple SMTP test email and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect and score articles, but do not send email or update memory.",
    )
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip full-text enrichment of thin RSS summaries.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Ignore and do not update seen-article memory.",
    )
    return parser.parse_args(argv)


def load_runtime_config(
    require_llm: bool = True,
    require_smtp: bool = True,
) -> dict[str, str]:
    gmail_user = os.environ.get("GMAIL_USER", "").strip()
    # Google app passwords are often copied with spaces; SMTP accepts either, but
    # stripping is the more reliable form across clients and GitHub secrets.
    gmail_password = re.sub(r"\s+", "", os.environ.get("GMAIL_APP_PASSWORD", "").strip())
    glm_api_key = (
        os.environ.get("GLM_API_KEY", "").strip()
        or os.environ.get("ZAI_API_KEY", "").strip()
        or os.environ.get("Z_AI_API_KEY", "").strip()
    )
    config = {
        "glm_api_key": glm_api_key,
        "gmail_user": gmail_user,
        "gmail_password": gmail_password,
        "email_to": os.environ.get("EMAIL_TO", "").strip() or gmail_user,
        "glm_model": os.environ.get("GLM_MODEL", "").strip() or DEFAULT_GLM_MODEL,
        "glm_base_url": os.environ.get("GLM_BASE_URL", "").strip() or DEFAULT_GLM_BASE_URL,
        "glm_timeout": os.environ.get("GLM_TIMEOUT", "").strip() or "180",
    }

    required: list[tuple[str, str]] = []
    if require_llm:
        required.append(("GLM_API_KEY", config["glm_api_key"]))
    if require_smtp:
        required.extend([
            ("GMAIL_USER", config["gmail_user"]),
            ("GMAIL_APP_PASSWORD", config["gmail_password"]),
        ])
    missing = [name for name, value in required if not value]
    if missing:
        raise SystemExit(f"Missing env: set {', '.join(missing)}.")
    return config


def log_health_check(config: dict[str, str]) -> None:
    logger.info("Config OK (Señal v2)")
    logger.info("GLM model: %s", config["glm_model"])
    logger.info("GLM base URL: %s", config["glm_base_url"])
    logger.info("GLM API key configured: %s", bool(config["glm_api_key"]))
    logger.info("Gmail user configured: %s", bool(config["gmail_user"]))
    logger.info("Email recipient: %s", config["email_to"])
    logger.info("Feeds configured: %d", len(AI_FEEDS))
    logger.info("Seen memory path: %s", SEEN_PATH)


# --- TEXT / URL HELPERS ---


def strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_url(url: str) -> str:
    """Collapse trivial URL variants so memory/dedup work better."""
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        path = parsed.path.rstrip("/") or "/"
        # Drop common tracking params
        query_parts = []
        if parsed.query:
            for part in parsed.query.split("&"):
                key = part.split("=", 1)[0].lower()
                if key.startswith("utm_") or key in {"ref", "source", "fbclid", "gclid"}:
                    continue
                query_parts.append(part)
        query = "&".join(query_parts)
        return urlunparse((parsed.scheme.lower() or "https", host, path, "", query, ""))
    except Exception:
        return url.strip()


def truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def article_fingerprint(title: str, url: str) -> str:
    base = f"{normalize_url(url)}|{title.strip().lower()}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


# --- MEMORY ---


def load_seen() -> dict[str, Any]:
    if not SEEN_PATH.exists():
        return {"version": 1, "articles": {}}
    try:
        data = json.loads(SEEN_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"version": 1, "articles": {}}
        data.setdefault("version", 1)
        data.setdefault("articles", {})
        return data
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not load seen memory (%s); starting fresh", e)
        return {"version": 1, "articles": {}}


def prune_seen(seen: dict[str, Any], retention_days: int = SEEN_RETENTION_DAYS) -> dict[str, Any]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    articles = seen.get("articles") or {}
    kept: dict[str, Any] = {}
    for key, meta in articles.items():
        sent_at = (meta or {}).get("sent_at")
        try:
            dt = datetime.fromisoformat(sent_at.replace("Z", "+00:00")) if sent_at else None
        except (TypeError, ValueError, AttributeError):
            dt = None
        if dt is None or dt >= cutoff:
            kept[key] = meta
    seen["articles"] = kept
    return seen


def save_seen(seen: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prune_seen(seen)
    SEEN_PATH.write_text(json.dumps(seen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Updated seen memory: %d entries → %s", len(seen.get("articles", {})), SEEN_PATH)


def mark_sent(seen: dict[str, Any], top_articles: list[dict], day_label: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    articles = seen.setdefault("articles", {})
    for a in top_articles:
        url = normalize_url(a.get("url") or "")
        title = a.get("title") or ""
        if not url and not title:
            continue
        key = normalize_url(url) or article_fingerprint(title, url)
        articles[key] = {
            "title": title,
            "url": url,
            "score": a.get("score"),
            "sent_at": now,
            "day": day_label,
        }


def filter_unseen(articles: list[dict], seen: dict[str, Any]) -> list[dict]:
    known = seen.get("articles") or {}
    out = []
    skipped = 0
    for a in articles:
        url_key = normalize_url(a.get("link") or "")
        if url_key and url_key in known:
            skipped += 1
            continue
        out.append(a)
    if skipped:
        logger.info("Memory filter: skipped %d previously sent articles", skipped)
    return out


# --- FEEDS ---


def parse_feed_date(entry: Any) -> datetime | None:
    for attr in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, attr, None)
        if parsed is None:
            continue
        try:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
    return None


def fetch_feed(url: str, source_name: str) -> list[dict]:
    """Fetch one RSS/Atom feed. Never raises."""
    import feedparser
    import requests

    headers = {"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*"}
    content: bytes | None = None
    try:
        r = requests.get(url, timeout=20, headers=headers)
        r.raise_for_status()
        content = r.content
    except Exception as e:
        logger.error("Feed HTTP failed: %s — %s", source_name, e)
        try:
            parsed = feedparser.parse(url, request_headers=headers)
        except Exception as e2:
            logger.error("Feed parse fallback failed: %s — %s", source_name, e2)
            return []
    else:
        parsed = feedparser.parse(content)

    if getattr(parsed, "bozo", False) and not getattr(parsed, "entries", None):
        logger.error("Feed unusable (bozo, no entries): %s", source_name)
        return []

    entries = getattr(parsed, "entries", None) or []
    if not entries:
        logger.warning("Feed empty: %s", source_name)
        return []

    feed_title = getattr(getattr(parsed, "feed", None), "title", None) or source_name

    def build(cutoff_days: int) -> list[dict]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=cutoff_days)
        results: list[dict] = []
        for entry in entries:
            pub_dt = parse_feed_date(entry)
            if pub_dt is None or pub_dt < cutoff:
                continue
            title = getattr(entry, "title", None) or "(No title)"
            link = getattr(entry, "link", None) or ""
            summary = (
                getattr(entry, "summary", None)
                or getattr(entry, "description", None)
                or ""
            )
            # Prefer content:encoded when present
            content_blocks = getattr(entry, "content", None) or []
            if content_blocks and isinstance(content_blocks, list):
                first = content_blocks[0]
                if isinstance(first, dict) and first.get("value"):
                    summary = first["value"]
                elif hasattr(first, "value"):
                    summary = first.value

            summary_plain = truncate(strip_html(summary), 700)
            results.append({
                "title": strip_html(title) or title,
                "link": link,
                "summary": summary_plain,
                "source": feed_title,
                "source_key": source_name,
                "published": pub_dt.strftime("%Y-%m-%d %H:%M UTC"),
                "published_dt": pub_dt,
            })
        return results

    articles = build(DAYS_BACK)
    if articles:
        logger.info("Feed OK (%dd): %s — %d", DAYS_BACK, source_name, len(articles))
        return articles

    fallback = build(FALLBACK_DAYS)
    if fallback:
        logger.warning(
            "Feed sparse; using %dd window: %s — %d",
            FALLBACK_DAYS,
            source_name,
            len(fallback),
        )
        return fallback

    logger.warning("Feed has no recent items: %s", source_name)
    return []


def deduplicate_articles(articles: list[dict]) -> list[dict]:
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[dict] = []
    for article in articles:
        url = normalize_url(article.get("link", ""))
        title = (article.get("title") or "").strip().lower()
        if url and url in seen_urls:
            continue
        if title and title in seen_titles:
            continue
        if url:
            seen_urls.add(url)
        if title:
            seen_titles.add(title)
        unique.append(article)
    return unique


def collect_articles(feed_config: list[tuple[str, str]]) -> list[dict]:
    all_articles: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(fetch_feed, url, name): name for url, name in feed_config
        }
        for fut in concurrent.futures.as_completed(futures):
            name = futures[fut]
            try:
                all_articles.extend(fut.result())
            except Exception as e:
                logger.error("Feed worker failed: %s — %s", name, e)

    all_articles.sort(key=lambda a: a.get("published_dt") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    all_articles = deduplicate_articles(all_articles)
    return all_articles[:MAX_ARTICLES]


# --- ENRICHMENT ---


def fetch_article_text(url: str, timeout: float = 12.0) -> str:
    """Best-effort main-text extraction. Returns empty string on failure."""
    if not url or not url.startswith("http"):
        return ""
    import requests

    # Skip likely non-article destinations
    host = urlparse(url).netloc.lower()
    if any(x in host for x in ("reddit.com", "news.ycombinator.com", "twitter.com", "x.com", "youtube.com")):
        return ""

    try:
        r = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            allow_redirects=True,
        )
        r.raise_for_status()
        ctype = (r.headers.get("content-type") or "").lower()
        if "html" not in ctype and "text" not in ctype:
            return ""
        text = strip_html(r.text)
        return truncate(text, ENRICH_MAX_CHARS)
    except Exception as e:
        logger.debug("Enrich failed for %s: %s", url, e)
        return ""


def enrich_articles(articles: list[dict], enabled: bool = True) -> list[dict]:
    """Pull full text for thin summaries so GLM can judge better."""
    if not enabled or not articles:
        return articles

    # Prefer articles with weak RSS blurbs, recent first (list already sorted)
    candidates = [
        a for a in articles
        if len(a.get("summary") or "") < ENRICH_MIN_SUMMARY_CHARS
    ][:ENRICH_CANDIDATES]

    if not candidates:
        logger.info("Enrichment: no thin summaries to expand")
        return articles

    logger.info("Enrichment: fetching body text for %d articles", len(candidates))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        future_map = {pool.submit(fetch_article_text, a["link"]): a for a in candidates}
        for fut in concurrent.futures.as_completed(future_map):
            article = future_map[fut]
            try:
                body = fut.result()
            except Exception:
                body = ""
            if body and len(body) > len(article.get("summary") or ""):
                article["summary"] = body
                article["enriched"] = True

    enriched_n = sum(1 for a in articles if a.get("enriched"))
    logger.info("Enrichment: expanded %d articles", enriched_n)
    return articles


# --- SCORING ---


def build_user_message(articles: list[dict]) -> str:
    blocks = []
    for i, a in enumerate(articles, 1):
        blocks.append(
            f"[{i}]\n"
            f"SOURCE: {a['source']}\n"
            f"TITLE: {a['title']}\n"
            f"PUBLISHED: {a.get('published', '')}\n"
            f"SUMMARY: {a['summary']}\n"
            f"URL: {a['link']}\n"
        )

    return f"""Here are today's candidate articles ({len(articles)} total).

Select the top {TOP_N} that deserve Gil's morning attention. If fewer than {TOP_N} clear the bar (score >= 6.5), return fewer — never pad with weak items.

Also write a day_brief: 2–4 sentences that synthesize what actually matters today across the set (not a list of titles).

Return this exact JSON structure:
{{
  "day_brief": "2-4 sentence executive synthesis of the day's signal.",
  "top_articles": [
    {{
      "title": "exact or cleaned title",
      "url": "canonical URL from the candidate list",
      "source": "source name",
      "score": 8.5,
      "summary": "One tight paragraph: what happened / what the piece claims, concrete and specific.",
      "why_it_matters": "One sentence: practical implication for a COO/AI practitioner deploying systems in real ops.",
      "angle": "One short line: the key takeaway or open question."
    }}
  ]
}}

Candidates:
{chr(10).join(blocks)}
"""


def extract_json_text(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # If model added noise, try to slice the outermost JSON object
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return text


def score_with_glm(
    articles: list[dict],
    api_key: str,
    model: str,
    base_url: str,
    timeout: float,
) -> dict[str, Any]:
    """
    Score candidates via Z.AI OpenAI-compatible chat completions.
    Returns {"day_brief": str, "top_articles": list}.
    """
    if not articles:
        return {"day_brief": "", "top_articles": []}

    import requests

    url = f"{base_url.rstrip('/')}/chat/completions"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(articles)},
            ],
            "max_tokens": 4096,
            "temperature": 0.25,
            "stream": False,
        },
        timeout=timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        body = response.text[:1200]
        logger.error("GLM API failed: HTTP %s — %s", response.status_code, body)
        raise SystemExit("GLM API request failed. See log for response body.") from e

    payload = response.json()
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        logger.error("GLM unexpected response shape: %s", payload)
        raise SystemExit("GLM API returned unexpected response shape.") from e

    if isinstance(content, list):
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    else:
        text = str(content)

    text = extract_json_text(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("GLM invalid JSON. Raw:\n%s", text[:4000])
        raise SystemExit("GLM API returned invalid JSON. See log for raw response.") from e

    top = data.get("top_articles") or []
    if not isinstance(top, list):
        top = []
    day_brief = (data.get("day_brief") or "").strip()
    return {
        "day_brief": day_brief,
        "top_articles": top[:TOP_N],
    }


# --- EMAIL ---


def send_html_email(
    subject: str,
    html: str,
    gmail_user: str,
    gmail_password: str,
    recipient: str,
) -> None:
    from email.utils import formatdate, make_msgid

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = recipient
    msg["Date"] = formatdate(localtime=False, usegmt=True)
    msg["Message-ID"] = make_msgid(domain=gmail_user.split("@")[-1] if "@" in gmail_user else "localhost")
    msg["X-Senal"] = "digest-v2"
    # Plain-text fallback helps some clients / spam filters.
    plain = strip_html(html)
    msg.attach(MIMEText(plain, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(gmail_user, gmail_password)
        refused = smtp.sendmail(gmail_user, [recipient], msg.as_string())
        if refused:
            raise RuntimeError(f"SMTP refused recipients: {refused}")
    logger.info("Email accepted by Gmail SMTP → %s | subject=%s", recipient, subject)


def format_digest_html(
    day_brief: str,
    top_articles: list[dict],
    day_label: str,
    date_str: str,
    articles_by_url: dict[str, dict] | None = None,
    model_name: str = DEFAULT_GLM_MODEL,
) -> str:
    articles_by_url = articles_by_url or {}
    brief = escape(day_brief) if day_brief else "No executive brief produced for this run."

    parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Señal — {escape(day_label)}, {escape(date_str)}</title></head>",
        "<body style=\"margin:0;padding:0;background:#f4f1ea;color:#1a1a1a;"
        "font-family:Georgia,'Times New Roman',serif;\">",
        "<table role='presentation' width='100%' cellspacing='0' cellpadding='0' style='background:#f4f1ea;padding:24px 12px;'>",
        "<tr><td align='center'>",
        "<table role='presentation' width='100%' cellspacing='0' cellpadding='0' "
        "style='max-width:640px;background:#fffdf8;border:1px solid #e6dfd2;"
        "border-radius:12px;overflow:hidden;'>",
        # header
        "<tr><td style='background:#152018;color:#f4f1ea;padding:28px 28px 22px;'>",
        "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "font-size:12px;letter-spacing:0.16em;text-transform:uppercase;color:#9db8a3;\">"
        "SEÑAL</div>",
        f"<div style='font-size:28px;line-height:1.2;margin-top:8px;font-weight:normal;'>"
        f"{escape(day_label)}</div>",
        f"<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        f"font-size:13px;color:#b7c9bb;margin-top:6px;\">{escape(date_str)} · AI signal for operators</div>",
        "</td></tr>",
        # day brief
        "<tr><td style='padding:24px 28px 8px;'>",
        "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "font-size:11px;letter-spacing:0.12em;text-transform:uppercase;color:#6b7280;"
        "margin-bottom:10px;\">Today's brief</div>",
        f"<p style='margin:0;font-size:17px;line-height:1.55;color:#243028;'>{brief}</p>",
        "</td></tr>",
        "<tr><td style='padding:16px 28px 0;'><div style='height:1px;background:#ece5d8;'></div></td></tr>",
    ]

    if not top_articles:
        parts.append(
            "<tr><td style='padding:28px;font-size:16px;color:#555;'>"
            "No articles cleared the bar today. Quiet is sometimes the signal."
            "</td></tr>"
        )
    else:
        for i, a in enumerate(top_articles, 1):
            title = escape(a.get("title", ""))
            url = escape(a.get("url", "#"))
            source = escape(a.get("source", ""))
            published = articles_by_url.get(a.get("url", ""), {}).get("published", "")
            meta = source
            if published:
                meta = f"{source} · {escape(published)}"
            summary = escape(a.get("summary", ""))
            why = escape(a.get("why_it_matters") or a.get("angle") or "")
            angle = escape(a.get("angle", ""))
            score = a.get("score")
            score_html = ""
            if isinstance(score, (int, float)):
                score_html = (
                    f"<span style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
                    f"font-size:12px;color:#6b7280;margin-left:8px;\">· {score:g}/10</span>"
                )

            parts.append("<tr><td style='padding:22px 28px;'>")
            parts.append(
                f"<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
                f"font-size:12px;color:#6b7280;margin-bottom:6px;\">"
                f"<span style='display:inline-block;min-width:1.4em;color:#152018;font-weight:600;'>"
                f"{i:02d}</span> {meta}{score_html}</div>"
            )
            parts.append(
                f"<div style='font-size:20px;line-height:1.3;margin:0 0 10px;'>"
                f"<a href='{url}' style='color:#102016;text-decoration:none;border-bottom:1px solid #c9d6cc;'>"
                f"{title}</a></div>"
            )
            if summary:
                parts.append(
                    f"<p style='margin:0 0 12px;font-size:15.5px;line-height:1.55;color:#2c3330;'>{summary}</p>"
                )
            if why:
                parts.append(
                    "<div style='background:#eef5ef;border-left:3px solid #2f5d3a;padding:10px 12px;"
                    "font-size:14.5px;line-height:1.45;color:#1f3326;'>"
                    f"<strong style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
                    f"font-size:11px;letter-spacing:0.08em;text-transform:uppercase;color:#2f5d3a;\">"
                    f"Why it matters</strong><br>{why}</div>"
                )
            if angle and angle != why:
                parts.append(
                    f"<p style=\"margin:10px 0 0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
                    f"font-size:13px;color:#5b6560;\"><strong>Takeaway:</strong> {angle}</p>"
                )
            parts.append("</td></tr>")
            if i < len(top_articles):
                parts.append(
                    "<tr><td style='padding:0 28px;'><div style='height:1px;background:#f0ebe1;'></div></td></tr>"
                )

    parts.extend([
        "<tr><td style='padding:20px 28px 28px;'>",
        "<div style='height:1px;background:#ece5d8;margin-bottom:16px;'></div>",
        f"<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        f"font-size:12px;color:#8a8f8b;line-height:1.5;\">"
        f"Señal v2 · scored with {escape(model_name)} · not news, signal."
        "</div>",
        "</td></tr>",
        "</table></td></tr></table></body></html>",
    ])
    return "\n".join(parts)


def send_digest_email(
    day_brief: str,
    top_articles: list[dict],
    articles: list[dict],
    day_label: str,
    date_str: str,
    gmail_user: str,
    gmail_password: str,
    recipient: str,
    model_name: str,
) -> None:
    subject = f"📡 Señal — {day_label}, {date_str}"
    by_url = {}
    for a in articles:
        by_url[a.get("link", "")] = a
        by_url[normalize_url(a.get("link", ""))] = a
    # Also index model-returned URLs against normalized forms
    for a in top_articles:
        u = a.get("url") or ""
        nu = normalize_url(u)
        if u in by_url:
            by_url[nu] = by_url[u]
        elif nu in by_url:
            by_url[u] = by_url[nu]

    html = format_digest_html(
        day_brief,
        top_articles,
        day_label,
        date_str,
        by_url,
        model_name=model_name,
    )
    send_html_email(subject, html, gmail_user, gmail_password, recipient)


def send_test_email(gmail_user: str, gmail_password: str, recipient: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subject = f"Señal SMTP test · {now}"
    html = (
        "<!DOCTYPE html><html><body style='font-family:system-ui,sans-serif;padding:24px;'>"
        "<h2 style='margin:0 0 12px;'>Señal v2 SMTP test</h2>"
        f"<p>If you can read this, delivery works.</p>"
        f"<p style='color:#666;'>Sent at {escape(now)}.<br>"
        f"From: {escape(gmail_user)}<br>To: {escape(recipient)}</p>"
        "<p style='color:#666;font-size:13px;'>Search Gmail for <strong>Señal SMTP test</strong> "
        "in <em>All Mail</em>, <em>Spam</em>, and <em>Sent</em> if it is not in Inbox.</p>"
        "</body></html>"
    )
    send_html_email(subject, html, gmail_user, gmail_password, recipient)


# --- MAIN ---


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_runtime_config(
        require_llm=not args.send_test_email,
        require_smtp=not args.dry_run,
    )

    if args.health_check:
        log_health_check(config)
        return

    if args.send_test_email:
        send_test_email(config["gmail_user"], config["gmail_password"], config["email_to"])
        return

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_label = day_names[now.weekday()]

    seen = {"version": 1, "articles": {}} if args.no_memory else load_seen()
    if not args.no_memory:
        prune_seen(seen)
        logger.info("Seen memory loaded: %d entries", len(seen.get("articles") or {}))

    articles = collect_articles(AI_FEEDS)
    logger.info("Collected %d unique recent articles", len(articles))

    if not args.no_memory:
        articles = filter_unseen(articles, seen)
        logger.info("%d candidates after memory filter", len(articles))

    if not articles:
        logger.warning("No candidates left after collection/memory filter; skipping GLM + email.")
        return

    articles = enrich_articles(articles, enabled=not args.no_enrich)

    result = score_with_glm(
        articles,
        config["glm_api_key"],
        config["glm_model"],
        config["glm_base_url"],
        float(config["glm_timeout"]),
    )
    day_brief = result.get("day_brief") or ""
    top_articles = result.get("top_articles") or []

    if not top_articles:
        logger.warning("GLM returned no top articles; skipping email.")
        return

    logger.info("Top picks: %d | brief chars: %d", len(top_articles), len(day_brief))
    for i, a in enumerate(top_articles, 1):
        logger.info(
            "  %d. [%.1f] %s",
            i,
            float(a.get("score") or 0),
            (a.get("title") or "")[:90],
        )

    if args.dry_run:
        logger.info("Dry run complete — would email %d items to %s", len(top_articles), config["email_to"])
        if day_brief:
            logger.info("Day brief: %s", day_brief)
        return

    send_digest_email(
        day_brief,
        top_articles,
        articles,
        day_label,
        date_str,
        config["gmail_user"],
        config["gmail_password"],
        config["email_to"],
        config["glm_model"],
    )

    if not args.no_memory:
        mark_sent(seen, top_articles, day_label)
        save_seen(seen)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        logger.exception("Digest run failed")
        sys.exit(1)
