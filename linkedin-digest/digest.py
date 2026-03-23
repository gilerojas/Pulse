"""
Señal: pulls AI news from RSS feeds, scores them via Claude API for relevance
to a specific profile, and emails the top 5 as a personalized HTML digest to
Gmail. Designed to run on a schedule (e.g. GitHub Actions) on Monday and
Thursday mornings.
"""

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

import feedparser
import requests
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---

AI_FEEDS = [
    # Newsletters & curated
    "https://www.deeplearning.ai/the-batch/feed",
    "https://tldr.tech/ai/rss",
    "https://benbites.beehiiv.com/feed",
    "https://importai.substack.com/feed",          # Import AI — Jack Clark
    "https://thegradient.pub/rss/",                # The Gradient — research-level AI
    # Blogs & researchers
    "https://simonwillison.net/atom/everything/",  # Simon Willison — LLM practitioner
    "https://huggingface.co/blog/feed.xml",        # Hugging Face blog
    "https://interconnects.ai/feed",               # Nathan Lambert — RLHF/alignment
    # News & forums
    "https://www.technologyreview.com/feed/",      # MIT Technology Review
    "https://hnrss.org/frontpage",                 # Hacker News front page
    "https://www.reddit.com/r/MachineLearning/.rss",
    "https://www.reddit.com/r/artificial/.rss",
]

SYSTEM_PROMPT = """
You are a content intelligence assistant for Gilberto Rojas, a 26-year-old COO based in Santo Domingo, Dominican Republic. He runs two companies in the paint and chemical sector and is transitioning toward becoming CEO. He has an MSc in Data Science & Deep Learning and uses AI extensively in real production operations (not experimentally). He wants to stay sharp on the AI frontier as a practitioner, not as a journalist or analyst.

His areas of deepest interest:
1. AI models, capabilities, and research breakthroughs — what actually changed and why it matters
2. AI in real operations — practical deployment, what works, what breaks, lessons from the field
3. AI tooling and infrastructure — new frameworks, APIs, agents, and deployment patterns
4. AI policy, safety, and industry dynamics — who is doing what, and the broader implications

Score each article from 0 to 10 based on:
- Significance of the AI development or insight (most important)
- Practical relevance to someone deploying AI in real systems
- Recency and credibility of the source

For each top article, write a "summary" that is one short, hooky paragraph: what the article/report is actually about, so the reader immediately gets the story and why it matters. Make it engaging, not dry.

Return ONLY valid JSON. No preamble, no explanation outside the JSON.
"""

MAX_ARTICLES = 60
DAYS_BACK = 3
TOP_N = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- HELPERS ---


def strip_html(text: str) -> str:
    """Remove HTML tags from a string."""
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


def parse_feed_date(entry) -> datetime | None:
    """Get published or updated datetime from a feed entry, timezone-aware UTC."""
    for attr in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, attr, None)
        if parsed is None:
            continue
        try:
            # time.struct_time to datetime (assume UTC if no tz)
            dt = datetime(*parsed[:6], tzinfo=timezone.utc)
            return dt
        except (TypeError, ValueError):
            continue
    return None


def fetch_feed(url: str, source_name: str) -> list[dict]:
    """
    Fetch and parse a single RSS feed. Returns list of article dicts.
    Uses requests as HTTP fallback if feedparser fails to fetch.
    First tries a 3-day window; if no items are found but the feed has older
    entries, falls back to a 7-day window. Does not raise.
    """
    resp = None
    try:
        resp = feedparser.parse(
            url,
            request_headers={"User-Agent": "Señal/1.0"},
        )
    except Exception as e:
        logger.error("Feed failed (parse via feedparser): %s — %s", url, e)
        # Try HTTP fallback
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "Señal/1.0"})
            r.raise_for_status()
            resp = feedparser.parse(r.content)
        except Exception as e2:
            logger.error("Feed failed (HTTP fallback): %s — %s", url, e2)
            return []

    if resp is None:
        logger.error("Feed failed: %s — no response object", url)
        return []

    if getattr(resp, "bozo", False) and not getattr(resp, "entries", None):
        logger.error("Feed failed (bozo parse with no entries): %s", url)
        return []

    entries = getattr(resp, "entries", []) or []
    if not entries:
        logger.warning("Feed has no usable entries: %s", url)
        return []

    feed_title = getattr(resp.feed, "title", None) or source_name

    def build_articles(cutoff_days: int) -> list[dict]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=cutoff_days)
        results: list[dict] = []
        for entry in entries:
            pub_dt = parse_feed_date(entry)
            if pub_dt is None or pub_dt < cutoff:
                continue

            title = getattr(entry, "title", None) or "(No title)"
            link = getattr(entry, "link", None) or ""
            summary = getattr(entry, "summary", None) or getattr(entry, "description", None) or ""
            summary_plain = strip_html(summary)
            if len(summary_plain) > 500:
                summary_plain = summary_plain[:497] + "..."

            results.append({
                "title": title,
                "link": link,
                "summary": summary_plain,
                "source": feed_title,
                "published": pub_dt.strftime("%Y-%m-%d %H:%M UTC"),
            })
        return results

    # First, try normal 3-day window
    articles = build_articles(DAYS_BACK)
    if articles:
        logger.info("Feed OK (last %d days): %s — %d articles", DAYS_BACK, url, len(articles))
        return articles

    # If nothing in 3 days but feed has older entries, fall back to 7 days
    fallback_days = 7
    fallback_articles = build_articles(fallback_days)
    if fallback_articles:
        logger.warning(
            "Feed has no items in last %d days; using last %d days instead: %s — %d articles",
            DAYS_BACK,
            fallback_days,
            url,
            len(fallback_articles),
        )
        return fallback_articles

    logger.warning(
        "Feed has entries but none within %d or %d days: %s",
        DAYS_BACK,
        fallback_days,
        url,
    )
    return []


# --- FEED PARSING ---


def collect_articles(feed_config: list[tuple[str, str]]) -> list[dict]:
    """
    Fetch all feeds and aggregate articles from the last DAYS_BACK days.
    Caps at MAX_ARTICLES. Single feed failures are logged and skipped.
    """
    all_articles = []
    for url, source_name in feed_config:
        articles = fetch_feed(url, source_name)
        all_articles.extend(articles)

    all_articles.sort(key=lambda a: a["published"], reverse=True)
    return all_articles[:MAX_ARTICLES]


def build_feed_config() -> list[tuple[str, str]]:
    """Build (url, source_name) list from AI_FEEDS."""
    return [(url, "AI") for url in AI_FEEDS]


# --- CLAUDE SCORING ---


def build_user_message(articles: list[dict]) -> str:
    """Build the user message sent to Claude with all article blocks."""
    blocks = []
    for a in articles:
        blocks.append(
            f"SOURCE: {a['source']}\n"
            f"TITLE: {a['title']}\n"
            f"SUMMARY: {a['summary']}\n"
            f"URL: {a['link']}\n"
        )
    body = "Here are today's articles. Score each and return the top 5.\n\n" + "\n".join(blocks)
    body += """

Return this exact JSON structure:
{
  "top_articles": [
    {
      "title": "...",
      "url": "...",
      "source": "...",
      "score": 8.5,
      "summary": "One hooky paragraph that summarizes what the article/report is actually about — so the reader immediately understands the story and why it matters, in an engaging way.",
      "angle": "One sentence: the key takeaway or question this raises for an AI practitioner"
    }
  ]
}
"""
    return body


def score_with_claude(articles: list[dict], api_key: str) -> list[dict]:
    """
    Send all articles to Claude and return the top N as structured data.
    Raises if the API call fails or response is not valid JSON.
    """
    if not articles:
        return []

    client = Anthropic(api_key=api_key)
    user_message = build_user_message(articles)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text

    text = text.strip()
    # Allow optional markdown code fence around JSON
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Claude API returned invalid JSON. Raw response:\n%s", text)
        raise SystemExit("Claude API returned invalid JSON. See log for raw response.") from e

    top = data.get("top_articles") or []
    return top[:TOP_N] if isinstance(top, list) else []


# --- EMAIL ---


def format_digest_html(
    top_articles: list[dict],
    day_label: str,
    date_str: str,
    articles_by_url: dict[str, dict] | None = None,
) -> str:
    """Build minimal HTML body for the digest email."""
    articles_by_url = articles_by_url or {}
    parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'></head><body style='font-family: system-ui, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; color: #333;'>",
        f"<h1 style='font-size: 1.5rem; margin-bottom: 0;'>Señal</h1>",
        f"<p style='color: #666; margin-top: 4px;'>{day_label}, {date_str}</p>",
        "<hr style='border: none; border-top: 1px solid #eee; margin: 24px 0;'>",
    ]
    for i, a in enumerate(top_articles):
        title = escape(a.get("title", ""))
        url = escape(a.get("url", "#"))
        source = escape(a.get("source", ""))
        published = articles_by_url.get(a.get("url", ""), {}).get("published", "")
        if published:
            source = f"{source} — {published}"
        summary = escape(a.get("summary", ""))
        angle = escape(a.get("angle", ""))
        parts.append(f"<h2 style='font-size: 1.1rem;'><a href='{url}' style='color: #0a66c2; text-decoration: none;'>{title}</a></h2>")
        parts.append(f"<p style='color: #666; font-size: 0.9rem; margin: 4px 0;'>{source}</p>")
        parts.append(f"<p style='margin: 8px 0; line-height: 1.5;'>{summary}</p>")
        parts.append(f"<p style='margin: 8px 0;'><strong>Takeaway:</strong> {angle}</p>")
        if i < len(top_articles) - 1:
            parts.append("<hr style='border: none; border-top: 1px solid #eee; margin: 24px 0;'>")
    parts.append("<hr style='border: none; border-top: 1px solid #eee; margin: 24px 0;'>")
    parts.append("<p style='color: #999; font-size: 0.85rem;'>Powered by Claude + GitHub Actions</p>")
    parts.append("</body></html>")
    return "\n".join(parts)


def send_digest_email(
    top_articles: list[dict],
    articles: list[dict],
    day_label: str,
    date_str: str,
    gmail_user: str,
    gmail_password: str,
) -> None:
    """
    Send the digest as HTML email via Gmail SMTP to the same address.
    Raises on SMTP failure (delivery failure must not be silent).
    """
    subject = f"📬 Señal — {day_label}, {date_str}"
    by_url = {a["link"]: a for a in articles}
    html = format_digest_html(top_articles, day_label, date_str, by_url)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = gmail_user
    msg.attach(MIMEText(html, "html", "utf-8"))

    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.starttls()
        smtp.login(gmail_user, gmail_password)
        smtp.sendmail(gmail_user, [gmail_user], msg.as_string())
    logger.info("Digest email sent to %s", gmail_user)


# --- MAIN ---


def main() -> None:
    """Pull feeds, score with Claude, send digest email."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    gmail_user = os.environ.get("GMAIL_USER")
    gmail_password = os.environ.get("GMAIL_APP_PASSWORD")

    if not api_key or not gmail_user or not gmail_password:
        raise SystemExit(
            "Missing env: set ANTHROPIC_API_KEY, GMAIL_USER, and GMAIL_APP_PASSWORD."
        )

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_label = day_names[now.weekday()]

    feed_config = build_feed_config()
    articles = collect_articles(feed_config)
    logger.info("Collected %d articles from feeds", len(articles))

    if not articles:
        logger.warning("No articles in the last %d days; skipping Claude and email.", DAYS_BACK)
        return

    top_articles = score_with_claude(articles, api_key)
    if not top_articles:
        logger.warning("Claude returned no top articles; skipping email.")
        return

    send_digest_email(
        top_articles,
        articles,
        day_label,
        date_str,
        gmail_user,
        gmail_password,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Digest run failed")
        sys.exit(1)
