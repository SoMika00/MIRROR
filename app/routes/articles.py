"""Articles routes — serve markdown articles."""

import os
import glob
from flask import Blueprint, jsonify, current_app, request
import markdown

articles_bp = Blueprint("articles", __name__)


_SUPPORTED_LANGS = ("fr", "en", "ja")


def _normalize_lang(lang: str | None) -> str | None:
    if not lang:
        return None
    l = str(lang).strip().lower()
    for s in _SUPPORTED_LANGS:
        if l.startswith(s):
            return s
    return None


def _base_slug_and_lang_from_filename(filename: str) -> tuple[str, str | None]:
    name = os.path.basename(filename)
    if not name.endswith(".md"):
        return os.path.splitext(name)[0], None
    stem = name[:-3]
    for lang in _SUPPORTED_LANGS:
        suffix = f".{lang}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], lang
    return stem, None


def _candidate_filepaths(articles_dir: str, slug: str, lang: str | None) -> list[str]:
    normalized = _normalize_lang(lang)
    order: list[str | None] = []
    if normalized:
        order.append(normalized)
    # requested fallback chain
    order.extend(["fr", "en", "ja", None])

    seen: set[str] = set()
    candidates: list[str] = []
    for l in order:
        if l is None:
            filename = f"{slug}.md"
        else:
            filename = f"{slug}.{l}.md"
        fp = os.path.join(articles_dir, filename)
        if fp in seen:
            continue
        seen.add(fp)
        candidates.append(fp)
    return candidates


def _load_article(filepath: str) -> dict:
    """Load a markdown article and extract metadata from frontmatter."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Simple frontmatter parsing (--- delimited)
    meta = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            body = parts[2].strip()
            for line in frontmatter.split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    meta[key.strip().lower()] = val.strip()

    html = markdown.markdown(body, extensions=["fenced_code", "tables", "toc"])
    slug = os.path.splitext(os.path.basename(filepath))[0]

    return {
        "slug": slug,
        "title": meta.get("title", slug.replace("-", " ").title()),
        "date": meta.get("date", ""),
        "tags": [t.strip() for t in meta.get("tags", "").split(",") if t.strip()],
        "summary": meta.get("summary", ""),
        "html": html,
        "markdown": body,
    }


@articles_bp.route("/list", methods=["GET"])
def list_articles():
    """List all available articles."""
    articles_dir = current_app.config["ARTICLES_FOLDER"]
    lang = _normalize_lang(request.args.get("lang"))
    files = sorted(glob.glob(os.path.join(articles_dir, "*.md")))

    grouped: dict[str, dict] = {}
    for f in files:
        base_slug, file_lang = _base_slug_and_lang_from_filename(f)
        entry = grouped.setdefault(base_slug, {"base_slug": base_slug, "langs": set(), "files": {}})
        if file_lang:
            entry["langs"].add(file_lang)
        entry["files"][file_lang] = f

    articles: list[dict] = []
    for base_slug, entry in grouped.items():
        # pick best file for requested lang
        chosen_fp = None
        for fp in _candidate_filepaths(articles_dir, base_slug, lang):
            if os.path.exists(fp):
                chosen_fp = fp
                break

        if not chosen_fp:
            continue

        try:
            article = _load_article(chosen_fp)
            articles.append({
                "slug": base_slug,
                "title": article["title"],
                "date": article["date"],
                "tags": article["tags"],
                "summary": article["summary"],
                "available_langs": sorted(list(entry["langs"])),
            })
        except Exception:
            continue

    # sort by date descending (newest first)
    articles.sort(key=lambda a: a.get("date", ""), reverse=True)
    return jsonify({"articles": articles})


@articles_bp.route("/<slug>", methods=["GET"])
def get_article(slug: str):
    """Get a single article by slug."""
    articles_dir = current_app.config["ARTICLES_FOLDER"]
    lang = _normalize_lang(request.args.get("lang"))
    filepath = None
    for fp in _candidate_filepaths(articles_dir, slug, lang):
        if os.path.exists(fp):
            filepath = fp
            break

    if not filepath:
        return jsonify({"error": "Article not found"}), 404

    try:
        article = _load_article(filepath)
        article["slug"] = slug
        article["lang"] = _base_slug_and_lang_from_filename(filepath)[1]
        return jsonify(article)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
