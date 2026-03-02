"""Main routes - landing page, CV showcase, articles page."""

import re
from flask import Blueprint, render_template, redirect, request, abort, current_app

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    return render_template("index.html")


@main_bp.route("/chat")
def chat():
    return render_template("chat.html")


@main_bp.route("/articles")
def articles():
    return render_template("articles.html")


@main_bp.route("/articles/<article_id>")
def article_detail(article_id):
    """Render a dedicated page for a single article."""
    from app.routes.articles import _candidate_filepaths, _load_article, _normalize_lang, _ARTICLE_IDS
    articles_dir = current_app.config["ARTICLES_FOLDER"]

    # Map ID to slug, fallback to treating ID as slug
    slug = _ARTICLE_IDS.get(article_id, article_id)

    # Determine language from query param or localStorage cookie fallback
    lang = _normalize_lang(request.args.get("lang"))

    filepath = None
    for fp in _candidate_filepaths(articles_dir, slug, lang):
        import os
        if os.path.exists(fp):
            filepath = fp
            break

    if not filepath:
        abort(404)

    article = _load_article(filepath)
    article["slug"] = slug
    article["id"] = article_id

    # Strip the first <h1> from HTML to avoid doubled title
    article["html"] = re.sub(r'<h1[^>]*>.*?</h1>\s*', '', article["html"], count=1)

    # Upgrade markdown tables to tech-table style
    article["html"] = article["html"].replace('<table>', '<table class="tech-table">')

    return render_template("article_detail.html", article=article)


@main_bp.route("/tech")
def tech():
    return render_template("tech.html")


@main_bp.route("/playbook")
def playbook():
    return redirect("/courses")


@main_bp.route("/courses")
def courses():
    return render_template("courses.html")
