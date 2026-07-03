import os
import threading
import logging
from flask import Flask
from flask_cors import CORS

logger = logging.getLogger(__name__)


def _init_knowledge(app):
    """Background thread: index the portfolio's own content for the AI chat."""
    with app.app_context():
        try:
            from app.services.knowledge import index_portfolio_content
            n = index_portfolio_content(
                articles_dir=app.config["ARTICLES_FOLDER"],
                docs_dir="./docs",
            )
            app.logger.info(f"Auto-init: portfolio knowledge indexed ({n} chunks)")
        except Exception as e:
            app.logger.error(f"Auto-init: knowledge indexing failed: {e}")


def create_app():
    app = Flask(__name__)
    CORS(app)

    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")
    app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_FOLDER", "./uploads")
    app.config["ARTICLES_FOLDER"] = os.environ.get("ARTICLES_FOLDER", "./articles")
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["ARTICLES_FOLDER"], exist_ok=True)
    os.makedirs("./data", exist_ok=True)

    # SQLite: app data + retrieval store (FTS5)
    from app.services.database import db
    db.init_db()
    from app.services.retrieval import retrieval_store
    retrieval_store.init_db()

    # Grok API client (fails loudly in logs if the key is missing)
    try:
        from app.services.llm import llm_service
        llm_service.load()
        app.logger.info("Grok API client ready")
    except Exception as e:
        app.logger.error(f"Grok API init failed: {e}")

    from app.routes.main import main_bp
    from app.routes.chat import chat_bp
    from app.routes.documents import documents_bp
    from app.routes.scraper import scraper_bp
    from app.routes.articles import articles_bp
    from app.routes.models_route import models_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(chat_bp, url_prefix="/api/chat")
    app.register_blueprint(documents_bp, url_prefix="/api/documents")
    app.register_blueprint(scraper_bp, url_prefix="/api/scraper")
    app.register_blueprint(articles_bp, url_prefix="/api/articles")
    app.register_blueprint(models_bp, url_prefix="/api/models")

    # Index portfolio content in the background so startup stays instant
    threading.Thread(target=_init_knowledge, args=(app,), daemon=True).start()

    return app
