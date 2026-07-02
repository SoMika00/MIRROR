import os
import threading
import logging
from flask import Flask
from flask_cors import CORS

logger = logging.getLogger(__name__)


def _auto_init_services(app):
    """Background thread: connect Qdrant, load embedding, reranker, init Grok API."""
    with app.app_context():
        # 1. Connect Qdrant
        try:
            from app.services.qdrant_store import qdrant_store
            qdrant_store.connect()
            app.logger.info("Auto-init: Qdrant connected")
        except Exception as e:
            app.logger.error(f"Auto-init: Qdrant failed: {e}")

        # 2. Load embedding model
        try:
            from app.services.embedding import embedding_service
            embedding_service.load()
            app.logger.info("Auto-init: Embedding model loaded")
        except Exception as e:
            app.logger.error(f"Auto-init: Embedding failed: {e}")

        # 3. Load reranker
        try:
            from app.services.reranker import reranker_service
            reranker_service.load()
            app.logger.info("Auto-init: Reranker loaded")
        except Exception as e:
            app.logger.error(f"Auto-init: Reranker failed: {e}")

        # 4. Init Grok API client
        try:
            from app.services.llm import llm_service
            llm_service.load()
            app.logger.info("Auto-init: Grok API client ready")
        except Exception as e:
            app.logger.error(f"Auto-init: Grok API failed: {e}")

        app.logger.info("Auto-init: All services initialization complete")


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

    # Initialize SQLite database
    from app.services.database import db
    db.init_db()

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

    # Auto-init all services in background so the app starts immediately
    thread = threading.Thread(target=_auto_init_services, args=(app,), daemon=True)
    thread.start()

    return app
