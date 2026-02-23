import os
from flask import Flask
from flask_cors import CORS


def create_app():
    app = Flask(__name__)
    CORS(app)

    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")
    app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_FOLDER", "./uploads")
    app.config["ARTICLES_FOLDER"] = os.environ.get("ARTICLES_FOLDER", "./articles")
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["ARTICLES_FOLDER"], exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    # Auto-load LLM if model file exists
    try:
        from app.services.llm import llm_service
        model_path = os.environ.get("MODEL_PATH", "./models/phi-4-Q4_K_M.gguf")
        if os.path.exists(model_path):
            app.logger.info(f"Auto-loading LLM: {model_path}")
            llm_service.load(model_path)
        else:
            app.logger.info(f"No model found at {model_path}, LLM will be loaded manually")
    except Exception as e:
        app.logger.error(f"Failed to auto-load LLM: {e}")

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

    return app
