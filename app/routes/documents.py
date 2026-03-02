"""Document upload and management routes."""

import os
import time
from flask import Blueprint, request, jsonify, current_app, make_response

from werkzeug.utils import secure_filename

from app.services.pdf_parser import parse_document, chunk_text
from app.services.embedding import embedding_service
from app.services.qdrant_store import qdrant_store
from app.services.database import db
from app.config import rag_cfg

documents_bp = Blueprint("documents", __name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def _get_user_id() -> str:
    """Get or create user from cookie."""
    user_id = request.cookies.get("mirror_uid")
    return db.get_or_create_user(user_id)


def _set_user_cookie(response, user_id: str):
    """Set persistent user cookie (1 year)."""
    response.set_cookie("mirror_uid", user_id, max_age=365 * 24 * 3600,
                         httponly=True, samesite="Lax")
    return response


@documents_bp.route("/upload", methods=["POST"])
def upload_document():
    """Upload and index a document into the vector store."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported format. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    user_id = _get_user_id()
    filename = secure_filename(file.filename)
    upload_dir = current_app.config["UPLOAD_FOLDER"]
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)

    try:
        start = time.time()

        # Parse document
        pages = parse_document(filepath)

        # Chunk all pages
        all_chunks = []
        for page_data in pages:
            chunks = chunk_text(page_data["text"], rag_cfg.chunk_size, rag_cfg.chunk_overlap)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "page": page_data.get("page", 1),
                    "chunk_index": i,
                })

        if not all_chunks:
            return jsonify({"error": "No text extracted from document"}), 400

        # Embed all chunks
        texts = [c["text"] for c in all_chunks]
        vectors = embedding_service.encode(texts).tolist()

        # Build payloads - include user_id for per-user scoping
        payloads = []
        for c in all_chunks:
            payloads.append({
                "source_name": filename,
                "source_type": "document",
                "page": c["page"],
                "chunk_index": c["chunk_index"],
                "user_id": user_id,
            })

        # Upsert to Qdrant
        qdrant_store.upsert(texts=texts, vectors=vectors, payloads=payloads)

        elapsed = time.time() - start

        resp = make_response(jsonify({
            "success": True,
            "filename": filename,
            "pages": len(pages),
            "chunks": len(all_chunks),
            "elapsed_seconds": round(elapsed, 2),
        }))
        return _set_user_cookie(resp, user_id)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@documents_bp.route("/list", methods=["GET"])
def list_documents():
    """List indexed documents for the current user."""
    user_id = _get_user_id()
    try:
        sources = qdrant_store.list_sources(user_id=user_id)
        resp = make_response(jsonify({"sources": sources}))
        return _set_user_cookie(resp, user_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@documents_bp.route("/delete/<source_name>", methods=["DELETE"])
def delete_document(source_name):
    """Delete a document from the vector store (scoped to current user)."""
    user_id = _get_user_id()
    try:
        qdrant_store.delete_by_source(source_name, user_id=user_id)
        # Also delete file if exists
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], source_name)
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"success": True, "deleted": source_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@documents_bp.route("/info", methods=["GET"])
def collection_info():
    """Get vector store collection info."""
    try:
        info = qdrant_store.get_collection_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e), "connected": False}), 500
