"""Main routes — landing page, CV showcase, articles page."""

from flask import Blueprint, render_template, redirect

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


@main_bp.route("/tech")
def tech():
    return render_template("tech.html")


@main_bp.route("/playbook")
def playbook():
    return redirect("/courses")


@main_bp.route("/courses")
def courses():
    return render_template("courses.html")
