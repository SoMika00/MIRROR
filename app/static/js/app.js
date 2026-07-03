/* ============================================================
   MIRROR - Global JavaScript
   Handles: i18n, status polling, toast notifications
   ============================================================ */

// --- i18n ---
const I18N_STORAGE_KEY = 'mirror_lang';
const I18N_DEFAULT_LANG = 'en';

const I18N_DICT = {
    en: {
        'nav.portfolio': 'Portfolio',
        'nav.chat': 'AI Chat',
        'nav.articles': 'Articles',
        'nav.courses': 'Courses',
        'nav.tech': 'Architecture',
        'home.subtitle': 'Lead AI/LLM Engineer',
        'home.summary': '5+ years shipping AI systems to production: enterprise RAG, LLM fine-tuning (LoRA/QLoRA), OCR/NER, and the MLOps to keep them reliable. Team lead (4-8 engineers). Based in Tokyo — open to international teams and French companies.',
        'home.ask_ai': 'Ask the AI about me',
        'home.tech_choices': 'How this site works',
        'home.projects': 'Selected Work',
        'home.skills': 'Technical Skills',
        'home.skill_cat.genai': 'Generative AI',
        'home.skill_cat.inference': 'Inference & Scaling',
        'home.skill_cat.vectorstores': 'Retrieval & Vector Search',
        'home.skill_cat.mlops': 'MLOps & GPU',
        'home.skill_cat.backend': 'Backend & Data',
        'home.skill_cat.cloud': 'Cloud',
        'home.experience': 'Experience',
        'home.education': 'Education',
        'home.languages': 'Languages',
        'home.certifications': 'Certifications',
        'home.site_note.title': 'This site is part of the portfolio',
        'home.site_note.body': 'MIRROR is not a template. It is a live, cost-engineered AI product: a RAG chatbot over my own knowledge base (Grok API + hybrid BM25 retrieval), document upload, web scraping, multi-user sessions — running on a small VPS for under $1/day. The architecture decisions are documented like I would for any client.',
        'home.site_note.cta': 'Read the architecture decisions →',
        'chat.upload_docs': 'Upload Documents',
        'chat.drop_here': 'Drop files here',
        'chat.supported_files': 'PDF, DOCX, TXT, MD',
        'chat.web_scraper': 'Web Scraper',
        'chat.url_placeholder': 'https://...',
        'chat.go': 'Go',
        'chat.indexed_docs': 'Indexed Documents',
        'chat.no_docs': 'No documents indexed',
        'chat.filter_source': 'Filter Source',
        'chat.welcome': "Hi — I'm <strong>MIRROR</strong>, Michail's AI assistant. Ask me anything about his experience, projects, or how this site is built. I answer with citations from his CV, articles and architecture docs. You can also drop in your own documents or scrape a URL and query them.",
        'chat.infra_note': 'Powered by the xAI Grok API with hybrid retrieval (BM25 · SQLite FTS5). A daily cost cap protects the budget — if I stop responding, I will be back tomorrow.',
        'chat.ask_placeholder': 'Ask about Michail, RAG, this architecture...',
        'chat.suggestion.experience': 'What did Michail build at SOMA?',
        'chat.suggestion.rag': 'How does the RAG on this site work?',
        'chat.suggestion.lead': 'What is his experience leading teams?',
        'articles.title': 'Articles',
        'articles.subtitle': 'Technical writing on AI, RAG, MLOps and production systems.',
        'articles.loading': 'Loading articles...',
        'articles.empty_title': 'No articles yet',
        'articles.empty_hint': 'Add markdown files to the articles/ directory to get started.',
        'articles.back': '← Back to articles',
        'tech.title': 'Architecture & Technical Choices',
        'tech.subtitle': 'How MIRROR is built, what it costs, and why — documented like a client deliverable.',
        'courses.title': 'Courses',
        'courses.subtitle': 'A clear, structured overview of core ML topics, explained with lists, examples, and practical intuition.',
        'courses.ml.title': 'Machine Learning (Classic)',
        'courses.ml.intro': 'Classical ML is often the fastest path to a strong production baseline: interpretable, cheap, and reliable when the data is right.',
        'courses.ml.supervised': 'Supervised learning',
        'courses.ml.unsupervised': 'Unsupervised & representation',
        'courses.ml.stats': 'Stats essentials',
        'courses.ml.production': 'Production mindset',
        'courses.dl.title': 'Deep Learning (Neural Networks)',
        'courses.dl.intro': 'Deep learning is mostly about learning representations. I group networks by what structure they exploit: sequences, images, graphs, and multimodal signals.',
        'courses.dl.foundations': 'Foundations',
        'courses.dl.networks': 'Main network families',
        'courses.dl.training': 'Training & scaling',
        'courses.dl.failure': 'Typical failure modes',
        'courses.cv.title': 'Computer Vision',
        'courses.cv.intro': 'Vision systems are about extracting structure from pixels. Modern SOTA is dominated by Transformers and strong pretraining.',
        'courses.cv.tasks': 'Core tasks',
        'courses.cv.models': 'Typical SOTA families',
        'courses.cv.metrics': 'Metrics',
        'courses.cv.production': 'Production notes',
        'courses.nlp.title': 'NLP & LLMs',
        'courses.nlp.intro': 'Modern NLP is Transformer-based. The practical skill is not just training, but controlling behavior: prompting, retrieval, evaluation, safety.',
        'courses.nlp.tasks': 'Core tasks',
        'courses.nlp.components': 'Key components',
        'courses.nlp.eval': 'Evaluation',
        'courses.nlp.production': 'Production notes',
        'courses.visual.bias_variance': 'Bias / Variance',
        'courses.visual.transformer': 'Transformer (high level)',
        'courses.visual.rag': 'RAG pipeline',
        'courses.footer': 'MIRROR · Courses · Michail Berjaoui',
        'footer.tagline': 'MIRROR · AI-powered portfolio · Flask · xAI Grok API · SQLite FTS5 · Docker + Caddy',
        'common.integrality': 'The entirety',
    },
    fr: {
        'nav.portfolio': 'Portfolio',
        'nav.chat': 'Chat IA',
        'nav.articles': 'Articles',
        'nav.courses': 'Cours',
        'nav.tech': 'Architecture',
        'home.subtitle': 'Lead AI/LLM Engineer',
        'home.summary': "5+ ans à mettre des systèmes d'IA en production : RAG entreprise, fine-tuning LLM (LoRA/QLoRA), OCR/NER, et le MLOps pour les fiabiliser. Lead d'équipe (4-8 ingénieurs). Basé à Tokyo — ouvert aux équipes internationales et aux entreprises françaises.",
        'home.ask_ai': "Demander à l'IA à mon sujet",
        'home.tech_choices': 'Comment ce site fonctionne',
        'home.projects': 'Réalisations',
        'home.skills': 'Compétences techniques',
        'home.skill_cat.genai': 'IA générative',
        'home.skill_cat.inference': 'Inférence & scaling',
        'home.skill_cat.vectorstores': 'Retrieval & recherche vectorielle',
        'home.skill_cat.mlops': 'MLOps & GPU',
        'home.skill_cat.backend': 'Backend & data',
        'home.skill_cat.cloud': 'Cloud',
        'home.experience': 'Expérience',
        'home.education': 'Formation',
        'home.languages': 'Langues',
        'home.certifications': 'Certifications',
        'home.site_note.title': 'Ce site fait partie du portfolio',
        'home.site_note.body': "MIRROR n'est pas un template. C'est un produit IA vivant et optimisé en coût : un chatbot RAG sur ma propre base de connaissances (API Grok + retrieval hybride BM25), upload de documents, scraping web, sessions multi-utilisateurs — sur un petit VPS pour moins de 1 $/jour. Les décisions d'architecture sont documentées comme pour un client.",
        'home.site_note.cta': "Lire les décisions d'architecture →",
        'chat.upload_docs': 'Téléverser des documents',
        'chat.drop_here': 'Dépose tes fichiers ici',
        'chat.supported_files': 'PDF, DOCX, TXT, MD',
        'chat.web_scraper': 'Scraper web',
        'chat.url_placeholder': 'https://...',
        'chat.go': 'Go',
        'chat.indexed_docs': 'Documents indexés',
        'chat.no_docs': 'Aucun document indexé',
        'chat.filter_source': 'Filtrer la source',
        'chat.welcome': "Bonjour — je suis <strong>MIRROR</strong>, l'assistant IA de Michail. Posez-moi vos questions sur son expérience, ses projets, ou la construction de ce site. Je réponds avec des citations tirées de son CV, ses articles et ses docs d'architecture. Vous pouvez aussi téléverser vos propres documents ou scraper une URL, puis les interroger.",
        'chat.infra_note': "Propulsé par l'API xAI Grok avec retrieval hybride (BM25 · SQLite FTS5). Un plafond de coût journalier protège le budget — si je ne réponds plus, je reviens demain.",
        'chat.ask_placeholder': 'Une question sur Michail, le RAG, cette architecture...',
        'chat.suggestion.experience': "Qu'a construit Michail chez SOMA ?",
        'chat.suggestion.rag': 'Comment fonctionne le RAG de ce site ?',
        'chat.suggestion.lead': "Quelle est son expérience de lead ?",
        'articles.title': 'Articles',
        'articles.subtitle': "Écriture technique sur l'IA, le RAG, le MLOps et les systèmes en production.",
        'articles.loading': 'Chargement des articles...',
        'articles.empty_title': "Pas d'articles pour l'instant",
        'articles.empty_hint': "Ajoute des fichiers markdown dans le dossier articles/ pour commencer.",
        'articles.back': '← Retour aux articles',
        'tech.title': 'Architecture & choix techniques',
        'tech.subtitle': 'Comment MIRROR est construit, ce que ça coûte, et pourquoi — documenté comme un livrable client.',
        'courses.title': 'Cours',
        'courses.subtitle': 'Une vue structurée et claire des sujets clés ML, expliqués avec listes, exemples et intuition.',
        'courses.ml.title': 'Machine Learning (Classique)',
        'courses.ml.intro': 'Le ML classique est souvent la voie la plus rapide vers un baseline solide en production : interprétable, peu coûteux et fiable.',
        'courses.dl.title': 'Deep Learning (Réseaux de neurones)',
        'courses.dl.intro': 'Le deep learning sert surtout à apprendre des représentations. Je regroupe les réseaux par structure : séquences, images, graphes, multimodal.',
        'courses.cv.title': 'Computer Vision',
        'courses.cv.intro': 'Les systèmes vision extraient de la structure depuis des pixels. Le SOTA moderne est dominé par les Transformers et le pré-entraînement.',
        'courses.nlp.title': 'NLP & LLM',
        'courses.nlp.intro': 'Le NLP moderne est basé sur Transformers. La compétence pratique: contrôler le comportement (prompting, retrieval, évaluation, sécurité).',
        'courses.visual.bias_variance': 'Biais / Variance',
        'courses.visual.transformer': 'Transformer (niveau haut)',
        'courses.visual.rag': 'Pipeline RAG',
        'courses.footer': 'MIRROR · Cours · Michail Berjaoui',
        'footer.tagline': 'MIRROR · Portfolio IA · Flask · API xAI Grok · SQLite FTS5 · Docker + Caddy',
        'common.integrality': "L'intégralité",
    },
    ja: {
        'nav.portfolio': 'ポートフォリオ',
        'nav.chat': 'AIチャット',
        'nav.articles': '記事',
        'nav.courses': 'コース',
        'nav.tech': 'アーキテクチャ',
        'home.subtitle': 'リードAI/LLMエンジニア',
        'home.summary': '5年以上、AIシステムを本番導入：エンタープライズRAG、LLM微調整（LoRA/QLoRA）、OCR/NER、MLOps。4〜8名のチームリード経験。東京在住 — 国際チームとフランス企業の両方にオープン。',
        'home.ask_ai': 'AIに私のことを聞く',
        'home.tech_choices': 'このサイトの仕組み',
        'home.projects': '主な実績',
        'home.skills': '技術スキル',
        'home.skill_cat.genai': '生成AI',
        'home.skill_cat.inference': '推論・スケーリング',
        'home.skill_cat.vectorstores': '検索・ベクトル検索',
        'home.skill_cat.mlops': 'MLOps・GPU',
        'home.skill_cat.backend': 'バックエンド・データ',
        'home.skill_cat.cloud': 'クラウド',
        'home.experience': '経験',
        'home.education': '学歴',
        'home.languages': '言語',
        'home.certifications': '資格',
        'home.site_note.title': 'このサイト自体がポートフォリオです',
        'home.site_note.body': 'MIRRORはテンプレートではありません。コスト最適化された本物のAIプロダクトです：Grok APIとハイブリッドBM25検索によるRAGチャットボット、ドキュメントアップロード、Webスクレイピング、マルチユーザーセッション — 小さなVPS上で1日1ドル未満で稼働。アーキテクチャの意思決定はクライアント向けと同じ品質で文書化しています。',
        'home.site_note.cta': 'アーキテクチャの意思決定を読む →',
        'chat.upload_docs': 'ドキュメントをアップロード',
        'chat.drop_here': 'ここにファイルをドロップ',
        'chat.supported_files': 'PDF / DOCX / TXT / MD',
        'chat.web_scraper': 'Webスクレイパー',
        'chat.url_placeholder': 'https://...',
        'chat.go': '実行',
        'chat.indexed_docs': 'インデックス済みドキュメント',
        'chat.no_docs': 'インデックス済みドキュメントはありません',
        'chat.filter_source': 'ソースで絞り込み',
        'chat.welcome': 'こんにちは — 私は<strong>MIRROR</strong>、MichailのAIアシスタントです。彼の経験、プロジェクト、このサイトの構築について何でも質問してください。CV・記事・アーキテクチャ文書から引用付きで回答します。ご自身のドキュメントをアップロードしたり、URLをスクレイプして質問することもできます。',
        'chat.infra_note': 'xAI Grok APIとハイブリッド検索（BM25 · SQLite FTS5）で動作。1日のコスト上限があります — 応答が止まった場合は明日戻ります。',
        'chat.ask_placeholder': 'Michail、RAG、このアーキテクチャについて質問...',
        'chat.suggestion.experience': 'MichailはSOMAで何を作りましたか？',
        'chat.suggestion.rag': 'このサイトのRAGはどう動いていますか？',
        'chat.suggestion.lead': 'チームリードとしての経験は？',
        'articles.title': '記事',
        'articles.subtitle': 'AI、RAG、MLOps、本番システムに関する技術記事。',
        'articles.loading': '記事を読み込み中...',
        'articles.empty_title': '記事はまだありません',
        'articles.empty_hint': 'articles/ ディレクトリにMarkdownファイルを追加してください。',
        'articles.back': '← 記事一覧へ戻る',
        'tech.title': 'アーキテクチャ・技術選定',
        'tech.subtitle': 'MIRRORの構築方法、コスト、その理由 — クライアント向け成果物と同じ品質で文書化。',
        'courses.title': 'コース',
        'courses.subtitle': 'MLの重要トピックを、リストと例で分かりやすく体系化した概要。',
        'courses.ml.title': '機械学習（クラシック）',
        'courses.ml.intro': '古典的MLは本番ベースラインに最適：解釈しやすく、安価で、データが揃えば強い。',
        'courses.dl.title': '深層学習（ニューラルネット）',
        'courses.dl.intro': '深層学習は表現学習が本質。構造（系列/画像/グラフ/マルチモーダル）で整理します。',
        'courses.cv.title': 'Computer Vision',
        'courses.cv.intro': 'Visionはピクセルから構造を抽出。SOTAはTransformerと強い事前学習が中心。',
        'courses.nlp.title': 'NLP & LLM',
        'courses.nlp.intro': '現代NLPはTransformer。実務は挙動制御（プロンプト/RAG/評価/安全）。',
        'courses.visual.bias_variance': 'バイアス/バリアンス',
        'courses.visual.transformer': 'Transformer（概要）',
        'courses.visual.rag': 'RAGパイプライン',
        'courses.footer': 'MIRROR · コース · Michail Berjaoui',
        'footer.tagline': 'MIRROR · AIポートフォリオ · Flask · xAI Grok API · SQLite FTS5 · Docker + Caddy',
        'common.integrality': '全体',
    }
};

function normalizeLang(lang) {
    if (!lang) return null;
    const l = String(lang).toLowerCase();
    if (l.startsWith('fr')) return 'fr';
    if (l.startsWith('ja')) return 'ja';
    if (l.startsWith('en')) return 'en';
    return null;
}

function detectPreferredLang() {
    try {
        const langs = Array.isArray(navigator.languages) ? navigator.languages : [navigator.language];
        const normalized = langs.map((l) => normalizeLang(l)).filter(Boolean);
        if (normalized.length > 0) return normalized[0];
    } catch (e) {}
    return I18N_DEFAULT_LANG;
}

function getStoredLang() {
    try {
        return normalizeLang(localStorage.getItem(I18N_STORAGE_KEY));
    } catch (e) {
        return null;
    }
}

function setStoredLang(lang) {
    try {
        localStorage.setItem(I18N_STORAGE_KEY, lang);
    } catch (e) {}
}

function translateKey(key, lang) {
    const l = normalizeLang(lang) || I18N_DEFAULT_LANG;
    return I18N_DICT[l]?.[key] ?? I18N_DICT[I18N_DEFAULT_LANG]?.[key] ?? null;
}

function applyTranslations(lang) {
    const l = normalizeLang(lang) || I18N_DEFAULT_LANG;
    document.documentElement.setAttribute('lang', l);

    document.querySelectorAll('[data-i18n-html]').forEach((el) => {
        const key = el.getAttribute('data-i18n-html');
        const html = translateKey(key, l);
        if (html) el.innerHTML = html;
    });

    document.querySelectorAll('[data-i18n]').forEach((el) => {
        const key = el.getAttribute('data-i18n');
        const txt = translateKey(key, l);
        if (txt) el.textContent = txt;
    });

    document.querySelectorAll('[data-i18n-placeholder]').forEach((el) => {
        const key = el.getAttribute('data-i18n-placeholder');
        const txt = translateKey(key, l);
        if (txt) el.setAttribute('placeholder', txt);
    });

    const bubbleLabel = document.getElementById('langBubbleLabel');
    if (bubbleLabel) bubbleLabel.textContent = l.toUpperCase();

    document.querySelectorAll('.lang-menu-item[data-lang]').forEach((btn) => {
        btn.classList.toggle('active', normalizeLang(btn.getAttribute('data-lang')) === l);
    });
}

window.MIRROR_I18N = {
    applyTranslations,
    getLang: () => getStoredLang() || detectPreferredLang(),
};

function initLanguageSwitcher() {
    const switchEl = document.getElementById('langSwitch');
    const bubble = document.getElementById('langBubble');
    const menu = document.getElementById('langMenu');
    if (!switchEl || !bubble || !menu) return;

    applyTranslations(getStoredLang() || detectPreferredLang());

    function closeMenu() {
        switchEl.classList.remove('open');
        bubble.setAttribute('aria-expanded', 'false');
    }

    function openMenu() {
        switchEl.classList.add('open');
        bubble.setAttribute('aria-expanded', 'true');
    }

    bubble.addEventListener('click', (e) => {
        e.stopPropagation();
        if (switchEl.classList.contains('open')) closeMenu();
        else openMenu();
    });

    menu.addEventListener('click', (e) => {
        const btn = e.target.closest('.lang-menu-item[data-lang]');
        if (!btn) return;
        const lang = normalizeLang(btn.getAttribute('data-lang')) || I18N_DEFAULT_LANG;
        setStoredLang(lang);
        applyTranslations(lang);
        closeMenu();
        window.dispatchEvent(new CustomEvent('mirror-lang-changed', { detail: { lang } }));
    });

    document.addEventListener('click', () => closeMenu());
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeMenu();
    });
}

document.addEventListener('DOMContentLoaded', initLanguageSwitcher);

// --- Toast Notifications ---
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// --- Service Status (nav dot) ---
async function refreshServiceStatus() {
    const dot = document.getElementById('statusDot');
    if (!dot) return;
    try {
        const resp = await fetch('/api/models/status');
        const data = await resp.json();
        const llmOk = data.llm?.loaded;
        const retrievalOk = data.retrieval?.connected;
        const budgetLeft = data.llm?.daily_budget?.remaining_usd ?? 1;

        if (llmOk && retrievalOk && budgetLeft > 0) {
            dot.className = 'status-dot online';
            dot.title = 'AI assistant online';
        } else if (llmOk && budgetLeft <= 0) {
            dot.className = 'status-dot loading';
            dot.title = 'Daily budget reached — assistant back tomorrow';
        } else {
            dot.className = 'status-dot offline';
            dot.title = 'AI assistant offline';
        }
    } catch (e) {
        dot.className = 'status-dot offline';
        dot.title = 'AI assistant offline';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    refreshServiceStatus();
    setInterval(refreshServiceStatus, 60000);
});
