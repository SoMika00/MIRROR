/* ============================================================
   MIRROR - Global JavaScript
   Handles: model manager, status polling, toast notifications
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
        'nav.tech': 'Tech Choices',
        'home.subtitle': 'Lead AI Engineer | LLM & MLOps',
        'home.summary': '5 years deploying AI models to production at scale. Enterprise RAG specialist, fine-tuning (LoRA/QLoRA), OCR/NER. Based in Tokyo, actively seeking opportunities in Japan.',
        'home.ask_ai': 'Ask the AI about me',
        'home.tech_choices': 'Technical Choices',
        'home.skills': 'Technical Skills',
        'home.skill_cat.genai': 'Generative AI',
        'home.skill_cat.inference': 'Inference & Scaling',
        'home.skill_cat.vectorstores': 'Vector Stores',
        'home.skill_cat.mlops': 'MLOps & GPU',
        'home.skill_cat.backend': 'Backend & Data',
        'home.skill_cat.cloud': 'Cloud',
        'home.experience': 'Experience',
        'home.education': 'Education',
        'home.languages': 'Languages',
        'home.certifications': 'Certifications',
        'chat.upload_docs': 'Upload Documents',
        'chat.drop_here': 'Drop files here',
        'chat.supported_files': 'PDF, DOCX, TXT, MD',
        'chat.web_scraper': 'Web Scraper',
        'chat.url_placeholder': 'https://...',
        'chat.go': 'Go',
        'chat.indexed_docs': 'Indexed Documents',
        'chat.no_docs': 'No documents indexed',
        'chat.scraped_pages': 'Scraped Pages',
        'chat.no_pages': 'No pages scraped',
        'chat.filter_source': 'Filter Source',
        'chat.filter.all': 'All sources',
        'chat.filter.documents': 'Documents only',
        'chat.filter.web': 'Web pages only',
        'chat.welcome': "Welcome! I'm <strong>MIRROR</strong>, Michail's AI assistant. Upload documents or scrape a URL, then ask me anything. I'll answer with source citations.",
        'chat.infra_note': 'Note: MIRROR runs on a low-cost, CPU-only infrastructure. The models used are intentionally small and quantized for fast response times. As a result, answers may occasionally be imprecise or inconsistent - this is a deliberate trade-off for accessibility and cost efficiency, not a reflection of production-grade deployments.',
        'chat.tip': 'Tip: Load a model via the ⚙ button in the top-right corner to enable AI responses. No model downloaded yet - use the model manager to download one first.',
        'chat.ask_placeholder': 'Ask a question...',
        'articles.title': 'Articles',
        'articles.subtitle': 'Technical writing on AI, RAG, MLOps and production systems.',
        'articles.loading': 'Loading articles...',
        'articles.empty_title': 'No articles yet',
        'articles.empty_hint': 'Add markdown files to the articles/ directory to get started.',
        'articles.back': '← Back to articles',
        'tech.title': 'Technical Choices',
        'tech.subtitle': 'Architecture decisions, benchmarks, and infrastructure optimization for MIRROR.',
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
        'footer.tagline': 'MIRROR · AI-Powered Portfolio · Michail Berjaoui',
        'common.integrality': 'The entirety',
    },
    fr: {
        'nav.portfolio': 'Portfolio',
        'nav.chat': 'Chat IA',
        'nav.articles': 'Articles',
        'nav.courses': 'Cours',
        'nav.tech': 'Choix techniques',
        'home.subtitle': 'Lead AI Engineer | LLM & MLOps',
        'home.summary': "5 ans à déployer des modèles d'IA en production à grande échelle. Spécialiste RAG entreprise, fine-tuning (LoRA/QLoRA), OCR/NER. Basé à Tokyo, en recherche active d'opportunités au Japon.",
        'home.ask_ai': "Demander à l'IA à mon sujet",
        'home.tech_choices': 'Choix techniques',
        'home.skills': 'Compétences techniques',
        'home.skill_cat.genai': 'IA générative',
        'home.skill_cat.inference': 'Inférence & passage à l’échelle',
        'home.skill_cat.vectorstores': 'Bases vectorielles',
        'home.skill_cat.mlops': 'MLOps & GPU',
        'home.skill_cat.backend': 'Backend & data',
        'home.skill_cat.cloud': 'Cloud',
        'home.experience': 'Expérience',
        'home.education': 'Formation',
        'home.languages': 'Langues',
        'home.certifications': 'Certifications',
        'chat.upload_docs': 'Téléverser des documents',
        'chat.drop_here': 'Dépose tes fichiers ici',
        'chat.supported_files': 'PDF, DOCX, TXT, MD',
        'chat.web_scraper': 'Scraper web',
        'chat.url_placeholder': 'https://...',
        'chat.go': 'Go',
        'chat.indexed_docs': 'Documents indexés',
        'chat.no_docs': 'Aucun document indexé',
        'chat.scraped_pages': 'Pages scrapées',
        'chat.no_pages': 'Aucune page scrapée',
        'chat.filter_source': 'Filtrer la source',
        'chat.filter.all': 'Toutes les sources',
        'chat.filter.documents': 'Documents uniquement',
        'chat.filter.web': 'Pages web uniquement',
        'chat.welcome': "Bienvenue ! Je suis <strong>MIRROR</strong>, l'assistant IA de Michail. Téléverse des documents ou scrape une URL, puis pose-moi n'importe quelle question. Je répondrai avec des sources.",
        'chat.infra_note': "Note : MIRROR tourne sur une infrastructure low-cost, exclusivement CPU. Les modèles utilisés sont volontairement petits et quantifiés pour garantir des temps de réponse rapides. Les réponses peuvent donc être parfois imprécises ou incohérentes - c'est un compromis assumé entre accessibilité et coût, et non le reflet d'un déploiement de production.",
        'chat.tip': 'Astuce : Chargez un modèle via le bouton ⚙ en haut à droite pour activer les réponses IA. Aucun modèle téléchargé - utilisez le gestionnaire de modèles pour en télécharger un.',
        'chat.ask_placeholder': 'Pose une question...',
        'articles.title': 'Articles',
        'articles.subtitle': "Écriture technique sur l'IA, le RAG, le MLOps et les systèmes en production.",
        'articles.loading': 'Chargement des articles...',
        'articles.empty_title': "Pas d’articles pour l’instant",
        'articles.empty_hint': "Ajoute des fichiers markdown dans le dossier articles/ pour commencer.",
        'articles.back': '← Retour aux articles',
        'tech.title': 'Choix techniques',
        'tech.subtitle': 'Décisions d’architecture, benchmarks et optimisation infra pour MIRROR.',
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
        'footer.tagline': 'MIRROR · Portfolio IA · Michail Berjaoui',
        'common.integrality': "L'intégralité",
    },
    ja: {
        'nav.portfolio': 'ポートフォリオ',
        'nav.chat': 'AIチャット',
        'nav.articles': '記事',
        'nav.courses': 'コース',
        'nav.tech': '技術選定',
        'home.subtitle': 'リードAIエンジニア | LLM & MLOps',
        'home.summary': '5年間、AIモデルを大規模に本番導入。エンタープライズRAG、微調整（LoRA/QLoRA）、OCR/NER。東京在住。日本での機会を積極的に探しています。',
        'home.ask_ai': 'AIに私のことを聞く',
        'home.tech_choices': '技術選定',
        'home.skills': '技術スキル',
        'home.skill_cat.genai': '生成AI',
        'home.skill_cat.inference': '推論・スケーリング',
        'home.skill_cat.vectorstores': 'ベクトルDB',
        'home.skill_cat.mlops': 'MLOps・GPU',
        'home.skill_cat.backend': 'バックエンド・データ',
        'home.skill_cat.cloud': 'クラウド',
        'home.experience': '経験',
        'home.education': '学歴',
        'home.languages': '言語',
        'home.certifications': '資格',
        'chat.upload_docs': 'ドキュメントをアップロード',
        'chat.drop_here': 'ここにファイルをドロップ',
        'chat.supported_files': 'PDF / DOCX / TXT / MD',
        'chat.web_scraper': 'Webスクレイパー',
        'chat.url_placeholder': 'https://...',
        'chat.go': '実行',
        'chat.indexed_docs': 'インデックス済みドキュメント',
        'chat.no_docs': 'インデックス済みドキュメントはありません',
        'chat.scraped_pages': '取得済みページ',
        'chat.no_pages': '取得済みページはありません',
        'chat.filter_source': 'ソースで絞り込み',
        'chat.filter.all': 'すべてのソース',
        'chat.filter.documents': 'ドキュメントのみ',
        'chat.filter.web': 'Webページのみ',
        'chat.welcome': "ようこそ！私は<strong>MIRROR</strong>、MichailのAIアシスタントです。ドキュメントをアップロードするかURLをスクレイプして、何でも質問してください。出典付きで回答します。",
        'chat.infra_note': '注意：MIRRORは低コストのCPU専用インフラで動作しています。高速な応答のため、意図的に小型・量子化モデルを使用しています。そのため、回答が不正確・不安定になる場合がありますが、これはアクセス性とコスト効率を優先した設計上のトレードオフです。',
        'chat.tip': 'ヒント：右上の⚙ボタンからモデルを読み込んでAI応答を有効にしてください。まだモデルがダウンロードされていません - モデルマネージャーでダウンロードしてください。',
        'chat.ask_placeholder': '質問を入力...',
        'articles.title': '記事',
        'articles.subtitle': 'AI、RAG、MLOps、本番システムに関する技術記事。',
        'articles.loading': '記事を読み込み中...',
        'articles.empty_title': '記事はまだありません',
        'articles.empty_hint': 'articles/ ディレクトリにMarkdownファイルを追加してください。',
        'articles.back': '← 記事一覧へ戻る',
        'tech.title': '技術選定',
        'tech.subtitle': 'MIRRORのアーキテクチャ判断、ベンチマーク、インフラ最適化。',
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
        'footer.tagline': 'MIRROR · AIポートフォリオ · Michail Berjaoui',
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
        if (normalized.includes('fr')) return 'fr';
        if (normalized.includes('en')) return 'en';
        if (normalized.includes('ja')) return 'ja';
    } catch (e) {}
    return I18N_DEFAULT_LANG;
}

function getStoredLang() {
    try {
        const v = localStorage.getItem(I18N_STORAGE_KEY);
        return normalizeLang(v);
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

    // Keep bubble label aligned
    const bubbleLabel = document.getElementById('langBubbleLabel');
    if (bubbleLabel) bubbleLabel.textContent = l.toUpperCase();

    // Mark active menu item
    document.querySelectorAll('.lang-menu-item[data-lang]').forEach((btn) => {
        btn.classList.toggle('active', normalizeLang(btn.getAttribute('data-lang')) === l);
    });
}

window.MIRROR_I18N = {
    applyTranslations,
    getLang: () => getStoredLang() || I18N_DEFAULT_LANG,
};

function initLanguageSwitcher() {
    const switchEl = document.getElementById('langSwitch');
    const bubble = document.getElementById('langBubble');
    const menu = document.getElementById('langMenu');
    if (!switchEl || !bubble || !menu) return;

    const initial = getStoredLang() || I18N_DEFAULT_LANG;
    applyTranslations(initial);

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

// --- Model Manager Modal ---
function openModelModal() {
    document.getElementById('modelModal').classList.add('active');
    refreshModelStatus();
}

function closeModelModal() {
    document.getElementById('modelModal').classList.remove('active');
}

document.getElementById('modelManagerBtn')?.addEventListener('click', openModelModal);

document.getElementById('modelModal')?.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal-overlay')) closeModelModal();
});

// --- Status Polling ---
async function refreshModelStatus() {
    try {
        const resp = await fetch('/api/models/status');
        const data = await resp.json();

        // LLM Status (compact header) + show/hide Unload & Test buttons
        const llmEl = document.getElementById('llmStatus');
        const llmActions = document.getElementById('llmActions');
        if (llmEl) {
            if (data.llm.loaded) {
                const name = data.llm.model_name || data.llm.model_path.split('/').pop();
                llmEl.innerHTML = `<span style="color:var(--success);">●</span> ${name}`;
                if (llmActions) llmActions.style.display = 'flex';
            } else {
                llmEl.innerHTML = '<span style="color:var(--text-muted);">○ None</span>';
                if (llmActions) llmActions.style.display = 'none';
            }
        }

        // Model Registry Cards
        await refreshRegistryCards();

        // Embedding Status
        const embEl = document.getElementById('embeddingStatus');
        if (embEl) {
            if (data.embedding.loaded) {
                embEl.innerHTML = `<span style="color:var(--success);">● Loaded</span>`;
            } else {
                embEl.innerHTML = '<span style="color:var(--text-muted);">○ Not loaded</span>';
            }
        }

        // Reranker Status
        const rrEl = document.getElementById('rerankerStatus');
        if (rrEl) {
            if (data.reranker?.loaded) {
                rrEl.innerHTML = `<span style="color:var(--success);">● Loaded</span>`;
            } else {
                rrEl.innerHTML = '<span style="color:var(--text-muted);">○ Not loaded</span>';
            }
        }

        // Qdrant Status
        const qdEl = document.getElementById('qdrantStatus');
        if (qdEl) {
            if (data.qdrant.connected) {
                qdEl.innerHTML = `<span style="color:var(--success);">● ${data.qdrant.vectors_count || 0} vecs</span>`;
            } else {
                qdEl.innerHTML = '<span style="color:var(--text-muted);">○ Disconnected</span>';
            }
        }

        // Vision Status
        const visEl = document.getElementById('visionStatus');
        if (visEl) {
            if (data.vision?.loaded) {
                visEl.innerHTML = `<span style="color:var(--success);">● Loaded</span>`;
            } else if (data.vision?.loading) {
                visEl.innerHTML = '<span style="color:orange;">● Loading...</span>';
            } else if (data.vision?.enabled) {
                visEl.innerHTML = '<span style="color:var(--text-muted);">○ Available</span>';
            } else {
                visEl.innerHTML = '<span style="color:var(--text-muted);">○ Disabled</span>';
            }
        }

        // Global status dot
        updateGlobalStatus(data);

    } catch (e) {
        console.error('Status check failed:', e);
    }
}

async function refreshRegistryCards() {
    const container = document.getElementById('modelRegistry');
    if (!container) return;
    try {
        const resp = await fetch('/api/models/llm/registry');
        const data = await resp.json();
        if (!data.models || data.models.length === 0) {
            container.innerHTML = '<div class="text-muted">No models in registry</div>';
            return;
        }
        // Group by family
        const families = {};
        data.models.forEach(m => {
            if (!families[m.family]) families[m.family] = [];
            families[m.family].push(m);
        });
        let html = '';
        for (const [family, models] of Object.entries(families)) {
            html += `<div style="margin-bottom:0.3rem;"><span style="font-size:0.75rem;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:0.05em;">${family}</span></div>`;
            models.forEach(m => {
                const isActive = m.active;
                const isDownloaded = m.downloaded;
                const borderColor = isActive ? 'var(--success)' : isDownloaded ? 'var(--accent-dim, rgba(139,92,246,0.3))' : 'var(--border)';
                const bgColor = isActive ? 'rgba(16,185,129,0.08)' : 'transparent';

                let actionBtn = '';
                if (isActive) {
                    actionBtn = `<span style="color:var(--success);font-size:0.75rem;font-weight:600;">● Active</span>`;
                } else if (isDownloaded) {
                    actionBtn = `<button class="btn btn-primary btn-sm" onclick="loadModelById('${m.id}')" style="font-size:0.7rem;padding:0.2rem 0.6rem;">Load</button>`;
                } else {
                    actionBtn = `<span style="color:var(--text-muted);font-size:0.7rem;">Not on disk</span>`;
                }

                html += `<div class="model-card" style="border-color:${borderColor};background:${bgColor};" id="card-${m.id}">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <div style="font-weight:600;font-size:0.85rem;">${m.name}${m.default ? ' <span style="font-size:0.65rem;color:var(--accent);">(default)</span>' : ''}</div>
                            <div style="font-size:0.72rem;color:var(--text-muted);margin-top:0.1rem;">${m.description}</div>
                        </div>
                        <div style="text-align:right;min-width:80px;">
                            ${actionBtn}
                        </div>
                    </div>
                    <div style="display:flex;gap:0.8rem;margin-top:0.3rem;font-size:0.7rem;color:var(--text-muted);">
                        <span>RAM: ~${m.ram_gb} GB</span>
                        <span>Speed: ${m.speed_estimate}</span>
                        <span>${m.quant}</span>
                    </div>
                </div>`;
            });
        }
        container.innerHTML = html;
    } catch (e) {
        container.innerHTML = '<div class="text-muted">Failed to load registry</div>';
    }
}

async function downloadModelById(modelId) {
    showOpBar('Starting download from HuggingFace...', 'indeterminate');
    const card = document.getElementById(`card-${modelId}`);
    const btn = card?.querySelector('.btn');
    if (btn) { btn.disabled = true; btn.textContent = 'Downloading...'; }
    try {
        const resp = await fetch('/api/models/llm/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId })
        });
        const data = await resp.json();
        if (data.error) {
            showOpBarError(`Download failed: ${data.error}`);
            if (btn) { btn.disabled = false; btn.textContent = 'Download'; }
            return;
        }
        if (data.message === 'Already downloaded') {
            showOpBarDone('Model already downloaded');
            refreshModelStatus();
            return;
        }

        // Poll progress with bottom bar
        let lastPct = -1;
        for (let i = 0; i < 600; i++) {
            await new Promise(r => setTimeout(r, 2000));
            try {
                const pr = await fetch(`/api/models/llm/download-progress/${modelId}`);
                const p = await pr.json();
                if (p.status === 'done') {
                    showOpBarDone('Download complete');
                    break;
                }
                if (p.status === 'error') {
                    showOpBarError(`Download error: ${p.error}`);
                    break;
                }
                const pct = Math.round((p.progress || 0) * 100);
                if (pct !== lastPct) {
                    lastPct = pct;
                    updateOpBar(`Downloading model... ${pct}%`, pct);
                    if (btn) btn.textContent = `${pct}%`;
                }
            } catch (e) { /* ignore poll errors */ }
        }

        refreshModelStatus();
    } catch (e) {
        showOpBarError(`Error: ${e.message}`);
        refreshModelStatus();
    }
}

function updateGlobalStatus(data) {
    const dot = document.getElementById('statusDot');
    if (!dot) return;

    const llmOk = data.llm?.loaded;
    const embOk = data.embedding?.loaded;
    const qdOk = data.qdrant?.connected;

    if (llmOk && embOk && qdOk) {
        dot.className = 'status-dot online';
        dot.title = 'All services online';
    } else if (llmOk || embOk || qdOk) {
        dot.className = 'status-dot loading';
        dot.title = 'Some services loading';
    } else {
        dot.className = 'status-dot offline';
        dot.title = 'Services offline';
    }
}

// --- Bottom Operation Bar ---
function showOpBar(text, mode = 'indeterminate') {
    const bar = document.getElementById('modelOpBar');
    const textEl = document.getElementById('modelOpText');
    const fill = document.getElementById('modelOpFill');
    if (!bar) return;
    textEl.textContent = text;
    bar.className = 'model-op-bar visible ' + mode;
    if (mode === 'indeterminate') fill.style.width = '30%';
    else fill.style.width = '0%';
}

function updateOpBar(text, pct) {
    const textEl = document.getElementById('modelOpText');
    const fill = document.getElementById('modelOpFill');
    const bar = document.getElementById('modelOpBar');
    if (textEl) textEl.textContent = text;
    if (fill) fill.style.width = pct + '%';
    if (bar) bar.className = 'model-op-bar visible';
}

function hideOpBar(delay = 2500) {
    setTimeout(() => {
        const bar = document.getElementById('modelOpBar');
        if (bar) bar.className = 'model-op-bar';
    }, delay);
}

function showOpBarDone(text) {
    const bar = document.getElementById('modelOpBar');
    const textEl = document.getElementById('modelOpText');
    const fill = document.getElementById('modelOpFill');
    if (textEl) textEl.textContent = text;
    if (fill) fill.style.width = '100%';
    if (bar) bar.className = 'model-op-bar visible done';
    hideOpBar(3000);
}

function showOpBarError(text) {
    const bar = document.getElementById('modelOpBar');
    const textEl = document.getElementById('modelOpText');
    if (textEl) textEl.textContent = text;
    if (bar) bar.className = 'model-op-bar visible error';
    hideOpBar(5000);
}

// --- Live Metrics Widget ---
let _metricsTimer = null;
async function refreshLiveMetrics() {
    const ramEl = document.getElementById('metricRam');
    const cpuEl = document.getElementById('metricCpu');
    const gpuEl = document.getElementById('metricGpu');
    const gpuRow = document.getElementById('metricGpuRow');
    const processEl = document.getElementById('metricProcess');
    const coresRow = document.getElementById('metricCoresRow');
    const coresEl = document.getElementById('metricCores');
    const infRow = document.getElementById('metricInferenceRow');
    const infEl = document.getElementById('metricInference');
    if (!ramEl || !cpuEl) return;

    try {
        const resp = await fetch('/api/models/metrics');
        const data = await resp.json();

        // RAM
        const ramPct = data?.ram?.percent;
        ramEl.textContent = (ramPct === 0 || ramPct) ? `${ramPct}%` : '--%';
        if (ramPct > 70) ramEl.style.color = 'var(--warning)';
        else if (ramPct > 90) ramEl.style.color = 'var(--error)';
        else ramEl.style.color = '';

        // Process RSS
        if (processEl && data?.process?.rss_gb !== undefined) {
            processEl.textContent = `${data.process.rss_gb} GB`;
        }

        // CPU (aggregate)
        const cpuPct = data?.cpu_percent;
        cpuEl.textContent = (cpuPct === 0 || cpuPct) ? `${cpuPct}%` : '--%';
        if (cpuPct > 70) cpuEl.style.color = 'var(--accent-light)';
        else if (cpuPct > 90) cpuEl.style.color = 'var(--error)';
        else cpuEl.style.color = '';

        // Per-core CPU bars
        const cores = data?.cpu_per_core;
        if (coresRow && coresEl && cores && cores.length > 0) {
            coresRow.style.display = '';
            let html = '';
            cores.forEach((pct, i) => {
                const color = pct > 80 ? 'var(--accent-light)' : pct > 50 ? 'var(--accent)' : 'var(--border)';
                html += `<div class="core-bar" title="Core ${i}: ${pct}%"><div class="core-bar-fill" style="height:${Math.max(2, pct)}%;background:${color};"></div></div>`;
            });
            coresEl.innerHTML = html;
        }

        // Inference telemetry
        const inf = data?.inference;
        if (infRow && infEl) {
            if (inf?.active) {
                infRow.style.display = '';
                infEl.textContent = `${inf.tokens_per_sec} t/s · ${inf.tokens_generated} tok`;
                infEl.style.color = 'var(--success)';
            } else if (inf?.tokens_generated > 0) {
                infRow.style.display = '';
                infEl.textContent = `${inf.tokens_per_sec} t/s · ${inf.tokens_generated} tok`;
                infEl.style.color = 'var(--accent-light)';
            } else {
                infRow.style.display = 'none';
            }
        }

        // GPU
        const gpuPct = data?.gpu?.gpu_percent;
        if (gpuRow) {
            if (gpuPct === 0 || gpuPct) {
                gpuRow.style.display = '';
                if (gpuEl) gpuEl.textContent = `${gpuPct}%`;
            } else {
                gpuRow.style.display = 'none';
            }
        }
    } catch (e) {
        // ignore
    }
}

function startLiveMetrics() {
    if (_metricsTimer) return;
    refreshLiveMetrics();
    // Poll every 1s for responsive real-time metrics during inference
    _metricsTimer = setInterval(refreshLiveMetrics, 1000);
}

document.addEventListener('DOMContentLoaded', startLiveMetrics);

// --- Model Load Status Polling ---
let _loadPollTimer = null;
function startLoadPoll() {
    stopLoadPoll();
    _loadPollTimer = setInterval(async () => {
        try {
            const resp = await fetch('/api/models/llm/load-status');
            const s = await resp.json();
            if (s.status === 'loading') {
                showOpBar(s.step || 'Loading model...', 'indeterminate');
            } else if (s.status === 'done') {
                showOpBarDone(`${s.model_name || 'Model'} loaded successfully`);
                stopLoadPoll();
                refreshModelStatus();
            } else if (s.status === 'error') {
                showOpBarError(`Load failed: ${s.error || 'Unknown error'}`);
                stopLoadPoll();
                refreshModelStatus();
            } else {
                stopLoadPoll();
            }
        } catch (e) { /* ignore poll errors */ }
    }, 1000);
}
function stopLoadPoll() {
    if (_loadPollTimer) { clearInterval(_loadPollTimer); _loadPollTimer = null; }
}

// --- Model Actions ---
async function loadLLM() {
    showOpBar('Loading default LLM...', 'indeterminate');
    try {
        const resp = await fetch('/api/models/llm/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        const data = await resp.json();
        if (data.success) startLoadPoll();
        else { showOpBarError(`Load failed: ${data.error}`); refreshModelStatus(); }
    } catch (e) {
        showOpBarError(`Error: ${e.message}`);
    }
}

async function loadModelById(modelId) {
    showOpBar('Preparing to load model...', 'indeterminate');
    const card = document.getElementById(`card-${modelId}`);
    const btn = card?.querySelector('.btn-primary');
    if (btn) { btn.disabled = true; btn.textContent = 'Loading...'; }
    try {
        const resp = await fetch('/api/models/llm/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId })
        });
        const data = await resp.json();
        if (data.success) startLoadPoll();
        else { showOpBarError(`Failed: ${data.error}`); refreshModelStatus(); }
    } catch (e) {
        showOpBarError(`Error: ${e.message}`);
        refreshModelStatus();
    }
}

async function loadLLMModel(modelName) {
    showOpBar(`Loading ${modelName}...`, 'indeterminate');
    try {
        const resp = await fetch('/api/models/llm/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_path: `./models/${modelName}` })
        });
        const data = await resp.json();
        if (data.success) startLoadPoll();
        else { showOpBarError(`Failed: ${data.error}`); refreshModelStatus(); }
    } catch (e) {
        showOpBarError(`Error: ${e.message}`);
    }
}

async function unloadLLM() {
    showOpBar('Unloading model...', 'indeterminate');
    try {
        await fetch('/api/models/llm/unload', { method: 'POST' });
        showOpBarDone('Model unloaded');
        refreshModelStatus();
    } catch (e) {
        showOpBarError(`Error: ${e.message}`);
    }
}

async function testLLM() {
    const resultEl = document.getElementById('testResult');
    if (resultEl) resultEl.textContent = 'Testing...';
    try {
        const resp = await fetch('/api/models/llm/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: 'Say hello in exactly one sentence.' })
        });
        const data = await resp.json();
        if (data.error) {
            if (resultEl) resultEl.textContent = `Error: ${data.error}`;
        } else {
            if (resultEl) resultEl.textContent = `${data.elapsed_seconds}s - ~${data.estimated_tokens} tokens - "${data.response.substring(0, 80)}..."`;
        }
    } catch (e) {
        if (resultEl) resultEl.textContent = `Failed: ${e.message}`;
    }
}

async function loadEmbedding() {
    showToast('Loading embedding model (may take a minute)...', 'info');
    try {
        const resp = await fetch('/api/models/embedding/load', { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            showToast('Embedding model loaded', 'success');
        } else {
            showToast(`Failed: ${data.error}`, 'error');
        }
        refreshModelStatus();
    } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
    }
}

async function connectQdrant() {
    try {
        const resp = await fetch('/api/models/qdrant/connect', { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            showToast('Qdrant connected', 'success');
        } else {
            showToast(`Failed: ${data.error}`, 'error');
        }
        refreshModelStatus();
    } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
    }
}

async function loadReranker() {
    showToast('Loading reranker...', 'info');
    try {
        const resp = await fetch('/api/models/reranker/load', { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            showToast('Reranker loaded', 'success');
        } else {
            showToast(`Failed: ${data.error}`, 'error');
        }
        refreshModelStatus();
    } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
    }
}

async function loadVision() {
    showToast('Loading vision model...', 'info');
    try {
        const resp = await fetch('/api/models/vision/load', { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            showToast('Vision model loaded', 'success');
        } else {
            showToast(`Failed: ${data.error}`, 'error');
        }
        refreshModelStatus();
    } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
    }
}

async function unloadVision() {
    try {
        await fetch('/api/models/vision/unload', { method: 'POST' });
        showToast('Vision model unloaded', 'info');
        refreshModelStatus();
    } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
    }
}

async function loadAll() {
    showOpBar('Loading all services...', 'indeterminate');
    try {
        const resp = await fetch('/api/models/load-all', { method: 'POST' });
        const data = await resp.json();
        const r = data.results || {};
        const hasError = Object.values(r).some(v => v.startsWith?.('error'));
        let msg = Object.entries(r).map(([k, v]) => `${k}: ${v}`).join(' | ');
        if (hasError) showOpBarError(msg);
        else showOpBarDone('All services loaded');
        refreshModelStatus();
    } catch (e) {
        showOpBarError(`Error: ${e.message}`);
    }
}

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
    // Initial status check
    refreshModelStatus();
    // Poll every 30s
    setInterval(refreshModelStatus, 30000);
});
