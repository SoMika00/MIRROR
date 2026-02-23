/* ============================================================
   MIRROR — Global JavaScript
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
        'footer.tagline': 'MIRROR — AI-powered portfolio by Michail Berjaoui · Built with Flask, Qdrant, BGE-M3 & Phi-4',
        'common.integrality': 'The entirety',
    },
    fr: {
        'nav.portfolio': 'Portfolio',
        'nav.chat': 'Chat IA',
        'nav.articles': 'Articles',
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
        'chat.welcome': "Bienvenue ! Je suis <strong>MIRROR</strong>, l'assistant IA de Michail. Téléverse des documents ou scrape une URL, puis pose-moi n’importe quelle question. Je répondrai avec des sources.",
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
        'footer.tagline': 'MIRROR — Portfolio boosté à l’IA par Michail Berjaoui · Construit avec Flask, Qdrant, BGE-M3 & Phi-4',
        'common.integrality': "L'intégralité",
    },
    ja: {
        'nav.portfolio': 'ポートフォリオ',
        'nav.chat': 'AIチャット',
        'nav.articles': '記事',
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
        'footer.tagline': 'MIRROR — Michail Berjaoui のAI搭載ポートフォリオ · Flask / Qdrant / BGE-M3 / Phi-4',
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
    getLang: () => getStoredLang() || detectPreferredLang(),
};

function initLanguageSwitcher() {
    const switchEl = document.getElementById('langSwitch');
    const bubble = document.getElementById('langBubble');
    const menu = document.getElementById('langMenu');
    if (!switchEl || !bubble || !menu) return;

    const initial = getStoredLang() || detectPreferredLang();
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

        // LLM Status
        const llmEl = document.getElementById('llmStatus');
        if (llmEl) {
            if (data.llm.loaded) {
                const modelName = data.llm.model_path.split('/').pop();
                llmEl.innerHTML = `<span style="color:var(--success);">● Loaded</span> <span class="text-muted" style="font-size:0.8rem;">${modelName}</span>`;
            } else {
                llmEl.innerHTML = '<span style="color:var(--text-muted);">○ Not loaded</span>';
            }
        }

        // Available models
        const modelsEl = document.getElementById('availableModels');
        if (modelsEl) {
            const modelsResp = await fetch('/api/models/llm/list');
            const modelsData = await modelsResp.json();
            if (modelsData.models && modelsData.models.length > 0) {
                modelsEl.innerHTML = modelsData.models.map(m =>
                    `<div class="doc-item" style="cursor:pointer;" onclick="loadLLMModel('${m}')">
                        <span class="doc-item-name">${m}</span>
                        <span class="text-muted" style="font-size:0.7rem;">click to load</span>
                    </div>`
                ).join('');
            } else {
                modelsEl.innerHTML = '<div class="text-muted" style="font-size:0.8rem;">No .gguf files in ./models/</div>';
            }
        }

        // Embedding Status
        const embEl = document.getElementById('embeddingStatus');
        if (embEl) {
            if (data.embedding.loaded) {
                embEl.innerHTML = `<span style="color:var(--success);">● Loaded</span> <span class="text-muted" style="font-size:0.8rem;">${data.embedding.model}</span>`;
            } else {
                embEl.innerHTML = '<span style="color:var(--text-muted);">○ Not loaded</span>';
            }
        }

        // Reranker Status
        const rrEl = document.getElementById('rerankerStatus');
        if (rrEl) {
            if (data.reranker?.loaded) {
                rrEl.innerHTML = `<span style="color:var(--success);">● Loaded</span> <span class="text-muted" style="font-size:0.8rem;">${data.reranker.model}</span>`;
            } else {
                rrEl.innerHTML = '<span style="color:var(--text-muted);">○ Not loaded</span>';
            }
        }

        // Qdrant Status
        const qdEl = document.getElementById('qdrantStatus');
        if (qdEl) {
            if (data.qdrant.connected) {
                qdEl.innerHTML = `<span style="color:var(--success);">● Connected</span> <span class="text-muted" style="font-size:0.8rem;">${data.qdrant.vectors_count || 0} vectors</span>`;
            } else {
                qdEl.innerHTML = '<span style="color:var(--text-muted);">○ Disconnected</span>';
            }
        }

        // Vision Status
        const visEl = document.getElementById('visionStatus');
        if (visEl) {
            if (data.vision?.loaded) {
                visEl.innerHTML = `<span style="color:var(--success);">● Loaded</span> <span class="text-muted" style="font-size:0.8rem;">${data.vision.model}</span>`;
            } else if (data.vision?.enabled) {
                visEl.innerHTML = '<span style="color:var(--text-muted);">○ Enabled, not loaded</span>';
            } else {
                visEl.innerHTML = '<span style="color:var(--text-muted);">○ Disabled (VISION_ENABLED=0)</span>';
            }
        }

        // Global status dot
        updateGlobalStatus(data);

    } catch (e) {
        console.error('Status check failed:', e);
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

// --- Model Actions ---
async function loadLLM() {
    showToast('Loading LLM...', 'info');
    try {
        const resp = await fetch('/api/models/llm/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        const data = await resp.json();
        if (data.success) {
            showToast('LLM loaded successfully', 'success');
        } else {
            showToast(`LLM load failed: ${data.error}`, 'error');
        }
        refreshModelStatus();
    } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
    }
}

async function loadLLMModel(modelName) {
    showToast(`Loading ${modelName}...`, 'info');
    try {
        const resp = await fetch('/api/models/llm/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_path: `./models/${modelName}` })
        });
        const data = await resp.json();
        if (data.success) {
            showToast(`${modelName} loaded`, 'success');
        } else {
            showToast(`Failed: ${data.error}`, 'error');
        }
        refreshModelStatus();
    } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
    }
}

async function unloadLLM() {
    try {
        await fetch('/api/models/llm/unload', { method: 'POST' });
        showToast('LLM unloaded', 'info');
        refreshModelStatus();
    } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
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
            if (resultEl) resultEl.textContent = `${data.elapsed_seconds}s — ~${data.estimated_tokens} tokens — "${data.response.substring(0, 80)}..."`;
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
    showToast('Loading all services... this may take a few minutes', 'info');
    try {
        const resp = await fetch('/api/models/load-all', { method: 'POST' });
        const data = await resp.json();
        const r = data.results || {};
        let msg = Object.entries(r).map(([k, v]) => `${k}: ${v}`).join(' | ');
        showToast(msg, 'success');
        refreshModelStatus();
    } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
    }
}

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
    // Initial status check
    refreshModelStatus();
    // Poll every 30s
    setInterval(refreshModelStatus, 30000);
});
