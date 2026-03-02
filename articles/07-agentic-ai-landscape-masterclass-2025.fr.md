---
title: "Le Paysage de l'IA Agentique : Des Chatbots aux Systèmes Autonomes (2025)"
date: 2025-11-15
tags: Agents, IA, OpenAI, Google, Anthropic, MCP, Multi-Agent, Orchestration, Production, Recherche, LangGraph, CrewAI, AutoGen
summary: "Vue d'ensemble pratique du paysage de l'IA agentique en 2025. Cartographie concurrentielle d'OpenAI, Google, Anthropic, Microsoft et HuggingFace, comparaison de 10+ frameworks, philosophies architecturales, et les patterns de production qui comptent pour construire de vrais systèmes."
---

# Le Paysage de l'IA Agentique (2025)
## Des Chatbots aux Systèmes Autonomes - Vue d'Ensemble Pratique

> *"Le prochain grand virage de l'IA n'est pas des modèles plus gros - c'est de donner aux modèles la capacité d'agir."*
> - Dario Amodei, CEO Anthropic, Octobre 2025

---

## Résumé

L'industrie de l'IA traverse sa mutation architecturale la plus significative depuis le papier Transformer (2017). Nous passons d'interactions **stateless prompt→réponse** à des **agents autonomes stateful, multi-étapes, utilisant des outils**, capables de planifier, exécuter, valider et récupérer d'erreurs.

Cet article couvre ce que j'ai appris en explorant et expérimentant avec l'IA agentique :

1. **Paysage concurrentiel** - OpenAI, Google DeepMind, Anthropic, Microsoft, Meta, HuggingFace
2. **Comparaison de 10+ frameworks** - architecture, primitives, compromis, maturité production
3. **Trois philosophies concurrentes** - verrouillage propriétaire vs. protocole d'abord vs. agnostique au modèle
4. **Patterns de production** - ce qui marche vraiment quand on dépasse le stade de la démo
5. **Perspectives 2026** - où la frontière de la recherche nous mène

---

## 1. Qu'est-ce qu'un Agent IA ?

Un agent IA est un système logiciel où un **modèle de langage agit comme couche d'orchestration**, décidant dynamiquement :

- **Quel outil appeler** (APIs, bases de données, interpréteurs de code, autres agents)
- **Quand déléguer** (transfert vers des sous-agents spécialisés)
- **Comment valider** (vérifier les sorties par rapport à des contraintes avant de retourner le résultat)
- **Quand s'arrêter** (réponse finale vs. continuer le raisonnement)

C'est fondamentalement différent d'un chatbot :

| Propriété | Chatbot | Agent |
|----------|---------|-------|
| État | Stateless (ou simple fenêtre de contexte) | Stateful (mémoire persistante, état de session) |
| Actions | Génération de texte uniquement | Appels d'outils, exécution de code, appels API, délégation |
| Flux de contrôle | Un seul appel LLM | Boucle multi-étapes avec branchements et récupération |
| Validation | Aucune | Guardrails, validation des sorties, humain dans la boucle |
| Composition | Monolithique | Orchestration multi-agents |

La boucle canonique d'un agent (tirée de la recherche "Building Effective Agents" d'Anthropic, 2025) :

```python
while not done:
    observation = gather_context(environment, memory)
    plan = llm.reason(observation, goal, history)
    action = plan.next_action()
    result = execute(action)  # appel d'outil, délégation, ou réponse finale
    memory.update(action, result)
    done = evaluate(result, goal)
```

### Références
- Yao et al. (2023). *"ReAct: Synergizing Reasoning and Acting in Language Models"*. ICLR 2023
- Shinn et al. (2023). *"Reflexion: Language Agents with Verbal Reinforcement Learning"*. NeurIPS 2023
- Anthropic (2025). *"Building Effective Agents"*. anthropic.com/research

---

## 2. Les Trois Philosophies Concurrentes

L'espace de l'IA agentique est dominé par trois approches stratégiques radicalement différentes :

### 2.1 OpenAI : La Stratégie de Plateforme Propriétaire

**Philosophie** : Construire les meilleurs modèles ET les meilleurs outils. Verrouiller les développeurs dans l'écosystème OpenAI avec une intégration transparente entre modèles, outils, traçabilité et déploiement.

**Mouvements clés** :
- **Agents SDK** (Mars 2025) - Python-first, primitives minimales (Agent, Runner, Handoff, Guardrail)
- **Responses API** - remplace Chat Completions comme API d'inférence principale, supporte nativement l'appel d'outils, la recherche web, la recherche de fichiers et l'utilisation d'ordinateur
- **Outils intégrés** - `WebSearchTool`, `FileSearchTool`, `CodeInterpreterTool`, `ComputerTool` - tous hébergés par OpenAI
- **Traçabilité & Évaluation** - chaque exécution d'agent automatiquement tracée dans le Dashboard OpenAI, alimente le pipeline de fine-tuning
- **API Temps Réel** - agents vocaux avec latence <300ms, détection de tours de parole et d'interruptions intégrée

**Forces** : Chemin le plus rapide de l'idée à l'agent fonctionnel. Boilerplate minimal. Modèles de pointe (GPT-4o, o3, o4-mini). Traçabilité intégrée.

**Faiblesses** : Verrouillage complet au modèle. Pas d'option locale/auto-hébergée. Tarification opaque pour les boucles d'agents complexes (chaque appel d'outil = appel API). Les données quittent votre infrastructure.

**Pattern architectural** : Hub-and-spoke. OpenAI est le hub ; tout se connecte à leur API.

### 2.2 Anthropic : Protocole d'Abord, Standards Ouverts

**Philosophie** : Ne pas construire la plateforme - construire les **protocoles** que chaque plateforme utilise. Si les standards sont universels, Claude n'a qu'à être le meilleur modèle à les utiliser.

**Mouvements clés** :
- **Model Context Protocol (MCP)** (Novembre 2024) - protocole ouvert pour connecter les LLMs à des outils et sources de données externes. Désormais adopté par OpenAI, Google, Microsoft, et plus de 50 fournisseurs d'outils
- **Claude Agent SDK** (Septembre 2025) - couche d'orchestration légère construite sur MCP
- **Computer Use** (Octobre 2024) - Claude peut contrôler un ordinateur via captures d'écran + actions souris/clavier
- **Agent Skills** (2025) - couche de connaissances procédurales donnée aux fondations ouvertes
- **Extended Thinking** - chaîne de pensée visible pour le développeur pour déboguer le raisonnement de l'agent

**Forces** : L'approche protocole-first crée un verrouillage d'écosystème sans verrouillage de modèle. MCP est en train de devenir l'USB-C de l'intégration d'outils IA. Le suivi d'instructions de Claude est le meilleur de sa catégorie pour les tâches agentiques.

**Faiblesses** : Écosystème plus petit qu'OpenAI. MCP est encore en maturation (découverte de serveurs, authentification). Pas d'orchestration multi-agents intégrée (par choix - ils préfèrent les patterns simples).

**Pattern architectural** : Maillage de protocoles. Des serveurs MCP partout ; n'importe quel modèle peut se connecter à n'importe quel outil.

### 2.3 Google DeepMind : Plateforme Multi-Agents Entreprise

**Philosophie** : Framework agnostique au modèle optimisé pour Google Cloud, avec les primitives d'orchestration multi-agents les plus sophistiquées.

**Mouvements clés** :
- **Agent Development Kit (ADK)** (Avril 2025) - framework modulaire avec agents Séquentiels, Parallèles et en Boucle
- **Gemini 2.0** - appels d'outils natifs, exécution de code, grounding avec Google Search
- **Vertex AI Agent Engine** - déploiement managé pour agents en production avec gestion de sessions intégrée
- **Protocole A2A** (Agent-to-Agent) - protocole ouvert pour la communication inter-agents entre organisations
- **NotebookLM** - agent RAG productisé (200M+ utilisateurs fin 2025)

**Forces** : Primitives multi-agents les plus sophistiquées (`SequentialAgent`, `ParallelAgent`, `LoopAgent`). Agnostique au modèle (fonctionne avec Gemini, GPT, Claude, modèles locaux). Gestion de sessions enterprise-grade. Intégration Google Cloud.

**Faiblesses** : API plus verbeuse qu'OpenAI. Biais Google Cloud. Adoption du protocole A2A encore précoce. Écart de qualité du modèle Gemini vs GPT-4o/Claude pour certaines tâches.

**Pattern architectural** : Plateforme d'orchestration entreprise. Google Cloud est le plan de contrôle.

---

## 3. Matrice de Comparaison des Frameworks (10+ Frameworks)

| Framework | Mainteneur | Architecture | Multi-Agent | MCP | Mémoire | Guardrails | Maturité Production |
|-----------|-----------|-------------|------------|-----|---------|-----------|---------------------|
| **OpenAI Agents SDK** | OpenAI | Boucle d'agent + handoffs | Handoffs, agents-comme-outils | Oui | Sessions (intégré) | Guardrails entrée/sortie | Élevée (infra OpenAI) |
| **Google ADK** | Google | Événementiel, hiérarchique | Agents Séquentiels/Parallèles/Boucle | Oui | SessionService (modulable) | Système de callbacks | Élevée (déploiement Vertex AI) |
| **Claude Agent SDK** | Anthropic | Orchestration native MCP | Via délégation d'outils MCP | Natif | Basé MCP | Extended thinking + callbacks | Moyenne-Haute |
| **LangGraph** | LangChain | Machine à états / graphe | Topologies de graphe arbitraires | Via outils | Checkpointing, persistance | Nœuds personnalisés | Élevée (le plus flexible) |
| **CrewAI** | CrewAI Inc. | Équipes basées sur les rôles | Agents avec rôles/objectifs/backstories | Via outils | Mémoire court/long terme | Validation des tâches | Moyenne |
| **AutoGen** | Microsoft | Basé sur la conversation | Chat multi-agents + exécution de code | Via outils | Historique de conversation | Sandbox de code | Moyenne (orienté recherche) |
| **Semantic Kernel** | Microsoft | Architecture de plugins | Multi-agent via Handlebars | Via plugins | KernelMemory | Filtres/middleware | Élevée (entreprise, focus .NET) |
| **Smolagents** | HuggingFace | Code-first, minimal | CodeAgent + ToolCallingAgent | Via outils | Basé sur les étapes | Exécution sandboxée | Moyenne |
| **PydanticAI** | Équipe Pydantic | Agents type-safe | Via injection de dépendances | Via outils | Types de résultats | Validation de types | Moyenne-Haute |
| **mcp-agent** | lastmileai | Orchestration native MCP | Patterns routeur + orchestrateur | Natif | Sessions MCP | Basé MCP | Moyenne |
| **Mastra** | Mastra | TypeScript-first | Moteur de workflows | Oui | Basé sur base de données | Middleware | Moyenne |
| **Bee Agent** | i.inc | Événementiel | Composition de modules | Via outils | Mémoire sérialisable | Basé événements | Moyenne |

### Analyse Approfondie : Le Top 5

#### OpenAI Agents SDK - Rapidité vers la Production

```python
from agents import Agent, Runner, function_tool

@function_tool
def search_docs(query: str) -> list[dict]:
    """Search the knowledge base."""
    return vector_store.search(query, top_k=5)

agent = Agent(
    name="Assistant",
    instructions="Answer using the knowledge base. Cite sources.",
    tools=[search_docs],
    output_guardrails=[fact_check_guardrail],
)

result = Runner.run_sync(agent, "How does HNSW indexing work?")
```

3 lignes pour définir un agent. Le `Runner` gère la boucle entière : appel LLM → exécution d'outil → réinjection du résultat → répéter jusqu'à la sortie finale. Les guardrails s'exécutent en parallèle avec l'agent pour zéro latence supplémentaire.

#### LangGraph - Flexibilité Maximale

```python
from langgraph.graph import StateGraph, MessagesState

def researcher(state: MessagesState):
    # Appel LLM avec outils de recherche
    return {"messages": [research_result]}

def writer(state: MessagesState):
    # Synthétiser la recherche en réponse
    return {"messages": [written_answer]}

def should_continue(state: MessagesState):
    if needs_more_research(state):
        return "researcher"
    return "writer"

graph = StateGraph(MessagesState)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_conditional_edges("researcher", should_continue)
graph.add_edge("writer", END)

app = graph.compile(checkpointer=SqliteSaver("checkpoints.db"))
```

LangGraph modélise les agents comme des **machines à états avec checkpointing**. Toute topologie est possible : cycles, branches, exécution parallèle, interruptions humain-dans-la-boucle. Le compromis est la complexité - vous construisez un graphe, pas un script.

#### MCP d'Anthropic - Le Protocole Universel

```python
# Serveur MCP (fournisseur d'outils)
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("knowledge-base")

@server.tool()
async def search(query: str) -> list[TextContent]:
    """Search the vector database."""
    results = await qdrant.search(query)
    return [TextContent(text=r.text) for r in results]

# Client MCP (n'importe quel framework agentique)
# L'agent se connecte a n'importe quel serveur MCP sans adaptateurs specifiques
```

La puissance de MCP est l'**interopérabilité** : construisez un serveur d'outils une fois, et il fonctionne avec Claude, GPT, Gemini, les modèles locaux, LangChain, CrewAI, et tout client compatible MCP. C'est le moment TCP/IP de l'outillage IA.

#### Google ADK - Orchestration Entreprise

```python
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent

researcher = Agent(name="researcher", model="gemini-2.0-flash", tools=[web_search])
writer = Agent(name="writer", model="gemini-2.0-flash", instruction="Write based on research.")
critic = Agent(name="critic", model="gemini-2.0-flash", instruction="Critique the writing.")

# Raffinement iteratif : ecrire, critiquer, reecrire (max 3 iterations)
refiner = LoopAgent(name="refine", sub_agents=[writer, critic], max_iterations=3)

# Pipeline complet : recherche en parallele, puis raffiner
pipeline = SequentialAgent(
    name="content_pipeline",
    sub_agents=[
        ParallelAgent(name="gather", sub_agents=[researcher, fact_checker]),
        refiner,
    ],
)
```

Les composables `SequentialAgent`, `ParallelAgent` et `LoopAgent` d'ADK permettent de construire des workflows multi-agents complexes de manière déclarative. Pas besoin de DSL de graphe - ce sont juste des objets Python.

#### CrewAI - Équipes Basées sur les Rôles

```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Senior Research Analyst",
    goal="Find comprehensive information about the topic",
    backstory="You are an expert researcher with 20 years of experience...",
    tools=[web_search, arxiv_search],
)

writer = Agent(
    role="Technical Writer",
    goal="Create clear, engaging technical content",
    backstory="You are a skilled technical writer...",
)

research_task = Task(description="Research {topic}", agent=researcher)
writing_task = Task(description="Write article about {topic}", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[research_task, writing_task])
result = crew.kickoff(inputs={"topic": "HNSW indexing algorithms"})
```

Le pattern "rôle + objectif + backstory" de CrewAI est intuitif pour les non-ingénieurs et fonctionne bien pour les workflows de génération de contenu. Mais il est moins flexible que LangGraph pour les flux de contrôle personnalisés.

---

## 4. La Révolution MCP

Le **Model Context Protocol** (MCP), introduit par Anthropic en novembre 2024, est sans doute le développement d'infrastructure le plus important dans l'espace de l'IA agentique. Il fournit un standard universel pour connecter les LLMs à des outils et sources de données externes.

### Pourquoi MCP Est Important

Avant MCP, chaque framework avait son propre format d'intégration d'outils :
- LangChain : décorateur `@tool` avec schéma personnalisé
- OpenAI : schémas de fonctions JSON dans l'API
- Google : `FunctionTool` avec son propre format
- Chaque nouveau framework : encore un autre format

MCP unifie tout cela : **construisez un serveur d'outils une fois, connectez-le à n'importe quel agent**.

### Architecture MCP

```
+-------------------+     Protocole MCP      +--------------------+
|    Agent LLM      | <--------------------> |   Serveur MCP      |
|    (tout modele)  |    (JSON-RPC sur       |   (fournisseur     |
|                   |     stdio/SSE/HTTP)    |    d'outils)       |
+-------------------+                        +--------------------+
```

MCP définit trois primitives :
1. **Outils (Tools)** - fonctions que le modèle peut appeler (recherche, calcul, opérations CRUD)
2. **Ressources (Resources)** - données que le modèle peut lire (fichiers, enregistrements de base de données, réponses API)
3. **Prompts** - modèles de prompts pré-construits pour les patterns courants

### Adoption (fin 2025)

| Entreprise | Support MCP | Statut |
|---------|-------------|--------|
| Anthropic | Natif (créateur de MCP) | Support complet dans Claude |
| OpenAI | Agents SDK + Responses API | Support complet depuis mars 2025 |
| Google | ADK MCPToolset | Support complet |
| Microsoft | Plugin MCP Semantic Kernel | Supporté |
| Cursor | Client MCP intégré | Production |
| Windsurf | Client MCP intégré | Production |
| Zed | Client MCP intégré | Production |

### Écosystème de Serveurs MCP

L'écosystème de serveurs MCP a explosé :
- **Plus de 50 serveurs officiels** (système de fichiers, GitHub, Slack, PostgreSQL, Qdrant, etc.)
- **Serveurs communautaires** pour chaque API majeure (Stripe, Twilio, AWS, GCP)
- **Serveurs entreprise** pour les outils internes (CRM, ERP, systèmes de ticketing)

### Références
- Anthropic (2024). *"Introducing the Model Context Protocol"*. anthropic.com/news
- Spécification MCP. modelcontextprotocol.io
- OpenAI (2025). *"MCP support in the Agents SDK"*. openai.com/blog

---

## 5. Les Patterns de Production Qui Fonctionnent Vraiment

Après avoir étudié des dizaines de déploiements d'agents en production, voici les patterns qui séparent les systèmes fonctionnels des démos de conférence.

### Pattern 1 : Boucles d'Agent Contraintes (Pas Ouvertes)

**Problème** : Les agents non contraints spiralent - ils appellent les outils de manière répétée, restent bloqués dans des boucles, ou dévient de leur objectif.

**Solution** : Définir des conditions de terminaison explicites :

```python
agent = Agent(
    instructions="""You have a maximum of 3 tool calls per query.
    After retrieving information, synthesize and respond immediately.
    Do NOT search for the same thing twice.""",
    max_tool_calls=3,  # Limite dure
)
```

Les agents de production sont **étroits et contraints**, pas généralistes. Chaque agent fait une seule chose bien.

### Pattern 2 : Cache Sémantique

**Problème** : Les requêtes similaires passent par le pipeline LLM complet à chaque fois. Coûteux et lent.

**Solution** : Mettre en cache les paires (embedding_requête, réponse). Sur une nouvelle requête, vérifier la similarité cosine avec le cache. Si >0.92, retourner la réponse cachée instantanément.

```python
# Recherche cache : <5ms vs. 5-30s pour le pipeline complet
cached = semantic_cache.lookup(query_embedding)
if cached:
    return cached  # ~0ms temps de generation

# Pipeline complet uniquement sur cache miss
result = full_rag_pipeline(query)
semantic_cache.store(query, query_embedding, result)
```

Cela réduit les coûts de 60-80% sur les charges de production avec des requêtes répétées/similaires (courant en support client, Q&A de documentation).

### Pattern 3 : Guardrails en Parallèle, Pas Séquentiels

**Problème** : Exécuter les vérifications de sécurité avant l'agent ajoute de la latence.

**Solution** (pattern OpenAI) : Exécuter les guardrails **en parallèle** avec l'agent. Si le guardrail se déclenche, annuler la réponse de l'agent.

```python
# Le guardrail s'execute simultanement avec l'agent principal
# Latence totale = max(temps_agent, temps_guardrail), PAS la somme
agent = Agent(
    input_guardrails=[PromptInjectionDetector()],  # S'execute en parallele
    output_guardrails=[FactChecker()],              # Verifie la sortie
)
```

### Pattern 4 : Routage Adaptatif des Requêtes

**Problème** : Toutes les requêtes n'ont pas besoin du pipeline d'agent complet. Les questions simples gaspillent du calcul.

**Solution** : Un classificateur léger route les requêtes vers le pipeline optimal :

```
SIMPLE  (salutation, personnel)       --> Réponse LLM directe (~100ms)
MEDIUM  (factuel, recherche)          --> Pipeline RAG standard (~2-5s)
COMPLEX (comparaison, multi-étapes)   --> RAG étendu + plus de contexte (~5-15s)
```

Cela réduit la latence moyenne de 40% et les coûts de 50% sur les charges mixtes.

### Pattern 5 : Humain dans la Boucle avec Approbation Asynchrone

**Problème** : Les agents ont besoin de l'approbation humaine pour les actions destructrices, mais les attentes bloquantes ruinent l'UX.

**Solution** : Approbation asynchrone via webhooks. L'agent demande l'approbation, continue d'autres tâches, reprend quand c'est approuvé.

---

## 6. Benchmarks : Performance des Agents (Novembre 2025)

### SWE-bench (Ingénierie Logicielle)

| Système d'Agent | Score | Modèle | Notes |
|-------------|-------|-------|-------|
| Claude Sonnet 3.5 + Computer Use | 49.0% | Claude 3.5 | Anthropic, Oct 2024 |
| OpenAI o3 + Codex Agent | 54.6% | o3 | OpenAI, 2025 |
| Devin (Cognition) | 53.1% | Custom | Agent de codage spécialisé |
| SWE-Agent + GPT-4o | 33.2% | GPT-4o | Baseline open-source |

### GAIA (Assistants IA Généraux)

| Système d'Agent | Niveau 1 | Niveau 2 | Niveau 3 |
|-------------|---------|---------|---------|
| GPT-4o + outils | 56% | 38% | 12% |
| Claude 3.5 + outils | 61% | 42% | 15% |
| Gemini 2.0 + ADK | 58% | 40% | 13% |
| Baseline humain | 92% | 86% | 74% |

### Point Clé
Les agents atteignent des **performances de niveau humain sur des tâches étroites et bien définies** (correction de code, recherche d'information) mais restent bien en dessous des humains sur le raisonnement ouvert et multi-étapes. L'écart se réduit chaque trimestre.

---

## 7. La Frontière de la Recherche : Ce Qui Arrive en 2026

### 7.1 Apprentissage par Renforcement pour les Agents

La tendance de recherche la plus importante : entraîner les agents de bout en bout avec du RL plutôt que de l'ingénierie de prompts. DeepSeek-R1 et OpenAI o3 ont montré que le RL sur les tâches de raisonnement produit un comportement qualitativement différent. La prochaine étape est le RL sur les tâches agentiques - entraîner les modèles à utiliser les outils efficacement, pas seulement à raisonner sur du texte.

### 7.2 Modèles du Monde pour la Planification

Les agents actuels planifient en "pensant étape par étape" dans du texte. La frontière, ce sont les **modèles du monde appris** - des simulations internes de l'environnement qui permettent aux agents de prédire les effets de leurs actions avant de les exécuter. Cela permettrait :
- La planification multi-étapes sans essai-erreur
- L'évaluation des risques avant les actions destructrices
- Le raisonnement hypothétique ("que se passe-t-il si j'appelle cette API avec ces paramètres ?")

### 7.3 Protocoles Agent-à-Agent

Le protocole A2A (Agent-to-Agent) de Google et le MCP d'Anthropic convergent vers un futur où les agents de différentes organisations communiquent directement :

```
Agent d'achat Entreprise A  --A2A-->  Agent de vente Entreprise B
    +-- negocier les termes, verifier l'inventaire, passer commande --+
```

C'est l'"économie des API" mais avec de la négociation en langage naturel au lieu de schémas REST fixes.

### 7.4 Agents Auto-Améliorants

La direction la plus spéculative mais potentiellement transformatrice : des agents qui améliorent leurs propres prompts, outils et stratégies basés sur le feedback des résultats. Les premiers résultats de Reflexion (Shinn et al., 2023) et Self-Refine (Madaan et al., 2023) montrent une amélioration de 10-30% sur les tâches de codage et de raisonnement grâce aux boucles d'auto-critique.

---

## 8. Recommandations Stratégiques

### Pour les Startups
- **Commencez avec OpenAI Agents SDK** pour le time-to-market le plus rapide
- **Construisez des serveurs MCP** pour vos outils - protégez-vous contre les changements de fournisseur de modèle
- **Utilisez le cache sémantique** dès le premier jour - c'est de la performance gratuite

### Pour les Entreprises
- **Google ADK + Vertex AI** pour le déploiement managé et la conformité
- **LangGraph** pour une flexibilité maximale sur les workflows complexes
- **Investissez dans les guardrails et l'évaluation** - la partie difficile est la fiabilité, pas la capacité

### Pour les Chercheurs / Projets Portfolio
- **Construisez un système d'agent full-stack** qui démontre :
  1. Architecture multi-modèles (modèles locaux + API)
  2. RAG avec cache sémantique et routage adaptatif
  3. Télémétrie et observabilité en temps réel
  4. Intégration d'outils compatible MCP
- **C'est exactement ce que MIRROR démontre** - un système de grade production tournant entièrement sur CPU sans dépendances cloud

---

## 9. Références

1. Yao et al. (2023). *"ReAct: Synergizing Reasoning and Acting in Language Models"*. ICLR 2023
2. Shinn et al. (2023). *"Reflexion: Language Agents with Verbal Reinforcement Learning"*. NeurIPS 2023
3. Madaan et al. (2023). *"Self-Refine: Iterative Refinement with Self-Feedback"*. NeurIPS 2023
4. Wei et al. (2022). *"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"*. NeurIPS 2022
5. Anthropic (2024). *"Introducing the Model Context Protocol"*. anthropic.com/news
6. Anthropic (2025). *"Building Effective Agents"*. anthropic.com/research
7. OpenAI (2025). *"Introducing the Agents SDK"*. openai.com/blog
8. Google (2025). *"Agent Development Kit: Build AI agents with Gemini"*. google.github.io/adk-docs
9. Google (2025). *"Agent-to-Agent (A2A) Protocol"*. github.com/google/A2A
10. Harrison Chase (2024). *"LangGraph: Multi-Actor Applications with LLMs"*. LangChain Blog
11. CrewAI Documentation. crewai.com/docs
12. Microsoft (2025). *"AutoGen: Enabling Next-Gen LLM Applications"*. microsoft.github.io/autogen
13. Microsoft (2025). *"Semantic Kernel"*. learn.microsoft.com/semantic-kernel
14. HuggingFace (2025). *"Smolagents: Lightweight agents for everyone"*. huggingface.co/docs/smolagents
15. Spécification MCP (2025). modelcontextprotocol.io/specification
16. Jimenez et al. (2024). *"SWE-bench: Can Language Models Resolve Real-World GitHub Issues?"*. ICLR 2024
17. Mialon et al. (2023). *"GAIA: A Benchmark for General AI Assistants"*. arXiv:2311.12983
18. Lewis et al. (2020). *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"*. NeurIPS 2020
19. Abdin et al. (2024). *"Phi-4 Technical Report"*. Microsoft Research. arXiv:2412.08905
20. Chen et al. (2024). *"BGE M3-Embedding"*. arXiv:2402.03216

---

*Michail Berjaoui - Novembre 2025*
*MIRROR - Portfolio alimenté par l'IA démontrant des patterns agentiques de grade production sur infrastructure CPU uniquement*
