#!/usr/bin/env python3
"""Tidus v1.2.0 -- Auto-Classifier POC Validation Script

Validates the heuristic + embedding classification algorithm on 500+ labeled
messages BEFORE integrating into Tidus. Runs standalone -- no Tidus imports.

Usage:
    uv run python scripts/poc_classifier.py
    uv run python scripts/poc_classifier.py --no-embedding   # Tier 1 only
    uv run python scripts/poc_classifier.py --verbose         # Print failures

Production validation sources:
    - LiteLLM Complexity Router (code/reasoning/technical-terms/token weights)
    - vLLM Semantic Router (3-tier cascade pattern, March 2026)
    - RouteLLM (Stanford 2024) -- 85%% cost reduction with heuristic routing
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Enums  (mirrors tidus/models/task.py -- no Tidus import needed in POC)
# ---------------------------------------------------------------------------

class Complexity(str, Enum):
    simple   = "simple"
    moderate = "moderate"
    complex  = "complex"
    critical = "critical"

class Domain(str, Enum):
    chat           = "chat"
    code           = "code"
    reasoning      = "reasoning"
    extraction     = "extraction"
    classification = "classification"
    summarization  = "summarization"
    creative       = "creative"

class Privacy(str, Enum):
    public       = "public"
    internal     = "internal"
    confidential = "confidential"


# ---------------------------------------------------------------------------
# Result / test types
# ---------------------------------------------------------------------------

@dataclass
class ClassResult:
    domain:      Domain
    complexity:  Complexity
    privacy:     Privacy
    tokens:      int
    domain_conf: float = 0.0
    cmplx_conf:  float = 0.0
    tier:        str   = "heuristic"   # heuristic | needs_embedding | embedding | needs_llm


class Case(NamedTuple):
    text:             str
    expected_domain:  Domain
    expected_cmplx:   Complexity
    expected_privacy: Privacy = Privacy.internal


# ---------------------------------------------------------------------------
# Tier 1 -- Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4.5))


# ---------------------------------------------------------------------------
# Tier 1 -- Privacy detection
# ---------------------------------------------------------------------------

_PII: dict[str, re.Pattern] = {
    "ssn":     re.compile(r'\b(?!000|666|9\d{2})\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'),
    "cc":      re.compile(r'\b(?:4\d{12,15}|5[1-5]\d{14}|3[47]\d{13}|6011\d{12})\b'),
    "email":   re.compile(r'\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b'),
    "phone":   re.compile(r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    "aws":     re.compile(r'\b(AKIA|AGPA|AIDA|AROA)[A-Z0-9]{16}\b'),
    "ghtoken": re.compile(r'\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b'),
    "secret":  re.compile(r'(?:api[_\-]?key|password|secret|token)\s*[:=]\s*\S{8,}', re.I),
}
_MEDICAL   = re.compile(r'\b(patient|diagnosis|prescription|PHI|HIPAA|medical record|clinical|dosage|treatment plan)\b', re.I)
_FINANCIAL = re.compile(r'\b(account number|routing number|wire transfer|bank statement|tax return|W-2|1099)\b', re.I)
_LEGAL     = re.compile(r'\b(attorney-client|privileged|under seal|confidential settlement|NDA|trade secret)\b', re.I)


def _luhn(digits: str) -> bool:
    """Luhn checksum -- eliminates ~60%% of false-positive credit card matches."""
    clean = re.sub(r'\D', '', digits)
    total = 0
    for i, ch in enumerate(reversed(clean)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def detect_privacy(text: str) -> tuple[Privacy, float]:
    for name, pat in _PII.items():
        m = pat.search(text)
        if not m:
            continue
        if name == "cc" and not _luhn(m.group()):
            continue
        ctx = text[max(0, m.start() - 200):m.end() + 200].lower()
        has_ctx = any(kw in ctx for kw in [
            "my ssn", "card number", "social security", "billing",
            "patient id", "account", "credential", "key is", "token is",
        ])
        return Privacy.confidential, (0.85 if has_ctx else 0.75)

    if _MEDICAL.search(text) or _FINANCIAL.search(text) or _LEGAL.search(text):
        return Privacy.internal, 0.65

    tokens = estimate_tokens(text)
    pronoun_hits = len(re.findall(r'\b(I |my |our )\b', text, re.I))
    if tokens > 0 and pronoun_hits / tokens > 0.08:
        return Privacy.internal, 0.55

    return Privacy.internal, 0.50


# ---------------------------------------------------------------------------
# Tier 1 -- Complexity  (LiteLLM-aligned weighted scoring)
# ---------------------------------------------------------------------------

_CRITICAL_KW = frozenset({
    "diagnose", "diagnosis", "medical advice", "prescribe",
    "legal advice", "legal liability", "compliance", "regulatory",
    "financial risk", "investment advice", "credit risk",
    "life-critical", "safety-critical", "life or death",
})

# Reasoning markers -- Li & Roth (2002) WH-word taxonomy, 97.2%% TREC accuracy
_WH_HYPO = re.compile(
    r'\b(what\s+if|what\s+would\s+happen|suppose\s+that|hypothetically|'
    r'what\s+are\s+the\s+(implications|consequences|trade.offs))', re.I
)
_WH_EXPLAIN = re.compile(
    r'\bexplain\b', re.I  # anywhere in text (catches "Can you explain..." mid-sentence)
)
_WH_COMPLEX = re.compile(
    r'(?:^|\n)\s*(how\b|why\b|analyze\b|analyse\b|compare\b|'  # "explain" removed — has its own signal
    r'evaluate\b|discuss\b|assess\b|critique\b|contrast\b|'
    r'what\s+(would|could|should|if)|in\s+what\s+way\b)',
    re.I
)
_WH_SIMPLE = re.compile(
    r'(?:^|\n)\s*(what\s+(is|are|does|was|were)\b|when\b|where\b|who\b|'
    r'which\b|how\s+(many|much)\b|list\b|name\s+the\b|define\b)',
    re.I
)
_MODAL = re.compile(
    r'\b(should|would|could|might|must|ought\s+to|need\s+to)\b', re.I
)

# Imperative command patterns -- design vs moderate vs simple imperative
_IMP_DESIGN = re.compile(
    r'(?:^|\n)\s*(design\s+(a|an)\b|architect\s+|'
    # build a/an + 1-5 words (incl. hyphenated like "Raft-based", "multi-region") + arch noun
    r'build\s+(a|an)\s+(?:\w[\w-]*\s+){1,5}(system|service|architecture|framework|engine|'
    r'platform|pipeline|infrastructure|microservice|handler|coordinator|scheduler|'
    r'middleware|processor|manager|driver|store|cache|proxy|agent|module|library|api)|'
    # implement a/an + 1-5 words + arch noun
    r'implement\s+(a|an)\s+(?:\w[\w-]*\s+){1,5}(system|service|framework|protocol|algorithm|'
    r'handler|coordinator|processor|cache|store|driver|module))',
    re.I
)
_IMP_MODERATE = re.compile(  # moderate imperative tasks → +0.25
    r'(?:^|\n)\s*(implement\b|'
    r'set\s+up\b|'
    r'configure\b|'
    r'handle\b|'
    r'refactor\b|'
    r'debug\s+(?:this|the|my|a)\b|'
    r'add\s+(?!a\b|an\b|the\b)\w|'    # "add Prometheus", "add type hints" etc.
    # "explain" removed — handled by _WH_EXPLAIN (+0.20) to avoid double-counting with WH_COMPLEX
    r'classify\s+these\b|categorize\s+these\b|'  # multi-item classification tasks
    r'tag\s+each\b|label\s+each\b|assign\s+each\b|'
    r'parse\s+)',   # structured-data parsing (logs, JSON, CSV, XML) — almost always moderate+
    re.I
)
_IMP_WRITE = re.compile(  # simple-to-moderate imperative tasks → +0.15
    r'(?:^|\n)\s*(write\s+(a|an|me)\b|'
    r'write\s+\w+\s+\w+\s+(?:for|to|about)\b|'  # "write marketing copy for", "write product docs to"
    r'create\s+(a|an)\b|'
    r'fix\s+(this|the|my)\b|'
    r'draft\s+(a|an)\b|'
    r'develop\s+(a|an)\b|'
    r'summarize\s+|summarise\s+|'  # summarization tasks (length determines simple vs moderate)
    r'condense\s+|distill\s+)',
    re.I
)

# Scale / multi-entity / batch-operation signal
_SCALE_COMPLEX = re.compile(
    r'(\d+[KMB]\s+(?:users?|requests?|events?|messages?|devices?|records?|'
    r'readers?|customers?|subscribers?|visitors?|investors?|downloads?)|'
    r'millions?\s+of\s+(?:users?|requests?)|'
    r'\bthese\s+\d+\b|'              # "these 200 tickets", "these 50 resumes"
    r'\bthis\s+\d+[-\s]?page|'       # "this 50-page contract"
    r'\b\d+[-\s]page\b|'             # "50-page document"
    r'\bthese\s+(?:two|three|four|five|six|seven|several|multiple|few|many)\b|'  # spelled-out counts
    r'\bfrom\s+this\s+(?:html|xml|json|csv|yaml|log|schema|config(?:uration)?|'
    r'report|contract|article|document|database|sitemap|transcript|nginx|audit|'
    r'kubernetes|erp|transaction|access\s+log|server\s+log)\b|'  # "from this HTML page"
    r'\bfrom\s+[1-9][\d,]+\s|'       # "from 200 academic papers", "from 10,000 contracts"
    r'\bconsidering\b|'
    r'vs\.?\s+\w+|versus\s+\w+|'
    r'across\s+(?:multiple|several|many)\b)',
    re.I
)

# Advanced / expert-level concepts -- strong indicator of complex tier
_ADVANCED = re.compile(
    r'\b(lock.free|wait.free|zero.copy|zero.downtime|'
    r'exactly.once|at.least.once|split.brain|'
    r'sliding\s+window|token\s+bucket|leaky\s+bucket|'
    r'operational\s+transform|crdt|'
    r'homomorphic|zero.knowledge\s+proof|'
    r'raft\b|paxos\b|pbft\b|zab\b|'
    r'backpressure|bulkhead\s+pattern|circuit\s+breaker|'  # resilience patterns
    r'consistent\s+hashing|rendezvous\s+hashing|'
    r'saga\s+pattern|outbox\s+pattern|'
    r'multi.region|active.active|blue.green|'
    r'event\s+sourc|cqrs\b|'                  # event-driven architecture
    r'linearizab|serializab|'                   # distributed consistency models
    r'thread.safe|lock.free|'                   # concurrency guarantees
    r'LRU\b|TTL\b|'                             # cache eviction strategies
    r'memory\s+management|ownership\s+model|'  # language runtime theory
    r'migrat\w+\s+from\b)',                     # architectural migration (always complex)
    re.I
)

# Word-count specification — "500-word X" implies non-trivial writing task
_WORD_COUNT = re.compile(r'\b\d{3,4}[-\s]words?\b', re.I)

# Quality/scope adjectives — distinguish moderate writing tasks from simple ones (+0.08)
# Fires when the task has explicit scope, audience, or format complexity markers.
_QUALITY_ADJ = re.compile(
    r'\b(comprehensive|in.depth|end.to.end|step.by.step|'
    r'detailed\b|'                     # "a detailed README"
    r'narrative\b|'                    # "a narrative postmortem"
    r'content\s+calendar|'             # business content artifact
    r'marketing\s+copy|drip\s+campaign|'  # multi-piece content production
    r'user\s+persona|buyer\s+persona|'    # UX/marketing artifacts
    r'onboarding\s+(?:guide|doc)|'     # structured technical documentation
    r'incident\s+postmortem|'          # formal engineering artifacts
    r'press\s+release|pitch\s+deck|product\s+announcement)\b',  # formal business formats
    re.I
)

# Secondary requirement — "summarize X AND/,+participle highlight Y" → multi-step moderate task (+0.08)
# Note: no \b after group — catches participles ("highlighting", "identifying", etc.)
_SECONDARY_REQ = re.compile(
    r'(?:\band\s+|,\s*)(highlight|identif|includ|tag|extract|label|categori|'
    r'flag|mark\b|list\b|note\b|call\s+out|point\s+out)',
    re.I
)

# Code structural signals -- GitHub Linguist priority order
_CODE_PAT = [
    (re.compile(r'```[\w]*\n'), 0.30),
    (re.compile(r'(?m)^(def |class |import |from \w+ import )'), 0.25),
    (re.compile(r'\b(function\s*\(|=>\s*\{|async\s+function)'), 0.20),
    (re.compile(r'[{};]\s*\n'), 0.15),
]
_OP_RE = re.compile(r'[{}()\[\];:=<>!+\-*/|&^~@]')

# Technical term density -- broad set covering code + infra + compliance
_TECH = re.compile(
    r'\b(API|SDK|OAuth|JWT|regex|algorithm|schema|endpoint|webhook|'
    r'database|query|index|cache|latency|throughput|async|concurrent|'
    r'recursion|complexity|tensor|gradient|embedding|'
    r'HIPAA|GDPR|SOC2|PCI|SLA|uptime|microservice|distributed|'
    r'kubernetes|docker|CI[/\-]CD|pipeline|refactor|architecture|'
    r'Redis|Kafka|Postgres|MySQL|MongoDB|Elasticsearch|'
    r'atomic|transaction|deadlock|race\s+condition|locking|'
    r'sharding|replication|consensus|partitioning|fault\s+tolerance|'
    r'circuit\s+breaker|backpressure|rate\s+limit|load\s+balanc|'
    r'trade.offs?|scalability|availability|consistency|durability|'
    r'concurren|parallelism|'
    r'DOM|virtual\s+DOM|state\s+management|component|hydration|'
    r'OAuth2|RBAC|zero.trust|mTLS|certificate|encryption)\b', re.I
)

_COND = re.compile(
    r'\b(unless|provided that|assuming|given that|'
    r'in case|as long as|only if|even if)\b', re.I
)
_NEG = re.compile(
    r'\b(not|never|neither|nor|cannot|without|except|unless)\b', re.I
)


def detect_complexity(text: str, tokens: int) -> tuple[Complexity, float]:
    low = text.lower()

    # Critical keyword veto
    for kw in _CRITICAL_KW:
        if kw in low:
            return Complexity.critical, 1.0

    score = 0.0

    # Code presence (weight 0.30) -- computed first as it gates other signals
    code_s = sum(w for pat, w in _CODE_PAT if pat.search(text))
    if len(_OP_RE.findall(text)) / max(len(text), 1) > 0.08:
        code_s += 0.15
    code_s = min(code_s, 0.30)
    score += code_s

    # Reasoning / WH-word markers -- suppressed when code block is the main signal
    # Also scale boost by text length: short "Why is X Y?" questions score 0.10, not 0.30
    if code_s < 0.25:
        if _WH_HYPO.search(text):
            score += 0.40
        elif _WH_EXPLAIN.search(text):
            score += 0.20  # "explain X" anywhere → moderate signal (no token scaling)
        elif _WH_COMPLEX.search(text) or len(_MODAL.findall(text)) >= 2:
            score += 0.30 if tokens >= 15 else 0.10  # short definitional → less boost
        elif _WH_SIMPLE.search(text):
            score -= 0.08

    # Imperative command complexity -- 3 tiers
    # IMP_MODERATE/WRITE skipped when code block present (code_s handles it)
    if _IMP_DESIGN.search(text):
        score += 0.40   # raised 0.35→0.40: design+tech(0.15)=0.55 → complex ✓
    elif code_s < 0.25:
        if _IMP_MODERATE.search(text):
            score += 0.25
        elif _IMP_WRITE.search(text):
            score += 0.15  # reduced from 0.20 to keep "Write a simple function" below moderate threshold

    # Technical term density -- lower cap when code block already provides signal
    tech = len(_TECH.findall(text))
    tech_cap = 0.10 if code_s >= 0.25 else 0.15
    score += min((tech / max(tokens, 1)) * 6, tech_cap)

    # Scale / batch-operation / multi-entity comparison signal (+0.20)
    if _SCALE_COMPLEX.search(text):
        score += 0.20

    # Advanced / expert-level concept signal (+0.15)
    if _ADVANCED.search(text):
        score += 0.15

    # Word-count specification (e.g. "500-word blog post") → non-trivial writing (+0.15)
    if _WORD_COUNT.search(text):
        score += 0.15

    # Quality/scope adjectives: "detailed", "comprehensive", "narrative", "content calendar" etc.
    # Lifts moderate writing tasks that are otherwise under-scored by IMP_WRITE (+0.15 - tok_penalty)
    if _QUALITY_ADJ.search(text):
        score += 0.08

    # Secondary requirement: "summarize X AND highlight Y" → two-step task (+0.08)
    if _SECONDARY_REQ.search(text):
        score += 0.08

    # Conditional / negation density -- "if" removed (too common in natural language)
    cond_n = len(_COND.findall(text)) + len(_NEG.findall(text))
    cond_d = cond_n / max(tokens, 1)
    score += 0.15 if cond_d > 0.10 else (0.08 if cond_d > 0.05 else 0)

    # Token count
    if tokens > 200:   score += 0.10  # noqa: E701
    elif tokens > 80:  score += 0.06  # noqa: E701
    elif tokens < 20:  score -= 0.05  # noqa: E701

    # Moderate threshold = 0.15 (lowered from 0.18; IMP_WRITE reduced to 0.15 to compensate)
    # Complex threshold = 0.55 (raised from 0.50; avoids SCALE+WH_COMPLEX combos hitting complex)
    if score >= 0.55:   return Complexity.complex,  min(score, 0.95)   # noqa: E701
    elif score >= 0.15: return Complexity.moderate, 0.50 + score * 0.28  # noqa: E701
    else:               return Complexity.simple,   max(0.60 - score, 0.50)  # noqa: E701


# ---------------------------------------------------------------------------
# Tier 1 -- Domain detection (structural + keyword)
# ---------------------------------------------------------------------------

_DOM_KW: dict[Domain, set[str]] = {
    Domain.code: {
        "def ", "class ", "import ", "function ", "algorithm",
        "debug", "compile", "syntax error", "refactor", "```",
        "variable", "loop", "array", "stack overflow", "git commit",
        "type error", "runtime error", "null pointer", "unit test",
        "mock ", "fixture", "pytest", "jest ", "webpack", "linter",
        # Language names with context (avoids "trust"←"rust", "pythonic", etc.)
        "python", "javascript ", "typescript ", " golang", " rust ",
        " java", "c++ ", "c# ", " ruby ", " swift ", " kotlin ",
        # Language-in-context patterns (catch "in Python?", "Python's", "Rust, Go")
        "in python", "in javascript", "in typescript", "in golang",
        "in rust", "in go ", "in kotlin", "in swift", "in java",
        # Frameworks / runtimes
        "fastapi", "django", "flask", "asyncio", "sqlalchemy",
        "node.js", "react ", "angular ", "express ", "kubernetes ",
        "docker ", "bash ", "shell script",
        # Code-task verbs / patterns not elsewhere covered
        "implement ", "recursion", "data structure", "binary tree",
        "linked list", "hash map", "queue ", "deadlock", "concurren",
        "memory leak", "thread-safe", "thread safe",
        "git ", "pip ", "pip install", "using pip", "npm ", "dockerfile",
        # Architecture / system design terms (fix Design X → chat misses)
        "graphql", "grpc", " kafka", "cqrs", "event sourcing",
        "sharding", "circuit breaker", "jwt", "consistent hashing",
        "zero-downtime", "multi-region", "active-active",
        "b-tree", "b+tree", "raft ", "paxos", "consensus algorithm",
    },
    Domain.reasoning: {
        "explain why", "prove ", "analyze", "logic",
        "hypothesis", "infer", "argument", "root cause",
        "therefore", "implication", "evidence", "reasoning",
        "cause and effect", "trade-off", "trade off", "because",
        "justify", "rationale", "underlying", "mechanism", "why does",
        "why is ", "why are ", "why do ", "why would ", "why should ",
        # Comparison / contrast patterns
        "difference between", "compare ", "comparison", " versus ",
        " vs ", "vs.", "contrast ", "pros and cons",
        # Removed "how do " / "how does" -- these steal code-domain "How do I X in Python?" cases
    },
    Domain.extraction: {
        "extract", "parse ", "find all", "list all", "pull out",
        "from the text", "identify all", "csv", "json", " xml ",
        "regex pattern", "field", "column", "row ", "table",
        "scrape", "mine ", "gather from", "collect from", "get all",
        "retrieve all", "enumerate all", "output all", "pick out",
    },
    Domain.classification: {
        "classify", "categorize", "label ", "tag ", "what type",
        "which category", "sort into", "group by",
        "assign ", "segment", "bucket", "tier ", "categorise",
        "taxonomy", "class label", "belongs to", "is it a", "is this ",
        "what kind of", "what category",
    },
    Domain.summarization: {
        "summarize", "summarise", "tldr", "tl;dr", "recap",
        "brief overview", "key points", "in short",
        "highlight", "condense", "main idea", "overview of",
        "abstract", "synopsis", "summary of", "shorten", "gist of",
        "sum up", "boil down", "key takeaways", "main takeaways",
        "distill ",  # "Distill the key insights from..." cases
    },
    Domain.creative: {
        "write a story", "write a poem", "creative writing",
        "imagine ", "invent ", "compose ", "brainstorm ideas",
        "generate ideas", "fictional", "narrative", "write me a",
        "screenplay", "blog post", "marketing copy", "tagline",
        "metaphor", "analogy", "come up with", "brand name",
        "slogan", "pitch ", "creative ", "story about",
        # Shorter poetry / creative formats
        "poem", "haiku", "limerick", "sonnet", "verse ",
        "motivational", "birthday ", "out-of-office",
        # Content creation formats
        "brainstorm ", "write a ", "create a ",
        "onboarding", "readme", "faq ", "persona ",
        "newsletter", "press release", "drip campaign",
        "content calendar", "feature article",
        "linkedin", "cover letter", "pitch deck",
        "product description", "announcement email",
    },
    Domain.chat: set(),   # catch-all
}

_CODE_STRUCT = [
    (re.compile(r'(?m)^#!(/usr/bin|/bin)'), 0.95),
    (re.compile(r'```[\w]*\n[\s\S]+?```'), 0.90),
    (re.compile(r'(?m)^(def |class |import |from \w+ import )'), 0.85),
    (re.compile(r'[{};]\s*\n'), 0.80),
    (re.compile(r'\b(function\s*\(|=>\s*\{|async\s+function)'), 0.80),
    (re.compile(r'\b[a-z]+_[a-z]+_[a-z]+\b'), 0.65),
]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def detect_domain(text: str) -> tuple[Domain, float]:
    low = text.lower()

    # Level 1 -- structural code signals
    code_struct = max(
        (w for pat, w in _CODE_STRUCT if pat.search(text)),
        default=0.0,
    )
    if len(_OP_RE.findall(text)) / max(len(text), 1) > 0.08:
        code_struct = max(code_struct, 0.75)
    if code_struct >= 0.65:
        return Domain.code, code_struct

    # Level 2 -- keyword hit-count sigmoid
    # Key fix: sigmoid on RAW HIT COUNT (not coverage ratio)
    #   0 hits -> 0.12,  1 hit -> 0.50,  2 hits -> 0.88,  3 hits -> 0.98
    best_dom  = Domain.chat
    best_conf = 0.12    # chat prior

    for dom, kws in _DOM_KW.items():
        if not kws:
            continue
        hits = sum(1 for kw in kws if kw in low)
        if hits == 0:
            continue
        conf = _sigmoid(2.0 * (hits - 1.0))
        if conf > best_conf:
            best_conf = conf
            best_dom  = dom

    if code_struct > best_conf:
        return Domain.code, code_struct

    return best_dom, best_conf


# ---------------------------------------------------------------------------
# Full Tier 1 classify
# ---------------------------------------------------------------------------

HEURISTIC_THRESHOLD = 0.55   # 2+ keyword hits (conf ~0.88) accepted; 1 hit -> Tier 2


def classify_t1(text: str) -> ClassResult:
    tokens         = estimate_tokens(text)
    privacy, _     = detect_privacy(text)
    complexity, cc = detect_complexity(text, tokens)
    domain, dc     = detect_domain(text)
    tier = "heuristic" if dc >= HEURISTIC_THRESHOLD else "needs_embedding"
    return ClassResult(
        domain=domain, complexity=complexity, privacy=privacy,
        tokens=tokens, domain_conf=dc, cmplx_conf=cc, tier=tier,
    )


# ---------------------------------------------------------------------------
# Tier 2 -- embedding cosine similarity
# ---------------------------------------------------------------------------

_DOM_DESC = {
    Domain.code:           "Writing, reading, debugging, reviewing, explaining, or refactoring code or programming logic",
    Domain.reasoning:      "Logical analysis, mathematical proof, causal explanation, hypothesis testing, or step-by-step problem solving",
    Domain.extraction:     "Pulling out specific structured data, fields, lists, or entities from a block of text",
    Domain.classification: "Categorising, labelling, tagging, or sorting content into predefined groups or types",
    Domain.summarization:  "Condensing a long document, conversation, or article into a shorter summary or key points",
    Domain.creative:       "Generating original creative content such as stories, poems, scripts, ideas, or marketing copy",
    Domain.chat:           "Conversational questions, general knowledge lookup, casual assistance, or simple factual queries",
}

_embed_model   = None
_label_vecs: dict[Domain, object] | None = None


def load_embeddings() -> bool:
    global _embed_model, _label_vecs
    try:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        vecs = _embed_model.encode(
            list(_DOM_DESC.values()), normalize_embeddings=True
        )
        _label_vecs = {d: vecs[i] for i, d in enumerate(_DOM_DESC)}
        return True
    except ImportError:
        return False


def classify_t2(text: str, t1: ClassResult) -> ClassResult:
    if _embed_model is None or _label_vecs is None:
        return t1
    try:
        import numpy as np
        vec  = _embed_model.encode([text[:400]], normalize_embeddings=True)[0]
        sims = {d: float(np.dot(vec, lv)) for d, lv in _label_vecs.items()}
        best = max(sims, key=sims.get)
        conf = sims[best]
        tier = "embedding" if conf >= 0.45 else "needs_llm"
        return ClassResult(
            domain=best, complexity=t1.complexity, privacy=t1.privacy,
            tokens=t1.tokens, domain_conf=conf, cmplx_conf=t1.cmplx_conf,
            tier=tier,
        )
    except Exception:
        return t1


# ---------------------------------------------------------------------------
# Test dataset (500+ labeled cases)
# ---------------------------------------------------------------------------

def _build_cases() -> list[Case]:
    C = Case
    D = Domain
    X = Complexity
    P = Privacy

    # ── code / simple ────────────────────────────────────────────────────────
    code_simple = [
        C("Write a Python function to reverse a string.", D.code, X.simple),
        C("Write a JavaScript function to sort an array of numbers.", D.code, X.simple),
        C("Write a Go function to check if a number is prime.", D.code, X.simple),
        C("Write a TypeScript function to flatten a nested array.", D.code, X.simple),
        C("Write a Rust function to find the maximum in a vector.", D.code, X.simple),
        C("Write a Python function to count word frequencies in a string.", D.code, X.simple),
        C("Write a function that converts Celsius to Fahrenheit.", D.code, X.simple),
        C("Create a hello world program in Python.", D.code, X.simple),
        C("How do I read a file line by line in Python?", D.code, X.simple),
        C("How do I split a string on commas in JavaScript?", D.code, X.simple),
        C("What does `list.append()` do in Python?", D.code, X.simple),
        C("What does `Array.map()` return in JavaScript?", D.code, X.simple),
        C("What is the difference between `let` and `const` in JavaScript?", D.code, X.simple),
        C("What is a list comprehension in Python?", D.code, X.simple),
        C("How do I print to stderr in Python?", D.code, X.simple),
        C("Fix this Python syntax error: `def foo(x) return x`", D.code, X.simple),
        C("How do I import a module in Python?", D.code, X.simple),
        C("How do I declare a typed variable in TypeScript?", D.code, X.simple),
        C("How do I iterate over a dictionary in Python?", D.code, X.simple),
        C("How do I catch exceptions in Python with try/except?", D.code, X.simple),
        C("What is a lambda function in Python?", D.code, X.simple),
        C("How do I concatenate strings in Python?", D.code, X.simple),
        C("Write a Python function to check if a string is a palindrome.", D.code, X.simple),
        C("Write a function to calculate factorial of n recursively.", D.code, X.simple),
        C("How do I install a package using pip?", D.code, X.simple),
        C("What does `git commit -m` do?", D.code, X.simple),
        C("How do I run unit tests with pytest?", D.code, X.simple),
        C("How do I create a virtual environment in Python?", D.code, X.simple),
        C("What is a null pointer exception?", D.code, X.simple),
        C("Write a Python function to remove duplicates from a list.", D.code, X.simple),
    ]

    # ── code / moderate ───────────────────────────────────────────────────────
    code_moderate = [
        C("Refactor this Python class to use async/await and add error handling:\n```python\nclass Fetcher:\n    def fetch(self, url):\n        import requests\n        return requests.get(url).json()\n```", D.code, X.moderate),
        C("Debug this React component that causes infinite re-renders:\n```jsx\nuseEffect(() => { setCount(count + 1); });\n```", D.code, X.moderate),
        C("Why is my recursive Fibonacci implementation hitting maximum recursion depth on large inputs?", D.code, X.moderate),
        C("Explain the difference between shallow copy and deep copy in Python with examples.", D.code, X.moderate),
        C("Implement a binary search tree in TypeScript with insert and search methods.", D.code, X.moderate),
        C("How do I optimize this SQL query that runs slowly on 10M rows?\n```sql\nSELECT * FROM orders WHERE status='pending';\n```", D.code, X.moderate),
        C("Explain how Python generators work and when to use them over lists.", D.code, X.moderate),
        C("Implement rate limiting in a FastAPI endpoint.", D.code, X.moderate),
        C("What is the time complexity of this nested-loop algorithm and how can I improve it?", D.code, X.moderate),
        C("Implement a debounce function in JavaScript that limits call frequency.", D.code, X.moderate),
        C("How do I mock an external API call in pytest without calling it?", D.code, X.moderate),
        C("Implement the Observer design pattern in Python.", D.code, X.moderate),
        C("Explain how Python's GIL affects multithreaded programs.", D.code, X.moderate),
        C("Set up CI/CD with GitHub Actions for a Python project.", D.code, X.moderate),
        C("Explain the difference between `__str__` and `__repr__` in Python.", D.code, X.moderate),
        C("Handle database transactions in SQLAlchemy to avoid deadlocks.", D.code, X.moderate),
        C("Implement JWT authentication in a FastAPI application.", D.code, X.moderate),
        C("Explain the JavaScript event loop and how promises work.", D.code, X.moderate),
        C("Add Prometheus metrics instrumentation to a FastAPI application.", D.code, X.moderate),
        C("Implement a retry mechanism with exponential backoff in Python.", D.code, X.moderate),
        C("Refactor this function to remove code duplication.", D.code, X.moderate),
        C("Add type hints to an existing Python codebase.", D.code, X.moderate),
        C("Configure Docker multi-stage builds to reduce image size.", D.code, X.moderate),
        C("Implement a thread-safe counter in Python using a lock.", D.code, X.moderate),
        C("Explain the difference between abstract classes and interfaces in TypeScript.", D.code, X.moderate),
        C("Debug this memory leak in a Node.js application that grows over time.", D.code, X.moderate),
        C("Implement pagination in a REST API endpoint in Python.", D.code, X.moderate),
        C("Explain how CORS works and configure it in FastAPI.", D.code, X.moderate),
        C("Write a custom Kubernetes liveness probe for a FastAPI service.", D.code, X.moderate),
        C("Implement a simple WebSocket server in Python using asyncio.", D.code, X.moderate),
    ]

    # ── code / complex ────────────────────────────────────────────────────────
    code_complex = [
        C("Design a distributed rate limiter in Python using Redis that handles multiple instances, clock skew, and supports sliding window and token bucket algorithms.", D.code, X.complex),
        C("Compare trade-offs between REST, GraphQL, and gRPC for a microservices architecture serving 10M daily requests, considering latency, schema evolution, and caching.", D.code, X.complex),
        C("Implement a lock-free concurrent queue in Rust supporting multiple producers and consumers without mutex overhead.", D.code, X.complex),
        C("Design a database sharding strategy for a social platform with 500M users, covering hotspot prevention, cross-shard queries, and consistent hashing.", D.code, X.complex),
        C("Design an event sourcing system with CQRS for an e-commerce platform supporting complex refunds, partial fulfillments, and full audit trails.", D.code, X.complex),
        C("Analyze the security implications of JWT token design -- compare HS256 vs RS256 vs ES256 for a multi-tenant SaaS platform.", D.code, X.complex),
        C("Design a streaming pipeline using Kafka that processes 1M events/second with exactly-once semantics and backpressure handling.", D.code, X.complex),
        C("Implement a thread-safe LRU cache with TTL expiration in Python using only standard library primitives.", D.code, X.complex),
        C("Explain how B-tree indexing works internally in PostgreSQL and when a composite index outperforms multiple single-column indexes.", D.code, X.complex),
        C("Design a circuit breaker in Go with closed/open/half-open state transitions and metrics instrumentation.", D.code, X.complex),
        C("Implement a custom consensus algorithm for distributed state management that handles network partitions and split-brain scenarios.", D.code, X.complex),
        C("Compare memory management models of Rust, Go, and C++, and explain when GC overhead becomes a bottleneck.", D.code, X.complex),
        C("Design a zero-downtime migration strategy for a 1TB Postgres database with 99.99%% SLA requirements.", D.code, X.complex),
        C("Implement a real-time collaborative editor using operational transforms -- explain conflict resolution.", D.code, X.complex),
        C("Design a multi-region active-active payments API requiring strong consistency without distributed transactions.", D.code, X.complex),
    ]

    # ── reasoning / simple ────────────────────────────────────────────────────
    reasoning_simple = [
        C("What is the difference between HTTP and HTTPS?", D.reasoning, X.simple),
        C("Why is Python considered an interpreted language?", D.reasoning, X.simple),
        C("What is a REST API?", D.reasoning, X.simple),
        C("Why do we use version control systems?", D.reasoning, X.simple),
        C("What is the purpose of a load balancer?", D.reasoning, X.simple),
        C("Why is SQL called a declarative language?", D.reasoning, X.simple),
        C("What is the difference between RAM and ROM?", D.reasoning, X.simple),
        C("Why is JSON preferred over XML in modern web APIs?", D.reasoning, X.simple),
        C("What is the difference between a process and a thread?", D.reasoning, X.simple),
        C("Why do databases use indexing?", D.reasoning, X.simple),
        C("What is the CAP theorem?", D.reasoning, X.simple),
        C("Why is idempotency important in API design?", D.reasoning, X.simple),
        C("What is the difference between authorization and authentication?", D.reasoning, X.simple),
        C("Why is HTTPS used instead of HTTP for sensitive traffic?", D.reasoning, X.simple),
        C("What is the difference between stack and heap memory?", D.reasoning, X.simple),
    ]

    # ── reasoning / moderate ──────────────────────────────────────────────────
    reasoning_moderate = [
        C("Explain why eventual consistency is sometimes acceptable in distributed systems and when it is not.", D.reasoning, X.moderate),
        C("Analyze the trade-offs between monorepo vs. multiple repositories for a team of 50 engineers.", D.reasoning, X.moderate),
        C("Why does React use a virtual DOM instead of directly manipulating the real DOM?", D.reasoning, X.moderate),
        C("Explain why microservices introduce more failure modes than a monolith.", D.reasoning, X.moderate),
        C("Why is it harder to test asynchronous code than synchronous code?", D.reasoning, X.moderate),
        C("Compare the pros and cons of ORM vs. raw SQL queries in production.", D.reasoning, X.moderate),
        C("Explain why functional programming advocates for immutability and pure functions.", D.reasoning, X.moderate),
        C("Why do distributed systems struggle with distributed transactions, and what alternatives exist?", D.reasoning, X.moderate),
        C("Analyze the trade-offs between horizontal and vertical scaling for a database-heavy application.", D.reasoning, X.moderate),
        C("Why should you avoid premature optimization in software development?", D.reasoning, X.moderate),
        C("Explain why hash tables have O(1) average lookup but can degrade to O(n) worst-case.", D.reasoning, X.moderate),
        C("Why is database normalization important, and when should you intentionally denormalize?", D.reasoning, X.moderate),
        C("Compare the trade-offs between NoSQL and SQL databases for different workloads.", D.reasoning, X.moderate),
        C("Explain why consensus is hard to achieve in distributed systems with unreliable networks.", D.reasoning, X.moderate),
        C("Why is cache invalidation considered one of the hardest problems in computer science?", D.reasoning, X.moderate),
        C("Analyze the trade-offs between synchronous and asynchronous inter-service communication.", D.reasoning, X.moderate),
        C("Explain why the actor model simplifies concurrent programming compared to shared-memory threading.", D.reasoning, X.moderate),
        C("Why are blue-green deployments safer than in-place deployments for critical services?", D.reasoning, X.moderate),
        C("Explain why strong typing in TypeScript catches bugs that runtime JavaScript would miss.", D.reasoning, X.moderate),
        C("Analyze why Kubernetes adds complexity despite Docker being straightforward.", D.reasoning, X.moderate),
        C("Why is it problematic to store passwords in plain text and what should you use instead?", D.reasoning, X.moderate),
        C("Explain why SOLID principles help maintain large codebases.", D.reasoning, X.moderate),
        C("Why do some teams prefer event-driven architecture despite its added complexity?", D.reasoning, X.moderate),
        C("Explain why recursive algorithms can be less efficient than iterative ones in practice.", D.reasoning, X.moderate),
        C("Analyze the trade-offs between strong and weak consistency models in distributed caching.", D.reasoning, X.moderate),
    ]

    # ── reasoning / complex ───────────────────────────────────────────────────
    reasoning_complex = [
        C("Compare trade-offs between eventual consistency and strong consistency, considering CAP theorem, PACELC model, and implications for financial transactions.", D.reasoning, X.complex),
        C("Evaluate the architectural implications of migrating from a Django monolith to microservices for a 5-year-old platform with 50M users.", D.reasoning, X.complex),
        C("Analyze implications of CQRS with Event Sourcing vs. traditional CRUD for a healthcare system needing both high-throughput writes and complex historical queries.", D.reasoning, X.complex),
        C("Compare distributed consensus algorithms (Raft vs. Paxos vs. PBFT) for a permissioned blockchain handling financial contracts.", D.reasoning, X.complex),
        C("Evaluate zero-trust security architecture vs. perimeter-based security for a multi-cloud enterprise with 10,000 employees in 30 countries.", D.reasoning, X.complex),
        C("Analyze distributed database consistency models (linearizability, sequential, causal) for a globally distributed social network.", D.reasoning, X.complex),
        C("Compare trade-offs between serverless and container-based deployment for batch processing with variable load and strict cost controls.", D.reasoning, X.complex),
        C("Evaluate homomorphic encryption vs. secure enclaves for processing sensitive financial data in a cloud environment.", D.reasoning, X.complex),
        C("Compare serialization formats (Protobuf, Avro, MessagePack, JSON) for a high-throughput event streaming system at 1M events/second.", D.reasoning, X.complex),
        C("Compare optimistic and pessimistic locking trade-offs in a high-concurrency financial system with high row-level contention.", D.reasoning, X.complex),
    ]

    # ── extraction / simple ───────────────────────────────────────────────────
    extraction_simple = [
        C("List all email addresses in this text: 'Contact us at support@co.com or billing@co.com'", D.extraction, X.simple),
        C("Extract all numbers from: 'Order #12345 for $299.99 placed on 2024-01-15'", D.extraction, X.simple),
        C("Find all URLs in this paragraph.", D.extraction, X.simple),
        C("Parse this JSON and list all keys: {\"name\": \"Alice\", \"age\": 30}", D.extraction, X.simple),
        C("What are all uppercase words in: 'The API KEY and SECRET TOKEN must be PRIVATE'?", D.extraction, X.simple),
        C("Extract all dates from: 'Meeting Jan 15, review Feb 3, deadline March 30'", D.extraction, X.simple),
        C("List all programming languages mentioned in this paragraph.", D.extraction, X.simple),
        C("Find all phone numbers in: 'Call (555) 123-4567 or 1-800-555-0199'", D.extraction, X.simple),
        C("Extract all hashtags from this tweet.", D.extraction, X.simple),
        C("List all country names mentioned in this article.", D.extraction, X.simple),
        C("Find all IP addresses in this log file excerpt.", D.extraction, X.simple),
        C("Extract all currency amounts from this invoice.", D.extraction, X.simple),
        C("List all bullet points from this document.", D.extraction, X.simple),
        C("Extract the first and last name from each line of this CSV.", D.extraction, X.simple),
        C("Get all table rows that contain the word 'error'.", D.extraction, X.simple),
    ]

    # ── extraction / moderate ──────────────────────────────────────────────────
    extraction_moderate = [
        C("Extract all email addresses and phone numbers from this database export and organize into CSV with headers.", D.extraction, X.moderate),
        C("Parse this JSON API response and identify all fields where the value is null or empty array.", D.extraction, X.moderate),
        C("From this HTML page, extract all product names, prices, and availability status.", D.extraction, X.moderate),
        C("From this 50-page contract, extract all dates, party names, and monetary amounts into a structured table.", D.extraction, X.moderate),
        C("Analyze this server access log and extract all IPs that made more than 100 requests in the past hour.", D.extraction, X.moderate),
        C("From this XML configuration file, extract all environment variables and their values.", D.extraction, X.moderate),
        C("Parse these 200 support tickets and extract category, priority, and resolution time for each.", D.extraction, X.moderate),
        C("Extract all named entities (persons, organizations, locations) from this news article.", D.extraction, X.moderate),
        C("From this SQL schema file, extract all table names, column names, and foreign key relationships.", D.extraction, X.moderate),
        C("From these 50 resumes, extract name, email, years of experience, and top skills for each.", D.extraction, X.moderate),
        C("Parse this nginx access log and extract all 4xx/5xx errors with paths and timestamps.", D.extraction, X.moderate),
        C("From this transaction JSON, extract all transactions above $10,000 grouped by merchant.", D.extraction, X.moderate),
        C("From this XML sitemap, extract all URLs and their last-modified dates.", D.extraction, X.moderate),
        C("Parse this CSV of customer data and extract all records where email domain is a competitor.", D.extraction, X.moderate),
        C("From this Kubernetes YAML, extract all container images and their version tags.", D.extraction, X.moderate),
    ]

    # ── extraction / complex ───────────────────────────────────────────────────
    extraction_complex = [
        C("Build a parser that extracts structured invoice data (vendor, line items, totals, tax, payment terms) from unstructured free-text invoices in multiple formats.", D.extraction, X.complex),
        C("Extract and reconcile transaction records from three different bank CSV exports with inconsistent column names, date formats, and currency representations.", D.extraction, X.complex),
        C("Given 10,000 legal contracts in varied formats, extract all termination clauses, governing law provisions, and liability caps.", D.extraction, X.complex),
        C("Parse a complex nested JSON from a legacy ERP system and extract all product hierarchies with attributes into a flat relational schema.", D.extraction, X.complex),
        C("Extract all code snippets, their programming languages, and surrounding context from 500 StackOverflow HTML pages.", D.extraction, X.complex),
        C("Build a system to extract and validate all PII fields (SSN, credit card, passport) from unstructured customer records.", D.extraction, X.complex),
        C("Extract and deduplicate all citations from 200 academic papers in multiple citation formats (APA, MLA, Chicago).", D.extraction, X.complex),
        C("Parse this 500-page regulatory document and extract all compliance requirements, effective dates, and affected entities.", D.extraction, X.complex),
        C("From 10,000 product descriptions, extract standardized attributes (dimensions, weight, material, color) even when described narratively.", D.extraction, X.complex),
        C("Extract relationship graphs (who-reports-to-whom) from 1,000 organizational announcement emails with no consistent format.", D.extraction, X.complex),
    ]

    # ── classification / simple ───────────────────────────────────────────────
    classification_simple = [
        C("Is this email spam or not spam?", D.classification, X.simple),
        C("Classify this support ticket as bug, feature request, or question.", D.classification, X.simple),
        C("Is this review positive, negative, or neutral?", D.classification, X.simple),
        C("Categorize this expense as travel, food, or equipment.", D.classification, X.simple),
        C("Tag this article as technology, politics, sports, or entertainment.", D.classification, X.simple),
        C("Classify this transaction as fraud or legitimate.", D.classification, X.simple),
        C("Is this customer feedback a complaint or a compliment?", D.classification, X.simple),
        C("Categorize this bug report as critical, high, medium, or low priority.", D.classification, X.simple),
        C("Is this code comment a TODO, FIXME, or HACK?", D.classification, X.simple),
        C("What type of document is this: invoice, contract, or receipt?", D.classification, X.simple),
        C("Is this social media post about sports, politics, or entertainment?", D.classification, X.simple),
        C("What kind of error is this: network, database, or application?", D.classification, X.simple),
    ]

    # ── classification / moderate ──────────────────────────────────────────────
    classification_moderate = [
        C("Categorize these 200 customer support tickets into 5-7 meaningful groups based on common themes.", D.classification, X.moderate),
        C("Classify each of these 50 news articles into the appropriate industry sector for an investment database.", D.classification, X.moderate),
        C("Given these 100 code snippets, label each as clean, needs refactoring, or technical debt.", D.classification, X.moderate),
        C("Classify these customer segments into personas based on their purchase history and behavior data.", D.classification, X.moderate),
        C("Tag each of these 300 job postings with required skills, seniority level, and team type.", D.classification, X.moderate),
        C("Categorize these API endpoints by CRUD type, resource, and required authentication level.", D.classification, X.moderate),
        C("Given these transaction records, classify each as legitimate, suspicious, or fraudulent using the provided fraud criteria.", D.classification, X.moderate),
        C("Categorize these 500 product descriptions by department, subcategory, and target demographic.", D.classification, X.moderate),
        C("Assign severity to these 100 security vulnerabilities using CVSS scoring criteria.", D.classification, X.moderate),
        C("Classify these user feedback items as UX issue, performance problem, feature request, or bug.", D.classification, X.moderate),
        C("Group these 1,000 support emails by root cause and assign to appropriate teams.", D.classification, X.moderate),
        C("Classify these legal documents as contracts, NDAs, employment agreements, or IP filings.", D.classification, X.moderate),
    ]

    # ── summarization / simple ────────────────────────────────────────────────
    summarization_simple = [
        C("Summarize this paragraph in one sentence.", D.summarization, X.simple),
        C("Give me a TLDR of this article.", D.summarization, X.simple),
        C("What are the key points of this email?", D.summarization, X.simple),
        C("Summarize this meeting transcript briefly.", D.summarization, X.simple),
        C("What is this document about in 2-3 sentences?", D.summarization, X.simple),
        C("Give me a quick overview of this report.", D.summarization, X.simple),
        C("Summarize the main argument of this essay.", D.summarization, X.simple),
        C("Condense this 500-word blog post into 3 bullet points.", D.summarization, X.simple),
        C("Summarize this changelog for me.", D.summarization, X.simple),
        C("Give me the gist of this research abstract.", D.summarization, X.simple),
        C("TL;DR this long article for me.", D.summarization, X.simple),
        C("Sum up this email thread in two sentences.", D.summarization, X.simple),
    ]

    # ── summarization / moderate ──────────────────────────────────────────────
    summarization_moderate = [
        C("Summarize this 5-page technical specification into key points, preserving all architectural decisions.", D.summarization, X.moderate),
        C("Create an executive summary of this quarterly business review for a non-technical audience.", D.summarization, X.moderate),
        C("Condense this 100-page legal contract into the 10 most important clauses with plain-English explanations.", D.summarization, X.moderate),
        C("Summarize the key differences between these three competing infrastructure migration proposals.", D.summarization, X.moderate),
        C("Create a structured summary of this research paper's methodology, findings, and limitations.", D.summarization, X.moderate),
        C("Summarize this week's Slack activity and highlight decisions made and open action items.", D.summarization, X.moderate),
        C("Distill the key insights from these 20 customer interview transcripts into recurring themes.", D.summarization, X.moderate),
        C("Summarize this security audit report, highlighting critical and high-severity findings.", D.summarization, X.moderate),
        C("Create a timeline summary of this incident postmortem report.", D.summarization, X.moderate),
        C("Summarize this thread of 50 emails about a project dispute, identifying each party's core position.", D.summarization, X.moderate),
        C("Condense this architecture document into a one-page overview for executive stakeholders.", D.summarization, X.moderate),
        C("Summarize these 5 competing research papers on transformer improvements, highlighting consensus and disagreements.", D.summarization, X.moderate),
        C("Create a summary of quarterly sales data with trend analysis from this spreadsheet.", D.summarization, X.moderate),
        C("Summarize the key takeaways from this all-hands meeting recording transcript.", D.summarization, X.moderate),
        C("Distill the 10 most actionable insights from this 200-page industry research report.", D.summarization, X.moderate),
    ]

    # ── creative / simple ──────────────────────────────────────────────────────
    creative_simple = [
        C("Write a short poem about autumn leaves.", D.creative, X.simple),
        C("Write a haiku about programming.", D.creative, X.simple),
        C("Give me 5 funny brand names for a coffee startup.", D.creative, X.simple),
        C("Write me a one-paragraph story about a robot learning to cook.", D.creative, X.simple),
        C("Create a catchy slogan for a gym.", D.creative, X.simple),
        C("Write a brief birthday message for a colleague.", D.creative, X.simple),
        C("Give me 3 creative names for an AI tools startup.", D.creative, X.simple),
        C("Write a fun tagline for a pizza restaurant.", D.creative, X.simple),
        C("Create a 50-word product description for noise-canceling headphones.", D.creative, X.simple),
        C("Write a motivational quote about learning from failure.", D.creative, X.simple),
        C("Brainstorm 5 ideas for a team-building activity.", D.creative, X.simple),
        C("Write a short limerick about missing deadlines.", D.creative, X.simple),
        C("Come up with a creative name for a developer podcast.", D.creative, X.simple),
        C("Write a funny out-of-office email reply for a vacation.", D.creative, X.simple),
        C("Brainstorm 5 taglines for a productivity app.", D.creative, X.simple),
    ]

    # ── creative / moderate ────────────────────────────────────────────────────
    creative_moderate = [
        C("Write a 500-word blog post about the future of AI in healthcare for a general audience.", D.creative, X.moderate),
        C("Create a compelling LinkedIn profile summary for a senior engineer transitioning to product management.", D.creative, X.moderate),
        C("Write a product announcement email for a new developer tool that competes with Postman.", D.creative, X.moderate),
        C("Create a persuasive 10-slide pitch deck outline for a Series A fundraise in enterprise AI.", D.creative, X.moderate),
        C("Write a comprehensive onboarding guide for new engineers joining a fast-growing startup.", D.creative, X.moderate),
        C("Create a 3-email drip campaign for converting free-tier users to paid customers.", D.creative, X.moderate),
        C("Write a detailed README for an open-source Python data validation library.", D.creative, X.moderate),
        C("Write a technical blog post explaining gradient descent for a non-technical marketing audience.", D.creative, X.moderate),
        C("Create a comprehensive FAQ document for a B2B SaaS pricing page.", D.creative, X.moderate),
        C("Develop a user persona profile for a mid-market enterprise customer of a project management tool.", D.creative, X.moderate),
        C("Write a 1000-word feature article on the impact of remote work on engineering culture.", D.creative, X.moderate),
        C("Write a narrative incident postmortem suitable for a public engineering blog.", D.creative, X.moderate),
        C("Create a social media content calendar for a developer tools company for one month.", D.creative, X.moderate),
        C("Write marketing copy for 3 pricing tiers of a developer productivity SaaS.", D.creative, X.moderate),
        C("Write a fictional short story set in 2050 where AI assistants are mandatory in government offices.", D.creative, X.moderate),
    ]

    # ── chat / simple ─────────────────────────────────────────────────────────
    chat_simple = [
        C("What is the capital of France?", D.chat, X.simple),
        C("When was Python first released?", D.chat, X.simple),
        C("Who invented the internet?", D.chat, X.simple),
        C("What time zone is Tokyo in?", D.chat, X.simple),
        C("What does GPT stand for?", D.chat, X.simple),
        C("How many bytes are in a kilobyte?", D.chat, X.simple),
        C("What is the most spoken language in the world?", D.chat, X.simple),
        C("What year was Docker released?", D.chat, X.simple),
        C("What does API stand for?", D.chat, X.simple),
        C("What is the boiling point of water in Fahrenheit?", D.chat, X.simple),
        C("How many planets are in our solar system?", D.chat, X.simple),
        C("What does HTML stand for?", D.chat, X.simple),
        C("Who founded Microsoft?", D.chat, X.simple),
        C("What year was the first iPhone released?", D.chat, X.simple),
        C("What is the speed of light in km/s?", D.chat, X.simple),
    ]

    # ── chat / moderate ────────────────────────────────────────────────────────
    chat_moderate = [
        C("Help me understand the difference between machine learning and deep learning.", D.chat, X.moderate),
        C("Can you explain how blockchain works in simple terms?", D.chat, X.moderate),
        C("I'm learning to code -- should I start with Python or JavaScript?", D.chat, X.moderate),
        C("What are some good resources for learning system design?", D.chat, X.moderate),
        C("Can you explain quantum computing like I'm 10 years old?", D.chat, X.moderate),
        C("What's the difference between Agile and Scrum?", D.chat, X.moderate),
        C("Help me understand what a CDN does and why websites use them.", D.chat, X.moderate),
        C("What's the best way to prepare for a software engineering interview?", D.chat, X.moderate),
        C("Can you explain the difference between supervised and unsupervised learning?", D.chat, X.moderate),
        C("Help me understand SOLID principles in software engineering.", D.chat, X.moderate),
        C("What are the main differences between SQL and NoSQL databases?", D.chat, X.moderate),
        C("Can you explain DevOps and why companies are adopting it?", D.chat, X.moderate),
        C("Help me understand how OAuth 2.0 works for app authentication.", D.chat, X.moderate),
        C("What's the difference between Docker containers and virtual machines?", D.chat, X.moderate),
        C("Can you explain event-driven architecture in plain English?", D.chat, X.moderate),
    ]

    # ── critical ───────────────────────────────────────────────────────────────
    critical = [
        C("What medication dosage should I prescribe for a patient with renal impairment and Type 2 diabetes?", D.reasoning, X.critical),
        C("What is the legal liability if our AI system provides incorrect medical diagnosis advice to users?", D.reasoning, X.critical),
        C("Our system is life-critical -- what redundancy should we implement for 99.999%% uptime?", D.reasoning, X.critical),
        C("What financial risk does our company face if we invest 40%% of assets in cryptocurrency?", D.reasoning, X.critical),
        C("Is it safe to diagnose appendicitis from these symptoms without imaging?", D.reasoning, X.critical),
        C("What are the HIPAA compliance requirements for storing patient medical records in a cloud environment?", D.reasoning, X.critical),
        C("What investment advice would you give a retiree with $2M needing $80K annual living expenses?", D.reasoning, X.critical),
        C("Our safety-critical embedded firmware must detect hardware faults -- what algorithm should we use?", D.code, X.critical),
        C("What are the legal implications of an AI making hiring decisions based on protected characteristics?", D.reasoning, X.critical),
        C("Prescribe the correct dosage of metformin for a 70kg patient with HbA1c of 8.5%%.", D.reasoning, X.critical),
        C("Our life or death patient monitoring system is failing silently -- diagnose the root cause.", D.code, X.critical),
        C("What regulatory filings are required when our financial algorithm caused a flash crash?", D.reasoning, X.critical),
        C("Assess the financial risk of our derivatives portfolio under a 3-sigma tail event scenario.", D.reasoning, X.critical),
        C("What is the FDA 21 CFR Part 11 compliance requirement for clinical trial data retention?", D.reasoning, X.critical),
        C("Our life-critical pacemaker firmware update is silently failing on 2%% of devices.", D.code, X.critical),
    ]

    # ── ambiguous (should escalate to Tier 2, not be wrong-assigned) ──────────
    ambiguous = [
        C("Can you help me with something?",          D.chat, X.simple),
        C("I need to process some data.",             D.chat, X.simple),
        C("How do I do this better?",                 D.chat, X.moderate),
        C("Fix it.",                                  D.chat, X.simple),
        C("What should I do?",                        D.chat, X.simple),
        C("Make it faster.",                          D.chat, X.simple),
        C("It's not working.",                        D.chat, X.simple),
        C("I have a question about the thing.",       D.chat, X.simple),
        C("Help.",                                    D.chat, X.simple),
        C("Can you look at this for me?",             D.chat, X.simple),
        C("I need to understand this.",               D.chat, X.simple),
        C("What does this mean?",                     D.chat, X.simple),
        C("I need help with my project.",             D.chat, X.simple),
        C("This is wrong, please fix it.",            D.chat, X.simple),
        C("Tell me more about this.",                 D.chat, X.simple),
        C("Explain this to me.",                      D.chat, X.simple),
        C("I'm not sure what to do here.",            D.chat, X.simple),
        C("What's the best way?",                     D.chat, X.simple),
        C("Which one should I pick?",                 D.chat, X.simple),
        C("Is this right?",                           D.chat, X.simple),
        C("Can you check this?",                      D.chat, X.simple),
        C("I want to improve this.",                  D.chat, X.simple),
        C("Something seems off.",                     D.chat, X.simple),
        C("What am I missing?",                       D.chat, X.simple),
        C("How should I approach this?",              D.chat, X.moderate),
    ]

    # ── privacy-specific cases ────────────────────────────────────────────────
    privacy_cases = [
        C("My SSN is 123-45-6789 and I need help with my tax return.",      D.chat, X.simple, P.confidential),
        C("Contact us at user@example.com for general support.",             D.chat, X.simple, P.internal),
        C("The patient HIPAA record shows diagnosis of type 2 diabetes.",    D.chat, X.simple, P.internal),
        C("api_key=sk_live_abc123456789 -- is this format valid?",           D.chat, X.simple, P.confidential),
        C("This is a public question about today's weather.",                D.chat, X.simple, P.internal),
        C("My credit card 4532015112830366 expires 12/25, help with bill.", D.chat, X.simple, P.confidential),
        C("The NDA prohibits disclosure of our trade secret pricing model.", D.chat, X.simple, P.internal),
        C("GitHub token ghp_xK2mN5pQ8rL3vW7yT1uJ9eA4dC6bF0hG2sZ is leaked.", D.chat, X.simple, P.confidential),
        C("AWS access key AKIAIOSFODNN7EXAMPLE was committed to git history.", D.chat, X.simple, P.confidential),
        C("The attorney-client privileged memo must not be disclosed.",      D.chat, X.simple, P.internal),
    ]

    return (
        code_simple + code_moderate + code_complex
        + reasoning_simple + reasoning_moderate + reasoning_complex
        + extraction_simple + extraction_moderate + extraction_complex
        + classification_simple + classification_moderate
        + summarization_simple + summarization_moderate
        + creative_simple + creative_moderate
        + chat_simple + chat_moderate
        + critical + ambiguous + privacy_cases
        + _build_extra_cases()
    )


def _build_extra_cases() -> list[Case]:
    """~1,625 extra labeled cases for the 2,000-case validation suite."""
    C = Case; D = Domain; X = Complexity; P = Privacy  # noqa: E702

    # code / simple: 2+ code keywords, score < 0.15
    # "Write a {lang} function to {task}." → IMP_WRITE(0.15) - tok_penalty(0.05) = 0.10
    _py_s = [
        "add two numbers", "find the minimum in a list", "count vowels in a string",
        "flatten a nested list one level deep", "check if a string contains only digits",
        "merge two dictionaries", "remove all None values from a list",
        "capitalize every word in a sentence", "find the second largest number in a list",
        "convert a list of tuples to a dictionary", "check if all elements are unique",
        "rotate a list by N positions", "find the longest common prefix of two strings",
        "compute the cumulative sum of a list", "check if a number is a perfect cube",
        "repeat a string N times", "find the index of the first occurrence of a value",
        "convert an integer to a binary string", "calculate the power of two numbers",
        "check if a list is a subset of another list",
    ]
    _js_s = [
        "calculate the median of an array of numbers", "find the most frequent element in an array",
        "remove trailing whitespace from a string", "convert a number to a Roman numeral",
        "check if two objects have the same keys", "deep-clone a plain object",
        "convert snake_case to camelCase", "find all divisors of a number",
        "flatten a two-level nested array", "check if a string ends with a vowel",
    ]
    _ts_s = [
        "check if a value is a non-empty string", "validate that an object has required fields",
        "pick specific properties from an object", "build a URL query string from a record",
        "omit one key from a typed object", "convert an array to a readonly tuple",
        "merge two Partial objects into one", "filter out undefined values from an array",
        "convert an interface to an array of key-value pairs",
        "partition an array into two groups by a predicate",
    ]
    _go_s = [
        "check if a slice of strings contains a value", "reverse a string rune-by-rune",
        "find the max in a slice of integers", "count occurrences of a character in a string",
        "convert a slice of strings to uppercase", "flatten a 2D slice into a 1D slice",
        "check if a string is a valid IPv4 address", "generate a Fibonacci sequence up to N",
    ]
    _ru_s = [
        "check if a vector of integers is sorted", "sum all elements in a vector of floats",
        "find the first duplicate in a vector", "filter even numbers from a vector",
        "zip two vectors into a vector of tuples", "count the number of words in a string",
        "find the index of a value in a vector", "convert a string to a vector of characters",
    ]
    code_s = (
        [C(f"Write a Python function to {t}.", D.code, X.simple) for t in _py_s]
        + [C(f"Write a JavaScript function to {t}.", D.code, X.simple) for t in _js_s]
        + [C(f"Write a TypeScript function to {t}.", D.code, X.simple) for t in _ts_s]
        + [C(f"Write a Golang function to {t}.", D.code, X.simple) for t in _go_s]
        + [C(f"Write a Rust function to {t}.", D.code, X.simple) for t in _ru_s]
        + [
            # "How do I X in Python?" — python + second code keyword, WH_COMPLEX short → 0.05
            C("How do I use recursion to compute a factorial in Python?", D.code, X.simple),
            C("How do I handle a deadlock in Python multithreading?", D.code, X.simple),
            C("How do I use a hash map for memoization in Python?", D.code, X.simple),
            C("How do I implement a queue using collections in Python?", D.code, X.simple),
            C("How do I build a linked list from scratch in Python?", D.code, X.simple),
            C("How do I use asyncio for concurrent tasks in Python?", D.code, X.simple),
            C("How do I iterate over items in a Python data structure?", D.code, X.simple),
            C("How do I add a linter to my Python project?", D.code, X.simple),
            C("How do I write unit tests for a Python module?", D.code, X.simple),
            C("How do I use git with a Python project?", D.code, X.simple),
            C("How do I import a module from a subpackage in Python?", D.code, X.simple),
            C("How do I install a library using pip in Python?", D.code, X.simple),
            C("How do I add type hints to a Python function definition?", D.code, X.simple),
            C("How do I mock a function in pytest?", D.code, X.simple),
            C("How do I create a Python class with a custom constructor?", D.code, X.simple),
            # "What is X in Python?" — WH_SIMPLE(-0.08) - tok_penalty(0.05) = -0.13 → simple
            C("What is a Python class method?", D.code, X.simple),
            C("What is Python's asyncio module used for?", D.code, X.simple),
            C("What does `import` do in Python?", D.code, X.simple),
            C("What is a Python data structure?", D.code, X.simple),
            C("What is a Python function decorator?", D.code, X.simple),
            C("What does `git commit -m` mean?", D.code, X.simple),
            C("What is a Dockerfile used for?", D.code, X.simple),
            C("What does `pip install -r requirements.txt` do?", D.code, X.simple),
            C("What is the purpose of asyncio.gather in Python?", D.code, X.simple),
            C("What is a Flask route decorator?", D.code, X.simple),
            # Short specific code questions
            C("What is a null pointer exception in Java?", D.code, X.simple),
            C("How do I fix a syntax error in Python?", D.code, X.simple),
            C("What is a runtime error in Python?", D.code, X.simple),
            C("What does `npm install` do?", D.code, X.simple),
            C("What is a Docker container?", D.code, X.simple),
            C("What does async/await do in JavaScript?", D.code, X.simple),
            C("What is a TypeScript interface?", D.code, X.simple),
            C("How do I use grep in a bash script?", D.code, X.simple),
            C("What is a React component?", D.code, X.simple),
            C("What does Node.js use for I/O operations?", D.code, X.simple),
            C("How do I push changes with git?", D.code, X.simple),
            C("What is a Kubernetes pod?", D.code, X.simple),
            C("How do I run a Python script from the terminal?", D.code, X.simple),
            C("What does undefined mean in JavaScript?", D.code, X.simple),
            C("What is a React hook?", D.code, X.simple),
            C("How do I format a Python string with f-strings?", D.code, X.simple),
            C("What does Array.reduce do in JavaScript?", D.code, X.simple),
            C("How do I check the Rust version installed?", D.code, X.simple),
            C("What is the Python `@property` decorator?", D.code, X.simple),
        ]
    )  # 100 cases
    # code / moderate: 2+ code keywords, 0.15 ≤ score < 0.55
    _expl_py = [
        "Python generators work and when to use yield",
        "Python's asyncio event loop schedules coroutines",
        "Python's garbage collector handles reference cycles",
        "Python decorators transform functions at definition time",
        "Python context managers work with __enter__ and __exit__",
        "Python metaclasses are used to customize class creation",
        "Python's descriptor protocol enables property and classmethod",
        "Python slots reduce memory footprint on large object sets",
        "Python's import system resolves packages and submodules",
        "Python's asyncio gather runs coroutines concurrently",
    ]
    _expl_js = [
        "JavaScript's prototype chain enables inheritance",
        "JavaScript's event loop processes the task and microtask queues",
        "JavaScript closures capture outer scope variables",
        "JavaScript hoisting moves function declarations to the top",
        "JavaScript WeakMap keys are garbage-collected automatically",
        "JavaScript generator functions pause and resume execution",
        "JavaScript Promise chaining avoids callback nesting",
        "JavaScript's ES module system resolves cyclic imports",
    ]
    _expl_ts = [
        "TypeScript's type inference narrows union types in conditionals",
        "TypeScript generic constraints restrict accepted type parameters",
        "TypeScript conditional types enable type-level branching",
        "TypeScript discriminated unions simplify exhaustive pattern matching",
    ]
    _expl_other = [
        "Golang goroutines and channels coordinate concurrent work",
        "Rust's ownership and borrowing rules prevent data races",
        "FastAPI dependency injection resolves nested dependencies",
    ]
    # WH_EXPLAIN(+0.20) → score 0.20 → moderate ✓
    _explain_cases = (
        [C(f"Explain how {t}.", D.code, X.moderate) for t in _expl_py]
        + [C(f"Explain how {t}.", D.code, X.moderate) for t in _expl_js]
        + [C(f"Explain how {t}.", D.code, X.moderate) for t in _expl_ts]
        + [C(f"Explain how {t}.", D.code, X.moderate) for t in _expl_other]
    )  # 25 cases
    code_m = _explain_cases + [
        # Implement (IMP_MODERATE +0.25, NOT triggering IMP_DESIGN) — 25 cases
        C("Implement a binary search tree with insert and delete in Python.", D.code, X.moderate),
        C("Implement a LRU cache without using any external library in Python.", D.code, X.moderate),
        C("Implement a retry mechanism with exponential backoff in Python.", D.code, X.moderate),
        C("Implement a rate limiter for a FastAPI endpoint.", D.code, X.moderate),
        C("Implement a priority queue using a heap in Python.", D.code, X.moderate),
        C("Implement a thread-safe counter in Python using asyncio locks.", D.code, X.moderate),
        C("Implement JWT authentication middleware in a FastAPI application.", D.code, X.moderate),
        C("Implement a connection pool for SQLAlchemy in a FastAPI app.", D.code, X.moderate),
        C("Implement a debounce utility in TypeScript.", D.code, X.moderate),
        C("Implement lazy loading for a React component tree.", D.code, X.moderate),
        C("Implement a custom React hook for infinite scroll pagination.", D.code, X.moderate),
        C("Implement a type-safe event emitter in TypeScript.", D.code, X.moderate),
        C("Implement a linked list with O(1) append and prepend in Python.", D.code, X.moderate),
        C("Implement a hash map collision resolver using open addressing in Python.", D.code, X.moderate),
        C("Implement a breadth-first search traversal for a binary tree in Python.", D.code, X.moderate),
        C("Implement OAuth 2.0 authorization code flow in a Node.js application.", D.code, X.moderate),
        C("Implement a middleware chain in an Express application.", D.code, X.moderate),
        C("Implement pagination with cursor-based navigation in a FastAPI endpoint.", D.code, X.moderate),
        C("Implement a queue-based job processor using asyncio in Python.", D.code, X.moderate),
        C("Implement a circuit breaker for an HTTP client in Python.", D.code, X.moderate),
        C("Implement a B-tree node insertion in Python.", D.code, X.moderate),
        C("Implement a Raft leader election stub in Golang.", D.code, X.moderate),
        C("Implement a WebSocket broadcast server using asyncio in Python.", D.code, X.moderate),
        C("Implement a recursive descent parser for simple math expressions in Python.", D.code, X.moderate),
        C("Implement a thread-safe bounded channel in Rust.", D.code, X.moderate),
        # Refactor (IMP_MODERATE +0.25) — 15 cases
        C("Refactor this Python class to use async/await throughout.", D.code, X.moderate),
        C("Refactor this Python module to remove circular import dependencies.", D.code, X.moderate),
        C("Refactor this JavaScript callback chain to use async/await.", D.code, X.moderate),
        C("Refactor this React class component to use function components and hooks.", D.code, X.moderate),
        C("Refactor this Python function to be testable with pytest fixtures.", D.code, X.moderate),
        C("Refactor this SQLAlchemy query to avoid N+1 issues.", D.code, X.moderate),
        C("Refactor this TypeScript module to use generics instead of `any`.", D.code, X.moderate),
        C("Refactor this Python script to split it into a proper package structure.", D.code, X.moderate),
        C("Refactor this Django view to use class-based views.", D.code, X.moderate),
        C("Refactor this Node.js express route handler to separate concerns.", D.code, X.moderate),
        C("Refactor this Python data structure traversal to use iterators.", D.code, X.moderate),
        C("Refactor this Flask application to use blueprints and factories.", D.code, X.moderate),
        C("Refactor this Python function to eliminate code duplication using recursion.", D.code, X.moderate),
        C("Refactor this JavaScript array manipulation to use functional methods.", D.code, X.moderate),
        C("Refactor this Golang function to return idiomatic errors.", D.code, X.moderate),
        # Debug (IMP_MODERATE +0.25, "debug this/the/my") — 15 cases
        C("Debug this Python asyncio function that raises RuntimeError on shutdown.", D.code, X.moderate),
        C("Debug this Python class that causes a memory leak on repeated instantiation.", D.code, X.moderate),
        C("Debug this JavaScript React component that triggers infinite re-renders.", D.code, X.moderate),
        C("Debug this TypeScript function that narrows the type incorrectly.", D.code, X.moderate),
        C("Debug this Python SQLAlchemy query that returns duplicate rows.", D.code, X.moderate),
        C("Debug this Python thread-safe counter that has a race condition.", D.code, X.moderate),
        C("Debug this Django ORM query that ignores the database index.", D.code, X.moderate),
        C("Debug this Node.js express middleware that swallows errors silently.", D.code, X.moderate),
        C("Debug this Python recursive function that hits maximum recursion depth.", D.code, X.moderate),
        C("Debug this Golang goroutine that causes a deadlock.", D.code, X.moderate),
        C("Debug this React hook that reads stale state inside an event handler.", D.code, X.moderate),
        C("Debug this Python asyncio task that never completes due to a missing await.", D.code, X.moderate),
        C("Debug this Rust function that triggers a borrow checker error.", D.code, X.moderate),
        C("Debug this Python import that raises a circular dependency error.", D.code, X.moderate),
        C("Debug this FastAPI endpoint that returns a 422 on valid JSON input.", D.code, X.moderate),
        # Why (WH_COMPLEX at start, ≥15 tokens → +0.30) — 10 cases
        C("Why does Python's Global Interpreter Lock prevent true parallelism in CPU-bound multithreaded programs?", D.code, X.moderate),
        C("Why does JavaScript use prototype-based inheritance instead of classical inheritance in the core language?", D.code, X.moderate),
        C("Why do Go goroutines scale better than OS threads for high-concurrency Golang servers?", D.code, X.moderate),
        C("Why does Rust's borrow checker reject sharing a mutable reference across thread boundaries?", D.code, X.moderate),
        C("Why does React batch state updates in event handlers but not in async callbacks?", D.code, X.moderate),
        C("Why does SQLAlchemy emit a SELECT before every UPDATE when the session identity map is dirty?", D.code, X.moderate),
        C("Why does Python's `dict.update` not preserve insertion order when merging conflicting keys?", D.code, X.moderate),
        C("Why does TypeScript's strict null checking prevent accessing properties on possibly undefined values?", D.code, X.moderate),
        C("Why does Kubernetes reschedule pods after a node failure even when the pod is in Running state?", D.code, X.moderate),
        C("Why does async/await in JavaScript not actually run code in parallel on a single event loop thread?", D.code, X.moderate),
        # Add / Configure / Set up (IMP_MODERATE) — 10 cases
        C("Add Prometheus metrics instrumentation to a FastAPI Python application.", D.code, X.moderate),
        C("Add type annotations to an existing Python codebase incrementally.", D.code, X.moderate),
        C("Configure Docker multi-stage builds to reduce a Python image size.", D.code, X.moderate),
        C("Set up GitHub Actions CI/CD for a Python project with pytest.", D.code, X.moderate),
        C("Add error boundary handling to a React component tree.", D.code, X.moderate),
        C("Configure ESLint and Prettier for a TypeScript project.", D.code, X.moderate),
        C("Set up SQLAlchemy Alembic migrations for a FastAPI application.", D.code, X.moderate),
        C("Add structured JSON logging to a Python FastAPI service.", D.code, X.moderate),
        C("Configure Kubernetes liveness and readiness probes for a FastAPI service.", D.code, X.moderate),
        C("Add OpenTelemetry tracing to a Python microservice.", D.code, X.moderate),
    ]  # 100 cases
    # code / complex: 2+ code keywords, score ≥ 0.55
    # IMP_DESIGN(0.35) + SCALE_COMPLEX(0.20) = 0.55 ✓
    # IMP_DESIGN(0.35) + ADVANCED(0.15) + TECH(0.05+) ≈ 0.55+ ✓
    # WH_COMPLEX(0.30) + SCALE_COMPLEX(0.20) + ADVANCED(0.15) = 0.65 ✓
    code_c = [
        # ─── Design a … considering/for scale/across multiple ───────────────
        C("Design a caching layer using consistent hashing and circuit breaker patterns for 100M requests per day.", D.code, X.complex),
        C("Design a Kafka-based streaming pipeline with CQRS and exactly-once semantics for 1M events per second.", D.code, X.complex),
        C("Design a database sharding strategy using consistent hashing for a social platform with 500M users.", D.code, X.complex),
        C("Design a distributed rate limiter using Redis consistent hashing across multiple regions.", D.code, X.complex),
        C("Design a Kafka consumer group coordinator with circuit breaker and backpressure handling for 1M messages per second.", D.code, X.complex),
        C("Design a multi-region active-active deployment with consistent hashing for a FastAPI service at 10K requests per second.", D.code, X.complex),
        C("Design a zero-downtime migration strategy for a sharded Postgres database with CQRS-based audit trail.", D.code, X.complex),
        C("Design a Raft-based leader election service for a distributed key-value store handling 500K requests per second.", D.code, X.complex),
        C("Design a service mesh using gRPC and circuit breaker for 1M inter-service calls per day.", D.code, X.complex),
        C("Design a B-tree based storage engine considering concurrent reads, sharding boundaries, and 1B records.", D.code, X.complex),
        C("Design a distributed job scheduler using Paxos consensus across multiple data centers.", D.code, X.complex),
        C("Design an event sourcing system with CQRS projection rebuilds considering 10K events per second.", D.code, X.complex),
        C("Design a GraphQL federation gateway considering circuit breaker, JWT validation, and 5M daily requests.", D.code, X.complex),
        C("Design a thread-safe connection pool for SQLAlchemy considering deadlock avoidance and 100K concurrent requests.", D.code, X.complex),
        C("Design a gRPC streaming API for real-time data using consistent hashing for 1M connected clients.", D.code, X.complex),
        C("Design a Kafka topic partitioning strategy considering CQRS event replay and 500K events per second.", D.code, X.complex),
        C("Design a Python microservices architecture using Docker and Kubernetes for 10M requests per day.", D.code, X.complex),
        C("Design a JWT token rotation service considering multi-region active-active deployment and 100K authentications per second.", D.code, X.complex),
        C("Design a distributed consensus algorithm using Raft for partition-tolerant metadata storage at 100K operations per second.", D.code, X.complex),
        C("Design a zero-downtime blue-green deployment pipeline for a Kubernetes-hosted Python service at scale.", D.code, X.complex),
        C("Design a backpressure-aware Kafka consumer pipeline with circuit breaker for 10K messages per second.", D.code, X.complex),
        C("Design a sharded time-series database schema for 100K device metrics per second.", D.code, X.complex),
        C("Design a multi-region GraphQL gateway with circuit breaker failover and 99.99%% SLA.", D.code, X.complex),
        C("Design a consistent hashing ring with virtual nodes for a distributed cache across multiple regions.", D.code, X.complex),
        C("Design a Paxos-based distributed lock manager considering split-brain scenarios and 50K operations per second.", D.code, X.complex),
        C("Design a CQRS event sourcing pipeline using Kafka for financial audit at 10K transactions per second.", D.code, X.complex),
        C("Design a gRPC bidirectional streaming service considering backpressure and circuit breaker at 1M connections.", D.code, X.complex),
        C("Design a distributed Raft cluster with automatic leader failover and 1M state machine operations per day.", D.code, X.complex),
        C("Design a thread-safe LRU cache with TTL expiration in Python considering deadlock prevention and 100K operations per second.", D.code, X.complex),
        C("Design a Kafka-backed saga orchestrator with CQRS rollback for 10K distributed transactions per second.", D.code, X.complex),
        C("Design a sharding strategy for a Postgres database using consistent hashing to support 1B records.", D.code, X.complex),
        C("Design a Python asyncio task scheduler with circuit breaker and deadlock detection for 100K tasks per second.", D.code, X.complex),
        C("Design a GraphQL subscriptions engine with Kafka fan-out for 5M concurrent subscribers.", D.code, X.complex),
        C("Design a multi-region active-active JWT authentication service considering clock skew and 10M daily logins.", D.code, X.complex),
        C("Design a Docker container orchestration system using Kubernetes with circuit breaker at 1M deployments per month.", D.code, X.complex),
        C("Design a distributed tracing pipeline for a microservices architecture using gRPC across multiple regions.", D.code, X.complex),
        C("Design a consensus-based configuration store using Raft for 1K distributed nodes.", D.code, X.complex),
        C("Design a zero-downtime schema migration strategy for a sharded Postgres database with CQRS read models.", D.code, X.complex),
        C("Design a backpressure-aware gRPC streaming server with circuit breaker for 500K concurrent streams.", D.code, X.complex),
        C("Design a Kafka-based CQRS write model with event sourcing replay support at 5M events per day.", D.code, X.complex),
        C("Design a distributed deadlock detector for a thread-safe job scheduler in Python.", D.code, X.complex),
        C("Design a multi-region consistent hashing cache invalidation strategy across multiple availability zones.", D.code, X.complex),
        C("Design a zero-copy serialization pipeline using gRPC and Kafka for 1M events per second.", D.code, X.complex),
        C("Design a Raft-backed distributed key-value store with consistent hashing for sharded partitions.", D.code, X.complex),
        C("Design a circuit breaker state machine in Python considering closed, open, and half-open transitions.", D.code, X.complex),
        C("Design a Kubernetes operator in Golang that reconciles CQRS projection state across multiple clusters.", D.code, X.complex),
        C("Design a CQRS command bus in Python with dead-letter queue handling for 10K commands per second.", D.code, X.complex),
        C("Design a distributed B-tree index using consistent hashing for a 1B-record database.", D.code, X.complex),
        C("Design a JWT introspection cache using consistent hashing and circuit breaker for 10M API calls per day.", D.code, X.complex),
        C("Design a Kafka-backed event sourcing store with CQRS snapshots for 100M events per month.", D.code, X.complex),
        C("Design a multi-region active-active Kafka cluster with CQRS event routing for 500K events per second.", D.code, X.complex),
        # ─── Compare/Analyze with ADVANCED + SCALE ──────────────────────────
        C("Compare consistent hashing vs. rendezvous hashing for routing 1M requests across 100 shards.", D.code, X.complex),
        C("Compare Raft vs. Paxos consensus algorithms for a distributed lock manager handling 100K operations/second.", D.code, X.complex),
        C("Compare CQRS with event sourcing vs. traditional CRUD for a financial system at 10K transactions/second.", D.code, X.complex),
        C("Compare gRPC vs. GraphQL vs. REST for a microservices architecture handling 5M daily requests.", D.code, X.complex),
        C("Analyze trade-offs of using Kafka with CQRS vs. a traditional event bus for 1M events/second.", D.code, X.complex),
        C("Compare sharding strategies (range vs. hash vs. consistent hashing) for a Postgres cluster at 1B records.", D.code, X.complex),
        C("Analyze the consistency guarantees of the Raft consensus algorithm under network partitions at 500K operations/second.", D.code, X.complex),
        C("Compare lock-free vs. mutex-based concurrency for a thread-safe queue in Python at 100K operations/second.", D.code, X.complex),
        C("Compare B-tree vs. LSM-tree indexing for a write-heavy workload at 1M records/second.", D.code, X.complex),
        C("Analyze gRPC backpressure mechanisms vs. Kafka consumer group lag for 10M messages/day.", D.code, X.complex),
        # ─── Implement {word} algorithm/framework/protocol at scale ─────────
        C("Implement a consensus algorithm for distributed coordination using Raft across 5 nodes handling 100K operations/second.", D.code, X.complex),
        C("Implement a lock-free concurrent queue in Python with backpressure for 1M operations/second.", D.code, X.complex),
        C("Implement a consistent hashing ring with virtual nodes across multiple regions for 10M cache keys.", D.code, X.complex),
        C("Implement a thread-safe circuit breaker in Python with half-open recovery and 100K concurrent requests.", D.code, X.complex),
        C("Implement a Raft-based replicated state machine in Golang for 50K operations/second.", D.code, X.complex),
        C("Implement a sharding coordinator using consistent hashing with automatic rebalancing for 1B records.", D.code, X.complex),
        C("Implement a CQRS command handler with event sourcing and Kafka-based projection rebuild.", D.code, X.complex),
        C("Implement a distributed deadlock detection algorithm for a Python asyncio job scheduler at scale.", D.code, X.complex),
        C("Implement a gRPC-based service discovery protocol with circuit breaker failover across multiple regions.", D.code, X.complex),
        C("Implement a zero-downtime Kafka consumer rebalancing strategy with CQRS for 5M events/day.", D.code, X.complex),
        # ─── Build a … system/service/pipeline at scale ─────────────────────
        C("Build a streaming pipeline using Kafka and CQRS that processes 1M events/second with exactly-once semantics.", D.code, X.complex),
        C("Build a gRPC service mesh with consistent hashing load balancing across multiple Kubernetes clusters.", D.code, X.complex),
        C("Build a distributed caching system using consistent hashing and circuit breaker for 500M daily lookups.", D.code, X.complex),
        C("Build a Raft-backed distributed configuration service handling 100K reads/second across multiple regions.", D.code, X.complex),
        C("Build a CQRS event sourcing microservice using Kafka that replays 1B historical events.", D.code, X.complex),
        C("Build a sharding-aware Python database driver with consistent hashing for a 1B-row Postgres cluster.", D.code, X.complex),
        C("Build a multi-region active-active JWT authentication service with circuit breaker for 10M daily users.", D.code, X.complex),
        C("Build a thread-safe Python task scheduler with deadlock detection for 100K concurrent tasks.", D.code, X.complex),
        C("Build a gRPC bidirectional streaming API with backpressure and circuit breaker for 1M clients.", D.code, X.complex),
        C("Build a Kafka-based saga orchestration framework in Python handling 5K distributed transactions/second.", D.code, X.complex),
        # ─── Implement a … algorithm/system/protocol ────────────────────────
        C("Implement a zero-downtime migration protocol for a sharded Postgres database with active-active replication.", D.code, X.complex),
        C("Implement a Paxos-based distributed transaction coordinator for 100K concurrent writes.", D.code, X.complex),
        C("Implement a consistent hashing partitioner for a gRPC service registry across multiple data centers.", D.code, X.complex),
        C("Implement a CQRS projection rebuild system using Kafka replay for 500M historical events.", D.code, X.complex),
        C("Implement a thread-safe, lock-free hash map in Rust for 10M concurrent read/write operations.", D.code, X.complex),
        C("Implement a Raft snapshot and log compaction protocol for a Python distributed key-value store.", D.code, X.complex),
        C("Implement a circuit breaker pattern in Python that coordinates with a Kafka dead-letter queue.", D.code, X.complex),
        C("Implement a multi-region consistent hashing replication strategy in Golang for 100M records.", D.code, X.complex),
        C("Implement a CQRS event sourcing framework in Python with Kafka-backed event store and 1M events/day.", D.code, X.complex),
        C("Implement a thread-safe B-tree in Python with deadlock-safe locking for 1M concurrent queries.", D.code, X.complex),
        C("Implement a backpressure-aware gRPC server with circuit breaker in Golang for 500K streams/second.", D.code, X.complex),
        C("Implement a distributed rate limiter using consistent hashing across multiple Redis shards for 10M API calls/day.", D.code, X.complex),
        C("Implement a Paxos multi-decree consensus protocol in Python for a distributed log at 50K entries/second.", D.code, X.complex),
        C("Implement a zero-copy Kafka consumer in Rust that processes 1M messages/second with backpressure.", D.code, X.complex),
        C("Implement a lock-free ring buffer in Rust for high-throughput inter-thread communication at 10M ops/second.", D.code, X.complex),
        C("Implement an active-active CQRS write model with conflict resolution for a multi-region event store.", D.code, X.complex),
        C("Implement a consistent hashing-based sharding middleware in Python for a 100-node Kafka cluster.", D.code, X.complex),
        C("Implement a gRPC interceptor chain with circuit breaker, JWT validation, and distributed tracing.", D.code, X.complex),
        C("Implement a Raft follower catchup protocol with log compaction for 1M state transitions/day.", D.code, X.complex),
        C("Implement a thread-safe LRU cache with TTL and consistent hashing sharding in Python.", D.code, X.complex),
    ]  # 100 cases
    # reasoning / simple: 2+ reasoning keywords, NO code domain keywords, score < 0.15
    # Template A: "What is the difference between X and Y, considering the trade-offs?"
    # → "difference between" + "trade-off" = 2 hits; WH_SIMPLE(-0.08)+SCALE(0.20)-tok(0.05)=0.07
    _rs_a = [
        ("Agile", "Waterfall methodologies"),
        ("synchronous", "asynchronous team communication"),
        ("monolithic", "microservices architecture"),
        ("open-source", "proprietary software"),
        ("centralized", "decentralized decision-making"),
        ("generalist", "specialist engineering roles"),
        ("pull-based", "push-based notification systems"),
        ("feature flags", "long-lived feature branches"),
        ("hiring junior", "senior developers for growth stages"),
        ("on-site", "remote work policies"),
        ("buying", "building software in-house"),
        ("formal", "informal documentation practices"),
        ("top-down", "bottom-up planning approaches"),
        ("short", "long software release cycles"),
        ("pair programming", "solo development"),
        ("synchronous", "asynchronous code review processes"),
        ("manual", "automated testing strategies"),
        ("continuous delivery", "infrequent large releases"),
        ("code ownership", "shared code ownership models"),
        ("strong", "weak typing in language design"),
        ("static", "dynamic analysis tools for quality"),
        ("scrum", "Kanban for team planning"),
        ("vertical", "horizontal organizational structures"),
        ("proactive", "reactive technical debt management"),
        ("incremental migration", "big-bang replacement strategies"),
    ]
    # Template B: "Why is X important for Y, and what are the trade-offs?"
    # → "why is " + "trade-off" = 2 hits; WH_COMPLEX(short +0.10)-tok(0.05)=0.05
    _rs_b = [
        ("technical debt management", "long-term development velocity"),
        ("documentation", "team knowledge sharing"),
        ("code review", "software quality assurance"),
        ("monitoring", "production reliability"),
        ("retrospective meetings", "team improvement"),
        ("incident postmortems", "organizational learning"),
        ("cross-functional teams", "product delivery speed"),
        ("psychological safety", "team innovation"),
        ("knowledge transfer", "team resilience"),
        ("stakeholder alignment", "project success"),
        ("iterative delivery", "customer feedback cycles"),
        ("design reviews", "architecture quality"),
        ("sprint planning", "team focus and predictability"),
        ("root cause analysis", "incident prevention"),
        ("feedback loops", "continuous organizational improvement"),
        ("capacity planning", "sustainable delivery pace"),
        ("security reviews", "regulatory compliance"),
        ("load testing", "production readiness assurance"),
        ("blameless postmortems", "psychological safety in teams"),
        ("observability", "diagnosing production incidents quickly"),
    ]
    # Template C: "What are the pros and cons of X versus Y?"
    # → "pros and cons" + " versus " = 2 hits; WH_SIMPLE + SCALE_COMPLEX = 0.07
    _rs_c = [
        ("synchronous remote work", "fully asynchronous remote work"),
        ("SQL", "NoSQL for analytical workloads"),
        ("open-source", "commercial off-the-shelf tooling"),
        ("functional", "object-oriented programming paradigms"),
        ("dedicated", "shared infrastructure for teams"),
        ("monolith-first", "microservices-first architecture"),
        ("contract testing", "end-to-end integration testing"),
        ("canary deployments", "blue-green deployment strategies"),
        ("centralized logging", "distributed tracing for observability"),
        ("single-cloud", "multi-cloud infrastructure strategies"),
        ("trunk-based development", "Gitflow branching strategies"),
        ("self-service platforms", "centralized shared services"),
        ("stream-aligned teams", "functional siloed teams"),
        ("evolutionary architecture", "big upfront design"),
        ("hypothesis-driven development", "traditional requirements-based delivery"),
    ]
    # Template D: "Why do teams prefer X, and what is the rationale?"
    # → "why do " + "rationale" = 2 hits; WH_COMPLEX(short +0.10)-tok(0.05)=0.05
    _rs_d = [
        ("agile over waterfall", "software projects"),
        ("async communication over synchronous meetings", "distributed teams"),
        ("flat hierarchies over traditional management", "engineering organizations"),
        ("trunk-based development over long-lived branches", "deployment speed"),
        ("observability over traditional monitoring", "production reliability"),
        ("feature flags over release branches", "risk mitigation"),
        ("blameless postmortems over blame culture", "team learning"),
        ("continuous discovery over big upfront requirements", "product development"),
        ("small batch sizes over large releases", "flow efficiency"),
        ("domain-oriented teams over functional silos", "product ownership"),
        ("evolutionary architecture over planned rewrites", "long-term agility"),
        ("product teams over project teams", "organizational alignment"),
        ("infrastructure as code over manual provisioning", "reliability"),
        ("site reliability engineering over traditional ops", "scalability"),
        ("hypothesis-driven features over assumption-based roadmaps", "product success"),
    ]
    reasoning_s = (
        [C(f"What is the difference between {a} and {b}, considering the trade-offs?", D.reasoning, X.moderate) for a, b in _rs_a]
        + [C(f"Why is {topic} important for {ctx}, and what are the trade-offs?", D.reasoning, X.moderate) for topic, ctx in _rs_b]
        + [C(f"What are the pros and cons of {a} versus {b}?", D.reasoning, X.moderate) for a, b in _rs_c]
        + [C(f"Why do teams prefer {pref}, and what is the rationale?", D.reasoning, X.moderate) for pref, ctx in _rs_d]
    )  # 75 cases — moderate: WH_COMPLEX(+0.10) + tech_terms → score 0.15–0.30

    # reasoning / moderate: 2+ reasoning keywords, no code keywords, 0.15 ≤ score < 0.55
    # Template A: "Explain why X leads to Y and what the trade-offs are."
    # → "explain why" + "trade-off" = 2 hits; WH_EXPLAIN(+0.20) = 0.20 → moderate ✓
    _rm_a = [
        "technical debt accumulation slows long-term development velocity",
        "microservices adoption increases operational complexity",
        "lack of production monitoring leads to silent failures",
        "premature optimization wastes engineering time and the trade-offs are significant",
        "big bang releases increase deployment risk",
        "poor documentation creates knowledge silos",
        "over-engineering increases time to market",
        "lack of code reviews allows defects to accumulate",
        "excessive meeting culture reduces deep work productivity",
        "tight coupling makes systems harder to change",
        "large batch sizes increase work in progress and reduce flow",
        "organizational silos prevent cross-functional collaboration",
        "manual testing bottlenecks slow continuous deployment",
        "lack of observability makes debugging production incidents harder",
        "over-reliance on a single vendor creates supply chain risk",
        "insufficient security reviews increase vulnerability exposure",
        "poor capacity planning leads to burnout and attrition",
        "lack of psychological safety reduces team feedback quality",
        "underspecification in requirements increases downstream rework costs",
        "frequent context switching reduces individual engineering productivity",
        "unclear ownership of shared components leads to quality degradation",
        "underinvestment in infrastructure reliability creates incidents",
        "inadequate knowledge transfer creates single points of failure in teams",
        "lack of automation in provisioning causes configuration drift",
        "ignoring non-functional requirements causes production scalability failures",
    ]
    # Template B: "Analyze the trade-offs and implications of X."
    # → "analyze" + "trade-off" + "implication" = 3 hits; WH_COMPLEX("analyze" +0.30 ≥15 tok)
    _rm_b = [
        "adopting serverless architecture for enterprise workloads",
        "implementing domain-driven design in a legacy codebase",
        "migrating from a monolith to microservices",
        "choosing event-driven over request-response architecture",
        "implementing continuous deployment in regulated industries",
        "adopting infrastructure as code for enterprise provisioning",
        "moving to fully asynchronous team communication",
        "implementing trunk-based development over feature branches",
        "adopting platform engineering over traditional DevOps",
        "using polyglot persistence for different workload types",
        "choosing chaos engineering for reliability testing",
        "adopting a product-led growth model",
        "implementing zero-trust security architecture",
        "migrating from on-premises to cloud-native infrastructure",
        "adopting site reliability engineering practices",
        "using behavior-driven development over test-driven development",
        "introducing architectural decision records",
        "implementing feature flags for large-scale releases",
        "choosing open-source components over proprietary solutions",
        "adopting a platform team model for internal tooling",
        "implementing contract testing between services",
        "using canary deployments over blue-green for releases",
        "adopting domain-oriented team structures over functional silos",
        "introducing observability-first engineering practices",
        "implementing blameless postmortem culture across engineering teams",
    ]
    # Template C: "Compare X and Y, considering their trade-offs and underlying mechanisms."
    # → "compare " + "trade-off" + "underlying" + "mechanism" = 4 hits
    # → WH_COMPLEX("compare" +0.30 ≥15tok) + SCALE_COMPLEX("considering" +0.20) = 0.50 → moderate ✓
    _rm_c_pairs = [
        ("event sourcing", "traditional CRUD for audit trail requirements"),
        ("serverless", "containerized workloads for cost efficiency"),
        ("synchronous RPC", "asynchronous messaging for inter-service communication"),
        ("blue-green deployment", "canary deployment for risk mitigation"),
        ("stream-aligned teams", "traditional project teams for delivery speed"),
        ("site reliability engineering", "traditional operations for production management"),
        ("chaos engineering", "traditional load testing for reliability assurance"),
        ("trunk-based development", "Gitflow for release management reliability"),
        ("monorepo", "polyrepo approaches for large-scale code organization"),
        ("centralized logging", "distributed tracing for production observability"),
        ("contract testing", "end-to-end testing for service integration reliability"),
        ("hypothesis-driven development", "traditional feature development for product teams"),
        ("blameless culture", "accountability culture for incident management"),
        ("continuous discovery", "big upfront requirements for product development"),
        ("infrastructure as code", "manual provisioning for configuration reliability"),
        ("golden path tooling", "team autonomy for developer experience"),
        ("feature-based", "layer-based team organization structures"),
        ("incremental migration", "big bang replacement for legacy modernization"),
        ("push-based", "pull-based deployment strategies for release reliability"),
        ("self-service platforms", "centralized shared services for engineering efficiency"),
        ("single-cloud", "multi-cloud strategies for enterprise infrastructure"),
        ("proactive", "reactive approaches to technical debt management"),
        ("pair programming", "async code review for quality assurance"),
        ("domain-driven design", "data-driven design for complex business rules"),
        ("platform engineering teams", "embedded DevOps for infrastructure ownership"),
    ]
    reasoning_m = (
        [C(f"Explain why {t} and what the trade-offs are.", D.reasoning, X.moderate) for t in _rm_a]
        + [C(f"Analyze the trade-offs and implications of {t}.", D.reasoning, X.moderate) for t in _rm_b]
        + [C(f"Compare {a} and {b}, considering their trade-offs and underlying mechanisms.", D.reasoning, X.moderate) for a, b in _rm_c_pairs]
    )  # 75 cases

    # reasoning / complex: 2+ reasoning keywords, no code keywords, score ≥ 0.55
    # WH_HYPO(+0.40) + SCALE_COMPLEX(+0.20) = 0.60 ✓
    # Template A: "What are the implications of X versus Y, and what are the trade-offs?"
    # → WH_HYPO("what are the implications") + SCALE_COMPLEX("versus")
    # → "implication" + " versus " + "trade-off" = 3 reasoning kws ✓
    _rc_a_pairs = [
        ("organizations fully adopting serverless", "containerized infrastructure"),
        ("teams switching to fully asynchronous communication", "hybrid synchronous models"),
        ("companies adopting monorepo", "polyrepo version control strategies"),
        ("engineering teams embracing blameless culture", "accountability-driven incident models"),
        ("organizations migrating from on-premises", "cloud-native infrastructure"),
        ("companies choosing open-source data pipelines", "proprietary vendor solutions"),
        ("teams adopting event-driven architecture", "traditional request-response systems"),
        ("organizations implementing continuous deployment", "quarterly release cycles"),
        ("companies choosing single-cloud", "multi-cloud platform strategies"),
        ("teams embracing trunk-based development", "long-lived feature branches"),
        ("organizations adopting platform engineering", "traditional DevOps models"),
        ("companies implementing zero-trust security", "perimeter-based security models"),
        ("teams using chaos engineering", "traditional QA testing approaches"),
        ("organizations adopting domain-driven design", "data-driven design approaches"),
        ("companies transitioning to product teams", "project-based delivery models"),
        ("teams implementing contract testing", "end-to-end integration tests"),
        ("organizations choosing site reliability engineering", "traditional operations models"),
        ("companies adopting stream-aligned team topologies", "functional organizational silos"),
        ("teams using canary deployments", "blue-green deployment strategies"),
        ("organizations embracing hypothesis-driven development", "traditional roadmap planning"),
        ("companies implementing observability-first engineering", "traditional monitoring"),
        ("teams adopting infrastructure as code", "manual provisioning approaches"),
        ("organizations choosing polyglot persistence", "single-database strategies"),
        ("companies implementing feature flags", "long-lived release branch strategies"),
        ("teams adopting continuous discovery", "big upfront requirements analysis"),
    ]
    # Template B: "What are the trade-offs between X and Y, considering the underlying mechanisms?"
    # → WH_HYPO("what are the trade-offs") + SCALE_COMPLEX("considering")
    # → "trade-off" + "underlying" + "mechanism" = 3 reasoning kws ✓
    _rc_b_pairs = [
        ("synchronous", "asynchronous team communication approaches"),
        ("centralized", "decentralized technical decision-making"),
        ("monolithic", "microservices architectures for startup velocity"),
        ("platform teams", "embedded operations in stream-aligned organizations"),
        ("short", "long software release cycles from a business perspective"),
        ("pull-based", "push-based deployment pipelines for release reliability"),
        ("proactive", "reactive approaches to managing technical debt"),
        ("feature-based", "component-based team structures in large organizations"),
        ("open-source", "proprietary observability tooling strategies"),
        ("hypothesis-driven", "traditional requirements-based development"),
        ("event-driven", "batch-processing architectures for data workflows"),
        ("synchronous", "asynchronous code review processes at scale"),
        ("self-service", "centralized provisioning models in enterprise engineering"),
        ("blameless", "blame-oriented incident postmortem cultures"),
        ("trunk-based", "branch-based version control strategies for large teams"),
        ("chaos engineering", "traditional load testing for reliability assurance"),
        ("generalist", "specialist engineering roles in product organizations"),
        ("continuous deployment", "staged release strategies under regulatory constraints"),
        ("domain-oriented", "functional team topologies for product delivery"),
        ("observability-driven", "monitoring-driven approaches to production management"),
        ("small-batch", "large-batch delivery models for enterprise software"),
        ("contract testing", "end-to-end testing strategies for integration reliability"),
        ("evolutionary", "planned architectural approaches for long-term maintainability"),
        ("cloud-native", "on-premises infrastructure strategies for organizational agility"),
        ("product-led", "sales-led growth models for scaling software companies"),
    ]
    # Template C: "What are the consequences of X versus Y, and what are the underlying mechanisms?"
    # → WH_HYPO("what are the consequences") + SCALE_COMPLEX("versus")
    # → " versus " + "underlying" + "mechanism" = 3 reasoning kws ✓
    _rc_c_pairs = [
        ("adopting event sourcing for all data", "using traditional CRUD"),
        ("fully automating deployment pipelines", "maintaining manual gating steps"),
        ("merging all teams into a platform model", "keeping stream-aligned product teams"),
        ("eliminating all synchronous meetings", "maintaining a hybrid meeting culture"),
        ("migrating to serverless for all workloads", "using containers throughout"),
        ("adopting a monorepo for all services", "maintaining per-team repositories"),
        ("implementing strict contract testing", "relying on end-to-end integration tests"),
        ("adopting chaos engineering organization-wide", "limiting it to critical systems only"),
        ("using hypothesis-driven development exclusively", "keeping traditional roadmaps"),
        ("fully embracing infrastructure as code", "maintaining some manual provisioning"),
        ("eliminating feature flags after release", "keeping long-lived flags for experiments"),
        ("adopting blameless culture in all retrospectives", "maintaining accountability models"),
        ("migrating all storage to NoSQL", "keeping a mixed SQL and NoSQL approach"),
        ("consolidating all services into a single cloud", "maintaining multi-cloud strategies"),
        ("requiring pair programming for all critical changes", "keeping solo development"),
        ("implementing zero-trust for all internal services", "keeping perimeter-based security"),
        ("adopting domain-driven design for all new services", "allowing technical-layer design"),
        ("centralizing all observability tooling", "allowing team-level tooling choices"),
        ("enforcing trunk-based development across all teams", "allowing team workflow autonomy"),
        ("using canary deployments for all releases", "keeping blue-green for critical systems"),
        ("adopting continuous discovery for all product development", "using phase-gate planning"),
        ("enforcing single ownership for all services", "allowing shared ownership"),
        ("standardizing the technology stack across all teams", "allowing polyglot choices"),
        ("eliminating all manual testing in favor of automation", "keeping exploratory testing"),
        ("requiring architectural decision records for all changes", "using informal documentation"),
    ]
    reasoning_c = (
        [C(f"What are the implications of {a} versus {b}, and what are the trade-offs?", D.reasoning, X.complex) for a, b in _rc_a_pairs]
        + [C(f"What are the trade-offs between {a} and {b}, considering the underlying mechanisms?", D.reasoning, X.complex) for a, b in _rc_b_pairs]
        + [C(f"What are the consequences of {a} versus {b}, and what are the underlying mechanisms?", D.reasoning, X.complex) for a, b in _rc_c_pairs]
    )  # 75 cases
    # extraction / simple: 2+ extraction keywords, score < 0.15
    # "Extract all X from this doc." → "extract"+"field/json/etc" = 2 hits; score ≈ -0.05
    _exs_a = [  # "Extract all {X} from this {doc}."
        ("email addresses", "CSV file"), ("phone numbers", "spreadsheet"),
        ("URLs", "HTML page"), ("dates", "table"), ("IP addresses", "log file"),
        ("prices", "JSON file"), ("product names", "XML feed"),
        ("hashtags", "social media post"), ("country names", "article"),
        ("person names", "document"), ("company names", "annual report"),
        ("invoice numbers", "CSV export"), ("error codes", "log file"),
        ("field names", "JSON schema"), ("column headers", "CSV table"),
    ]
    _exs_b = [  # "List all {X} in this {doc}." — WH_SIMPLE("list") → -0.13
        ("fields", "JSON object"), ("columns", "CSV file"),
        ("rows with errors", "table"), ("URLs", "HTML file"),
        ("keys", "YAML config"), ("table names", "SQL schema"),
        ("environment variables", "config file"), ("email domains", "CSV export"),
        ("image tags", "HTML page"), ("foreign keys", "database schema"),
        ("headers", "HTTP response"), ("query parameters", "URL list"),
        ("section headings", "document"), ("named entities", "article"),
        ("field values", "JSON array"),
    ]
    _exs_c = [  # "Find all {X} from the text." → "find all"+"from the text"=2 hits
        ("email addresses", "customer names"),
        ("phone numbers", "reference numbers"),
        ("dates and deadlines", "dollar amounts"),
        ("URLs and links", "hashtags and mentions"),
        ("IP addresses", "domain names"),
        ("currency values", "product codes"),
        ("person names", "organization names"),
        ("addresses", "zip codes"),
        ("error messages", "status codes"),
        ("timestamps", "log levels"),
        ("bold keywords", "capitalized terms"),
        ("numbered items", "bullet points"),
        ("quoted strings", "parenthetical notes"),
        ("version numbers", "file paths"),
        ("table rows", "field labels"),
    ]
    _exs_d = [  # "Get all/Retrieve all/Identify all X from this Y."
        ("Get all", "column values", "from this", "CSV"),
        ("Retrieve all", "field names", "from this", "JSON schema"),
        ("Identify all", "keys with null values", "from this", "JSON"),
        ("Get all", "table names", "from this", "SQL schema"),
        ("Retrieve all", "rows that contain errors", "from this", "table"),
        ("Identify all", "URLs with 404 status", "from this", "server log"),
        ("Get all", "environment variable names", "from this", "config file"),
        ("Retrieve all", "image source paths", "from this", "HTML file"),
        ("Identify all", "foreign key relationships", "from this", "database schema"),
        ("Get all", "query parameters", "from this", "URL list"),
        ("Retrieve all", "section headers", "from this", "document"),
        ("Identify all", "nested field names", "from this", "JSON"),
        ("Get all", "column headers", "from this", "CSV table"),
        ("Retrieve all", "named entities", "from this", "article"),
        ("Identify all", "key-value pairs", "from this", "XML"),
    ]
    extraction_s = (
        [C(f"Extract all {x} from this {d}.", D.extraction, X.simple) for x, d in _exs_a]
        + [C(f"List all {x} in this {d}.", D.extraction, X.simple) for x, d in _exs_b]
        + [C(f"Find all {a} and {b} from the text.", D.extraction, X.simple) for a, b in _exs_c]
        + [C(f"{action} {x} {prep} this {doc}.", D.extraction, X.simple) for action, x, prep, doc in _exs_d]
    )  # 60 cases

    # extraction / moderate: 2+ extraction keywords, 0.15 ≤ score < 0.55
    # "Parse this {doc} and {action}." → IMP_MODERATE("parse ") +0.25; 2+ extraction kws
    _exm_parse = [
        ("JSON API response", "extract all fields where the value is null or an empty array"),
        ("CSV export", "identify all rows where the email column is missing"),
        ("XML configuration", "retrieve all environment variables and their default values"),
        ("SQL schema file", "list all table names, column names, and foreign key relationships"),
        ("nginx access log", "extract all 4xx and 5xx errors with paths and timestamps"),
        ("server log", "get all unique IP addresses that appear more than 50 times"),
        ("Kubernetes YAML", "extract all container image names and version tags"),
        ("HTML page", "identify all form fields and their input types"),
        ("JSON array", "find all entries where the status field is not active"),
        ("YAML config", "extract all environment-specific overrides and their values"),
        ("CSV of transactions", "retrieve all rows where the amount exceeds a threshold"),
        ("XML sitemap", "list all URLs and their last-modified dates"),
        ("audit log", "identify all failed login attempts and associated user IDs"),
        ("ERP export", "extract all open purchase orders and their line items"),
        ("transaction log", "get all records grouped by merchant category"),
    ]
    # "Extract all X from this {doc_type} and {secondary_action}."
    # → "extract" + SCALE_COMPLEX("from this {doc_type}") → +0.20; 2+ extraction kws
    _exm_from = [
        ("email addresses and phone numbers", "report", "organize them into a structured CSV"),
        ("product names and prices", "HTML", "output them as a JSON array"),
        ("named entities", "article", "classify each as person, organization, or location"),
        ("date fields", "document", "identify any that are in an inconsistent format"),
        ("key-value pairs", "log", "flag any entries where the value is empty"),
        ("column headers and sample rows", "CSV", "identify columns with mixed data types"),
        ("table names and row counts", "database", "list tables with more than 10K rows"),
        ("error codes and messages", "server log", "group them by error category"),
        ("nested JSON keys", "schema", "identify keys that appear at more than two levels"),
        ("currency values and line items", "contract", "sum totals by category"),
        ("all URL parameters", "sitemap", "identify duplicate query strings"),
        ("all timestamps and durations", "transcript", "find gaps longer than 5 minutes"),
        ("all IP addresses and request counts", "access log", "sort by frequency"),
        ("all configuration keys", "YAML", "flag any that override the default values"),
        ("all field names and types", "JSON", "identify fields with inconsistent types"),
    ]
    # "From this {doc_type}, extract/list/identify all {X}." → SCALE_COMPLEX("from this {doc_type}") +0.20
    _exm_from2 = [
        ("HTML", "extract all hyperlinks and their anchor text"),
        ("XML", "list all attributes and their parent elements"),
        ("JSON", "retrieve all nested arrays and flatten them one level"),
        ("CSV", "identify all columns that contain only numeric values"),
        ("log", "extract all unique session IDs and their request counts"),
        ("schema", "list all nullable columns and their table names"),
        ("config", "extract all commented-out settings and their keys"),
        ("database", "identify all tables without a primary key"),
        ("document", "extract all section headings and their page numbers"),
        ("contract", "identify all defined terms and their first usage"),
        ("transcript", "extract all questions and their timestamps"),
        ("report", "list all tables and their column headers"),
        ("nginx", "extract all redirect rules and their target URLs"),
        ("sitemap", "identify all pages with a priority score below 0.5"),
        ("kubernetes", "extract all resource limits and request values per container"),
    ]
    extraction_m = (
        [C(f"Parse this {d} and {a}.", D.extraction, X.moderate) for d, a in _exm_parse]
        + [C(f"Extract all {x} from this {d} and {a}.", D.extraction, X.moderate) for x, d, a in _exm_from]
        + [C(f"From this {d}, {a}.", D.extraction, X.moderate) for d, a in _exm_from2]
    )  # 60 cases

    # extraction / complex: 2+ extraction keywords, score ≥ 0.55
    # IMP_DESIGN("design a/an/build a/implement a") + SCALE_COMPLEX = 0.55+ ✓
    _exc_design = [
        ("data extraction pipeline", "extract structured fields", "from 10M records"),
        ("log parsing service", "extract and classify log entries", "from 1M log records"),
        ("entity extraction system", "identify all named entities", "from 500K documents"),
        ("document parsing pipeline", "extract tables and fields", "from 1B records"),
        ("metadata extraction framework", "extract all field names and types", "from 100K JSON files"),
        ("web scraping platform", "extract product data and prices", "from 10M HTML pages"),
        ("invoice parsing algorithm", "extract line items and totals", "from 500K invoices"),
        ("log analysis pipeline", "extract all error rows and timestamps", "from 1M log entries"),
        ("schema extraction service", "parse and extract table definitions", "from 10K SQL schemas"),
        ("contract data extraction system", "identify dates and monetary fields", "from 100K contracts"),
        ("NLP entity extraction pipeline", "extract persons, orgs, and locations", "from 10M articles"),
        ("CSV transformation service", "extract and normalize all field values", "from 1M CSV rows"),
        ("XML parsing framework", "extract all attribute values", "from 500K XML feeds"),
        ("access log extraction system", "parse and extract all unique IPs", "from 1B log events"),
        ("transaction parsing algorithm", "extract merchant and amount fields", "from 100M transactions"),
    ]
    _exc_build = [
        ("Build a data pipeline to extract all", "fields and column values", "10M records"),
        ("Build an extraction system to parse and retrieve", "structured table data", "100K CSV files"),
        ("Build a document parsing platform to extract", "named entities and dates", "1M documents"),
        ("Build an ETL service to extract and transform", "all field values", "500K JSON records"),
        ("Build a log extraction service to identify", "all error rows and timestamps", "1B log events"),
        ("Build an entity extraction pipeline to parse", "persons and organizations", "10M articles"),
        ("Build a schema extraction tool to retrieve", "all table and column definitions", "10K databases"),
        ("Build a contract parsing service to extract", "all dates and monetary amounts", "500K contracts"),
        ("Build a web extraction system to scrape", "product names and prices", "10M HTML pages"),
        ("Build an invoice parsing pipeline to extract", "all line items and totals", "1M invoices"),
        ("Build a log parsing service to collect", "all unique IPs and request counts", "1B log entries"),
        ("Build a metadata extraction framework to retrieve", "all nested field names", "100M JSON files"),
        ("Build a CSV parsing system to identify", "all rows with missing fields", "50M records"),
        ("Build a sitemap extraction service to collect", "all URLs and last-modified dates", "1M sitemaps"),
        ("Build a transaction extraction pipeline to parse", "all merchant and amount fields", "100M records"),
    ]
    _exc_impl = [
        ("a parsing algorithm to extract structured fields from CSV", "10M records"),
        ("a log extraction framework to collect all error rows from server logs", "1B events"),
        ("an entity extraction pipeline to parse named entities from article text", "500K documents"),
        ("a document parsing system to retrieve tables and fields from contracts", "100K files"),
        ("an ETL pipeline to extract and transform all values from nested JSON", "1M records"),
        ("a schema parsing algorithm to extract all table definitions from SQL", "10K databases"),
        ("a metadata extraction service to collect all attribute values from XML", "500K feeds"),
        ("a data extraction framework to parse invoice line items and totals", "1M invoices"),
        ("a log parsing pipeline to identify all unique IP addresses and sessions", "1B log lines"),
        ("a NLP extraction system to find all named entities across a corpus", "10M articles"),
        ("a transaction parsing algorithm to extract merchant and amount fields", "100M records"),
        ("a web scraping service to collect product data from HTML pages", "10M pages"),
        ("a contract data extraction pipeline to identify dates and monetary fields", "500K contracts"),
        ("an access log extraction system to collect all 4xx and 5xx errors", "1B requests"),
        ("a sitemap parsing framework to retrieve all URLs and priorities", "10M sitemaps"),
    ]
    extraction_c = (
        [C(f"Design a {name} to {action} from this scale of {scale}.", D.extraction, X.complex) for name, action, scale in _exc_design]
        + [C(f"{prefix} {x} from {scale}.", D.extraction, X.complex) for prefix, x, scale in _exc_build]
        + [C(f"Implement {desc} at {scale}.", D.extraction, X.complex) for desc, scale in _exc_impl]
    )  # 60 cases
    # classification / simple: 2+ classification keywords, score < 0.15
    _cls_a = [  # "Classify each {X} and assign a {Y} label."
        ("email", "spam or not-spam"), ("transaction", "fraud or legitimate"),
        ("support ticket", "billing, technical, or general"), ("document", "public, internal, or confidential"),
        ("product review", "positive, neutral, or negative"), ("news article", "politics, sports, or technology"),
        ("image", "appropriate or inappropriate"), ("log entry", "error, warning, or info"),
        ("customer message", "complaint, inquiry, or feedback"), ("form submission", "valid or invalid"),
        ("user account action", "normal or suspicious"), ("job application", "qualified or unqualified"),
        ("code comment", "todo, fixme, or note"), ("pull request", "bug-fix, feature, or refactor"),
        ("incident report", "P1, P2, or P3 severity"),
    ]
    _cls_b = [  # "What category does this X belong to?"  → "what category"+"belongs to"=2 hits
        "document", "support ticket", "transaction", "email",
        "product review", "user complaint", "feature request", "API response",
        "log entry", "code change", "security alert", "data record",
        "customer inquiry", "system event", "deployment artifact",
    ]
    _cls_c = [  # "Is this X a {type} or {type2}?" → "is this " + second kw
        ("email", "spam", "label "),        ("message", "automated", "tag "),
        ("document", "public", "tier "),    ("review", "positive", "segment"),
        ("ticket", "bug", "tier "),         ("record", "duplicate", "bucket"),
        ("alert", "false positive", "classify"), ("request", "valid", "label "),
        ("transaction", "fraudulent", "tag "), ("entry", "error", "classify"),
        ("comment", "actionable", "label "), ("event", "critical", "tier "),
        ("file", "sensitive", "classify"), ("record", "stale", "bucket"),
        ("post", "spam", "tag "),
    ]
    _cls_d = [  # "Label each {X} with its tier and assign a category."
        "support ticket", "customer complaint", "system event",
        "incident report", "feature request", "pull request",
        "product review", "user action", "security alert",
        "log entry", "API call", "email thread",
        "deployment event", "data record", "audit trail entry",
    ]
    classification_s = (
        [C(f"Classify each {x} and assign a {c} label.", D.classification, X.simple) for x, c in _cls_a]
        + [C(f"What category does this {x} belong to?", D.classification, X.simple) for x in _cls_b]
        + [C(f"Is this {x} a {t}? Assign a {kw}.", D.classification, X.simple) for x, t, kw in _cls_c]
        + [C(f"Label each {x} with its tier and assign a category.", D.classification, X.simple) for x in _cls_d]
    )  # 60 cases

    # classification / moderate: 2+ classification keywords, 0.15 ≤ score < 0.55
    # IMP_MODERATE("classify these"/"categorize these"/"tag each"/"assign each") +0.25
    # SCALE_COMPLEX("these N") +0.20 → total 0.45 → moderate ✓
    _clm_a_items = [  # "Classify these {N} {items} by X and assign labels."
        (200, "support tickets"), (50, "customer complaints"), (500, "product reviews"),
        (100, "log entries"), (300, "transaction records"), (75, "feature requests"),
        (1000, "emails"), (150, "pull requests"), (400, "data records"),
        (80, "security alerts"), (250, "API responses"), (60, "incident reports"),
        (350, "user actions"), (90, "deployment artifacts"), (700, "system events"),
    ]
    _clm_b_items = [  # "Categorize these {N} {items} into tiers and tag each."
        (200, "customer messages"), (50, "bug reports"), (500, "news articles"),
        (100, "documents"), (300, "product descriptions"), (75, "user sessions"),
        (1000, "comments"), (150, "system events"), (400, "log lines"),
        (80, "code changes"), (250, "alerts"), (60, "feedback forms"),
        (350, "content items"), (90, "feature requests"), (700, "transactions"),
    ]
    _clm_c_items = [  # "Tag each of these {N} {items} and segment by criteria."
        (200, "emails"), (50, "records"), (500, "reviews"),
        (100, "events"), (300, "tickets"), (75, "messages"),
        (1000, "entries"), (150, "requests"), (400, "alerts"),
        (80, "documents"), (250, "responses"), (60, "artifacts"),
        (350, "comments"), (90, "actions"), (700, "items"),
    ]
    classification_m = (
        [C(f"Classify these {n} {items} by urgency and assign a priority label to each.", D.classification, X.moderate) for n, items in _clm_a_items]
        + [C(f"Categorize these {n} {items} into tiers and tag each with a class label.", D.classification, X.moderate) for n, items in _clm_b_items]
        + [C(f"Tag each of these {n} {items} and segment by type, assigning each to a bucket.", D.classification, X.moderate) for n, items in _clm_c_items]
    )  # 60 cases

    # classification / complex: 2+ classification keywords, score ≥ 0.55
    # IMP_DESIGN("design a {word} system/service/algorithm/framework") + SCALE_COMPLEX(scale unit) = 0.55+
    _clc_design = [  # "Design a classification system to categorize {scale} into tiers."
        ("1M support tickets", "urgency tier and segment"),
        ("500K customer emails", "intent category and priority label"),
        ("10M product reviews", "sentiment tier and quality bucket"),
        ("100K log entries", "severity label and component segment"),
        ("5M transactions", "fraud tier and risk bucket"),
        ("2M news articles", "topic taxonomy and relevance segment"),
        ("1M user actions", "intent class label and risk tier"),
        ("500K incident reports", "P1/P2/P3 tier and resolution bucket"),
        ("10M social media posts", "toxicity label and content segment"),
        ("1M API requests", "type tier and priority segment"),
    ]
    _clc_service = [  # "Build a classification service to label {scale} and assign them to buckets."
        ("500K messages", "intent and urgency"), ("1M records", "type and tier"),
        ("10M events", "category and priority"), ("100K documents", "sensitivity and segment"),
        ("2M reviews", "sentiment and quality"), ("5M transactions", "risk and fraud tier"),
        ("1M alerts", "severity and escalation tier"), ("500K requests", "type and routing tier"),
        ("10K incidents", "P-level and component segment"), ("1M emails", "intent and priority bucket"),
    ]
    _clc_impl = [  # "Implement a classification algorithm/framework to segment {scale} into tiers."
        ("a classification algorithm to segment 1M records into urgency tiers", "and assign labels"),
        ("a classification framework to categorize 500K messages by intent", "and assign priority buckets"),
        ("a taxonomy-based classification system to label 10M events", "by type and tier"),
        ("a classification service to segment 1M documents into sensitivity tiers", "and assign class labels"),
        ("a multi-tier classification algorithm to group 500K requests", "by category and assign buckets"),
        ("a classification pipeline to label 1M transactions by risk tier", "and segment by merchant"),
        ("a content classification framework to assign labels to 10M articles", "by topic and relevance"),
        ("a real-time classification service to tag 1M API calls", "by type and route them to buckets"),
        ("a hierarchical classification system to segment 500K customer records", "into tiers and labels"),
        ("a classification algorithm to group 1M log entries by severity", "and assign category labels"),
        ("a classification service to assign each of 500K tickets to a tier", "based on urgency and topic"),
        ("a taxonomy system to classify 1M product descriptions into segments", "and assign class labels"),
        ("a classification pipeline to tag 10M social posts by content type", "and assign moderation tiers"),
        ("a risk classification system to segment 1M transactions", "into fraud and legitimate buckets"),
        ("a multi-label classification framework to assign 500K documents", "to one or more segments"),
        ("a classification algorithm to tier 10M user events", "by engagement level and intent bucket"),
        ("a content taxonomy system to segment 1M support tickets", "into buckets by topic and tier"),
        ("a fraud classification service to label 500K transactions", "by risk tier and segment"),
        ("a classification pipeline to assign urgency tiers to 1M alerts", "and route to segments"),
        ("a taxonomy-based classification framework to group 10K incidents", "into P-level tiers"),
    ]
    classification_c = (
        [C(f"Design a classification system to categorize {scale} by {criteria}.", D.classification, X.complex) for scale, criteria in _clc_design]
        + [C(f"Build a classification service to label {scale} and assign them to {criteria} buckets.", D.classification, X.complex) for scale, criteria in _clc_service]
        + [C(f"Implement {desc} {action}.", D.classification, X.complex) for desc, action in _clc_impl]
    )  # 60 cases
    # summarization / simple: 2+ summ keywords, score < 0.15
    # IMP_WRITE("summarize ")+(0.15) - tok_penalty(<20 tok, -0.05) = 0.10 → simple ✓
    _sums_docs = [
        "meeting notes", "quarterly report", "technical spec", "legal contract",
        "research paper", "project proposal", "annual review", "incident report",
        "product roadmap", "design document", "changelog", "status update",
        "press release", "executive summary", "newsletter article",
    ]
    _sums_kp = [  # "What are the key points and main takeaways from this X?"
        "article", "chapter", "presentation", "workshop",
        "conference talk", "blog post", "white paper", "case study",
        "industry report", "earnings call", "product launch announcement",
        "retrospective", "sprint review", "team standup", "customer interview",
    ]
    _sums_other = [  # varied simple patterns
        C("Give a brief overview and synopsis of this document.", D.summarization, X.simple),
        C("Give a brief overview of this article and highlight the main idea.", D.summarization, X.simple),
        C("What is the main idea and abstract of this research paper?", D.summarization, X.simple),
        C("Distill the key points from this article.", D.summarization, X.simple),
        C("Distill the main takeaways and key points from this chapter.", D.summarization, X.simple),
        C("Sum up this document and highlight the key takeaways.", D.summarization, X.simple),
        C("Provide a brief overview and key points for this report.", D.summarization, X.simple),
        C("Summarise this paragraph and highlight the main idea.", D.summarization, X.simple),
        C("Condense this article and highlight the key points.", D.summarization, X.simple),
        C("What is the gist of this document and the key takeaways?", D.summarization, X.simple),
        C("Boil down this report to its main takeaways and key points.", D.summarization, X.simple),
        C("What are the main takeaways and abstract from this white paper?", D.summarization, X.simple),
        C("Provide a recap and key points from this presentation.", D.summarization, X.simple),
        C("Give a TL;DR of this chapter and list the key takeaways.", D.summarization, X.simple),
        C("Summarize this chapter and provide a brief overview of the abstract.", D.summarization, X.simple),
        C("What is the overview of this proposal and its main idea?", D.summarization, X.simple),
        C("Shorten this memo into a brief summary of the key points.", D.summarization, X.simple),
        C("What is the synopsis of this article and its key takeaways?", D.summarization, X.simple),
        C("Give the gist of this transcript and the main takeaways.", D.summarization, X.simple),
        C("Provide a brief overview and highlight the key points of this update.", D.summarization, X.simple),
        C("Sum up the key takeaways and abstract from this report.", D.summarization, X.simple),
        C("Boil down this document to a brief overview and main idea.", D.summarization, X.simple),
        C("What are the key points and main takeaways from this spec?", D.summarization, X.simple),
        C("Condense this chapter and distill the main idea.", D.summarization, X.simple),
        C("Summarize this article and give the key points.", D.summarization, X.simple),
        C("What is the gist of this report and its key takeaways?", D.summarization, X.simple),
        C("Give a brief synopsis and key points from this research paper.", D.summarization, X.simple),
        C("Provide a TL;DR and main takeaways for this document.", D.summarization, X.simple),
        C("What is the overview and abstract of this paper?", D.summarization, X.simple),
        C("Sum up this blog post and highlight the key takeaways.", D.summarization, X.simple),
    ]
    summarization_s = (
        [C(f"Summarize this {d} and give the key points.", D.summarization, X.simple) for d in _sums_docs]
        + [C(f"What are the key points and main takeaways from this {d}?", D.summarization, X.simple) for d in _sums_kp]
        + _sums_other
    )  # 15+15+30 = 60 cases

    # summarization / moderate: 2+ summ keywords, 0.15 ≤ score < 0.55
    # IMP_WRITE("summarize/condense") + SECONDARY_REQ("and highlight") → 0.23; or + SCALE_COMPLEX → 0.35
    _summ_and_highlight = [
        "annual report", "research paper", "product requirements document",
        "legal contract", "technical specification", "incident postmortem",
        "earnings transcript", "strategic plan", "board meeting notes",
        "customer feedback survey", "competitive analysis", "market research report",
        "design document", "sprint retrospective notes", "quarterly business review",
    ]
    _summ_50page = [  # "Summarize key findings from this 50-page X and highlight main takeaways."
        "contract", "report", "document", "proposal", "specification",
        "audit", "transcript", "white paper", "research study", "policy document",
    ]
    _summ_condense = [  # "Condense this X into 3 key points and highlight the main takeaways."
        "technical document", "meeting transcript", "product roadmap",
        "stakeholder presentation", "quarterly report", "research paper",
        "incident report", "project charter", "architecture review", "business case",
    ]
    _summ_distill = [  # "Distill the key insights from this X and highlight the main takeaways."
        "industry analysis", "competitive landscape", "market report",
        "earnings call transcript", "strategy document", "executive briefing",
        "conference keynote", "customer research", "analyst report", "product review",
    ]
    _summ_other_mod = [
        C("Summarize this document and identify the key decisions and main takeaways.", D.summarization, X.moderate),
        C("Condense this 50-page report and highlight the key insights and main takeaways.", D.summarization, X.moderate),
        C("Provide a detailed overview and key takeaways from this annual report.", D.summarization, X.moderate),
        C("Summarize this research paper and highlight the abstract and conclusions.", D.summarization, X.moderate),
        C("Distill the main ideas and key takeaways from these three conflicting reports.", D.summarization, X.moderate),
        C("Summarize the key findings from this database of 200 customer reviews, highlighting trends.", D.summarization, X.moderate),
        C("Condense this transcript and highlight the key points and action items.", D.summarization, X.moderate),
        C("Create a concise summary and highlight the main takeaways from this white paper.", D.summarization, X.moderate),
        C("Summarize this document, and identify all key decisions and main takeaways.", D.summarization, X.moderate),
        C("Provide an executive summary and key points from this earnings call transcript.", D.summarization, X.moderate),
    ]
    summarization_m = (
        [C(f"Summarize this {d} and highlight the key points and main takeaways.", D.summarization, X.moderate) for d in _summ_and_highlight]
        + [C(f"Summarize the key findings from this 50-page {d} and highlight the main takeaways.", D.summarization, X.moderate) for d in _summ_50page]
        + [C(f"Condense this {d} into 3 key points and highlight the main takeaways.", D.summarization, X.moderate) for d in _summ_condense]
        + [C(f"Distill the key insights from this {d} and highlight the main takeaways.", D.summarization, X.moderate) for d in _summ_distill]
        + _summ_other_mod
    )  # 15+10+10+10+10 = 55... need 5 more

    # fill to 60
    summarization_m += [
        C("Summarize this report and identify the key action items and main takeaways.", D.summarization, X.moderate),
        C("Condense this 50-page contract into a brief overview and highlight the key points.", D.summarization, X.moderate),
        C("Summarize this series of articles and highlight the main takeaways and key themes.", D.summarization, X.moderate),
        C("Distill the key points from these five conflicting reports and highlight discrepancies.", D.summarization, X.moderate),
        C("Create a detailed summary of this incident report and highlight the key findings.", D.summarization, X.moderate),
    ]  # +5 = 60

    # summarization / complex: 2+ summ keywords, score ≥ 0.55
    # IMP_DESIGN("design a/build a {word} service/pipeline") + SCALE_COMPLEX(scale) = 0.55+
    _sumc_design = [
        ("summarization pipeline", "distill key points from 1M records"),
        ("abstractive summarization service", "condense key insights from 500K documents"),
        ("document summarization system", "generate key points from 10M articles"),
        ("summarization framework", "produce abstracts and key takeaways from 100K reports"),
        ("automated summary service", "condense main takeaways from 1M meeting transcripts"),
        ("content distillation pipeline", "summarize key points from 500K user feedback records"),
        ("multi-document summarization system", "synthesize key insights from 10K conflicting reports"),
        ("executive summary generator", "distill key takeaways from 100K earnings call transcripts"),
        ("abstractive summarization framework", "condense research papers into abstracts for 1M records"),
        ("knowledge distillation pipeline", "extract key points and synopses from 500K articles"),
    ]
    _sumc_build = [
        ("a summarization service to distill key points", "from 1M records"),
        ("a document summary pipeline to generate abstracts", "for 500K reports"),
        ("a multi-document summarization system to condense key insights", "from 10M articles"),
        ("a summarization framework to produce main takeaways", "from 100K documents"),
        ("an abstractive summarization service to condense meeting notes", "from 1M records"),
        ("a content distillation pipeline to extract key points", "from 500K feedback records"),
        ("a summarization platform to generate executive summaries", "from 100K earnings calls"),
        ("a key-insight extraction service to distill synopses", "from 10M research papers"),
        ("a multi-source summarization engine to highlight main takeaways", "from 1M messages"),
        ("a document condensation pipeline to generate brief overviews", "from 500K contracts"),
    ]
    _sumc_impl = [
        ("a summarization algorithm to distill key points from 1M documents", "into concise abstracts"),
        ("an abstractive summarization pipeline to condense 500K reports", "into main takeaways"),
        ("a multi-document summarization framework to synthesize 10K reports", "into key insights"),
        ("a key-takeaway extraction service to distill 1M meeting transcripts", "into brief overviews"),
        ("a summarization pipeline to generate abstracts from 500K research papers", "at scale"),
        ("an executive summary generator to condense 100K quarterly reports", "into key points"),
        ("a content distillation algorithm to extract main ideas from 10M articles", "efficiently"),
        ("a summarization service to produce synopses from 1M technical documents", "automatically"),
        ("a multi-source summarization framework to reconcile key points from 500K conflicting reports", ""),
        ("a document condensation pipeline to generate brief overviews from 100K contracts", "at scale"),
        ("an automated abstraction service to distill 1M papers into key takeaways", ""),
        ("a real-time summarization engine to condense 500K messages into brief overviews", ""),
        ("a knowledge distillation system to extract key insights from 10M records", ""),
        ("a summarization algorithm to generate key points and synopses from 100K reports", ""),
        ("a hierarchical summarization pipeline to distill 1M documents into executive abstracts", ""),
        ("a multi-tier summarization framework to condense 500K articles into key takeaways", ""),
        ("a document abstraction service to produce brief overviews from 1M legal contracts", ""),
        ("a scalable summarization system to distill key points from 10M customer reviews", ""),
        ("a content summarization pipeline to generate main ideas from 500K product descriptions", ""),
        ("a summary distillation framework to condense 100K reports into synopses", ""),
    ]
    summarization_c = (
        [C(f"Design a {name} to {action}.", D.summarization, X.complex) for name, action in _sumc_design]
        + [C(f"Build {prefix} {suffix}.", D.summarization, X.complex) for prefix, suffix in _sumc_build]
        + [C(f"Implement {desc}{extra}.", D.summarization, X.complex) for desc, extra in _sumc_impl]
    )  # 10+10+20 = 40... need 20 more
    summarization_c += [
        C("Design a summarization system to condense and distill key points from 1M customer records.", D.summarization, X.complex),
        C("Design a content distillation pipeline to summarize key insights from 500K user messages.", D.summarization, X.complex),
        C("Build a summarization service to generate abstracts and key takeaways from 10M documents.", D.summarization, X.complex),
        C("Build a multi-document summarization platform to condense 1M reports into brief overviews.", D.summarization, X.complex),
        C("Build a key-insight distillation system to summarize 500K meeting transcripts into abstracts.", D.summarization, X.complex),
        C("Design a real-time abstractive summarization framework to condense 1M messages into synopses.", D.summarization, X.complex),
        C("Design a hierarchical summarization pipeline to distill 10M articles into key points.", D.summarization, X.complex),
        C("Build a cross-document summarization engine to condense 500K conflicting reports into key insights.", D.summarization, X.complex),
        C("Design an executive summary generator to distill key takeaways from 100K earnings records.", D.summarization, X.complex),
        C("Build a document condensation service to generate brief overviews from 1M legal contracts.", D.summarization, X.complex),
        C("Design a knowledge distillation system to extract key points and abstracts from 10M records.", D.summarization, X.complex),
        C("Build a multi-tier summarization pipeline to condense 500K articles into key takeaways.", D.summarization, X.complex),
        C("Design a content abstraction service to produce synopses from 1M technical documents.", D.summarization, X.complex),
        C("Build an automated summarization framework to distill 100K quarterly reports into main ideas.", D.summarization, X.complex),
        C("Design a summarization engine to condense key insights from 10M customer feedback records.", D.summarization, X.complex),
        C("Build a scalable abstractive summarization system to distill 500K papers into brief overviews.", D.summarization, X.complex),
        C("Design a real-time key-takeaway extraction service to condense 1M documents.", D.summarization, X.complex),
        C("Build a hierarchical key-points distillation system from 500K conflicting technical documents.", D.summarization, X.complex),
        C("Design a multi-source summarization pipeline to synthesize key insights from 10M records.", D.summarization, X.complex),
        C("Build a document synopsis generator to distill 100K reports into abstracts and main takeaways.", D.summarization, X.complex),
    ]  # +20 = 60

    # creative / simple: 2+ creative keywords, score < 0.15
    # "Write a poem/haiku/limerick about X." → "write a " + poetry kw; IMP_WRITE(0.15)-tok(0.05)=0.10
    _crs_poem_topics = [
        "autumn leaves", "the ocean at night", "a city at dawn",
        "lost memories", "the joy of discovery", "a quiet library",
        "morning coffee", "a mountain hike", "change and growth",
        "friendship and distance", "the night sky", "a rainy afternoon",
    ]
    _crs_haiku_topics = [
        "spring", "solitude", "a new beginning",
        "the moon", "a passing train", "teamwork",
        "failure and recovery", "a first day at work", "the color blue",
    ]
    _crs_story_topics = [
        "a robot who discovers emotions", "a time traveler's regret",
        "a chef who communicates through recipes",
        "a lighthouse keeper and a mystery",
        "a child who befriends an unlikely animal",
        "two strangers who keep crossing paths",
    ]
    _crs_tagline_products = [
        "a new coffee brand", "a fitness app", "a cloud storage service",
        "an online bookshop", "a meal delivery startup", "a mindfulness platform",
        "a remote team tool", "a language learning app", "a financial planning service",
    ]
    creative_s = (
        [C(f"Write a poem about {t}.", D.creative, X.simple) for t in _crs_poem_topics]
        + [C(f"Write a haiku about {t}.", D.creative, X.simple) for t in _crs_haiku_topics]
        + [C(f"Write a short story about {t}.", D.creative, X.simple) for t in _crs_story_topics]
        + [C(f"Come up with a catchy tagline for {p}.", D.creative, X.simple) for p in _crs_tagline_products]
        + [
            C("Write a limerick about working from home.", D.creative, X.simple),
            C("Write a limerick about software bugs.", D.creative, X.simple),
            C("Write a motivational quote about perseverance.", D.creative, X.simple),
            C("Write a motivational message for a team kickoff.", D.creative, X.simple),
            C("Draft a short, creative birthday message for a colleague.", D.creative, X.simple),
            C("Draft a creative out-of-office message for a vacation.", D.creative, X.simple),
            C("Generate ideas for a newsletter about remote work.", D.creative, X.simple),
            C("Generate ideas for a blog post about startup culture.", D.creative, X.simple),
            C("Brainstorm ideas for a creative team-building event.", D.creative, X.simple),
            C("Brainstorm ideas for a marketing campaign about sustainability.", D.creative, X.simple),
            C("Come up with a brand name for a wellness startup.", D.creative, X.simple),
            C("Come up with a slogan for a new productivity tool.", D.creative, X.simple),
            C("Write a verse about the beauty of simplicity in design.", D.creative, X.simple),
            C("Write a sonnet about overcoming imposter syndrome.", D.creative, X.simple),
            C("Create a short metaphor comparing agile development to jazz improvisation.", D.creative, X.simple),
        ]
    )  # 12+9+6+9+15 = 51 ... need 9 more
    creative_s += [
        C("Create a slogan for a sustainable clothing brand.", D.creative, X.simple),
        C("Write a poem about the beauty of clean code.", D.creative, X.simple),
        C("Write a haiku about a morning standup meeting.", D.creative, X.simple),
        C("Come up with a tagline for a developer productivity tool.", D.creative, X.simple),
        C("Brainstorm ideas for a press release about a new product launch.", D.creative, X.simple),
        C("Write a limerick about a never-ending backlog.", D.creative, X.simple),
        C("Draft a creative birthday card message for a remote colleague.", D.creative, X.simple),
        C("Generate ideas for a blog post about work-life balance.", D.creative, X.simple),
        C("Come up with a brand name for a no-code platform.", D.creative, X.simple),
    ]  # +9 = 60

    # creative / moderate: 2+ creative keywords, 0.15 ≤ score < 0.55
    # IMP_WRITE(0.15) + WORD_COUNT(0.15) = 0.30; or + QUALITY_ADJ(0.08) = 0.23
    _crm_wordcount_topics = [  # "Write a {N}-word blog post about X." → IMP_WRITE + WORD_COUNT
        ("500-word", "blog post", "the future of remote work"),
        ("500-word", "blog post", "machine learning in healthcare"),
        ("500-word", "blog post", "sustainable business practices"),
        ("500-word", "blog post", "the rise of no-code tools"),
        ("1000-word", "blog post", "developer experience trends"),
        ("500-word", "blog post", "the gig economy's impact on workers"),
        ("500-word", "blog post", "cybersecurity for small businesses"),
        ("500-word", "blog post", "the importance of documentation culture"),
        ("500-word", "blog post", "lessons from failed startups"),
        ("500-word", "blog post", "the ethics of AI in hiring"),
        ("500-word", "feature article", "platform engineering trends"),
        ("500-word", "feature article", "the evolution of cloud infrastructure"),
        ("500-word", "feature article", "the impact of generative AI on creative work"),
        ("500-word", "feature article", "why open-source tooling wins in the long run"),
        ("500-word", "feature article", "the future of developer productivity"),
    ]
    _crm_comprehensive = [  # "Write a comprehensive blog/cover letter/etc." → QUALITY_ADJ + IMP_WRITE
        ("blog post", "the importance of psychological safety in engineering teams"),
        ("blog post", "how observability transforms production debugging"),
        ("blog post", "the case for platform engineering over traditional DevOps"),
        ("feature article", "emerging trends in distributed system design"),
        ("newsletter", "monthly insights on product-led growth strategies"),
        ("press release", "the launch of a new developer productivity platform"),
        ("cover letter", "a senior software engineer applying to a distributed systems team"),
        ("cover letter", "a product manager applying to a consumer growth startup"),
        ("LinkedIn post", "sharing insights from an engineering leadership retreat"),
        ("announcement email", "a major product feature update"),
        ("README", "a new open-source data transformation library"),
        ("FAQ", "a self-hosted observability platform"),
        ("onboarding guide", "a new backend engineering team"),
        ("persona", "a typical enterprise DevOps engineer"),
        ("product description", "a next-generation observability SaaS platform"),
    ]
    creative_m = (
        [C(f"Write a {wc} {fmt} about {topic}.", D.creative, X.moderate) for wc, fmt, topic in _crm_wordcount_topics]
        + [C(f"Write a comprehensive {fmt} about {topic}.", D.creative, X.moderate) for fmt, topic in _crm_comprehensive]
        + [
            C("Create a detailed README and onboarding guide for a new open-source project.", D.creative, X.moderate),
            C("Create a detailed content calendar for a monthly newsletter about engineering leadership.", D.creative, X.moderate),
            C("Draft a comprehensive press release for a new AI product launch.", D.creative, X.moderate),
            C("Write a narrative postmortem for a major production outage.", D.creative, X.moderate),
            C("Create a detailed product description and marketing copy for a SaaS platform launch.", D.creative, X.moderate),
            C("Create a comprehensive pitch deck outline for a seed-stage startup.", D.creative, X.moderate),
            C("Draft a detailed cover letter for a principal engineer role at a distributed systems company.", D.creative, X.moderate),
            C("Write a comprehensive buyer persona for a mid-market enterprise software product.", D.creative, X.moderate),
            C("Create a comprehensive FAQ for a developer-facing API product.", D.creative, X.moderate),
            C("Draft a comprehensive LinkedIn post about lessons learned leading remote engineering teams.", D.creative, X.moderate),
            C("Create a detailed onboarding guide and FAQ for a new engineering team.", D.creative, X.moderate),
            C("Write a creative analogy comparing microservices to a city's transit system.", D.creative, X.moderate),
            C("Write a narrative blog post about a fictional team's journey from monolith to microservices.", D.creative, X.moderate),
            C("Draft a comprehensive press release about a major partnership announcement.", D.creative, X.moderate),
            C("Create a detailed drip campaign outline for a product-led growth funnel.", D.creative, X.moderate),
        ]
    )  # 15+15+15 = 45 ... need 15 more
    creative_m += [
        C("Write a 500-word marketing copy for a new developer tool launch.", D.creative, X.moderate),
        C("Write a comprehensive product announcement email for a major feature release.", D.creative, X.moderate),
        C("Create a detailed content calendar for a quarterly blog post series.", D.creative, X.moderate),
        C("Draft a comprehensive pitch deck for a Series A fundraising round.", D.creative, X.moderate),
        C("Write a comprehensive newsletter covering monthly engineering insights.", D.creative, X.moderate),
        C("Create a detailed README for a new developer SDK with onboarding examples.", D.creative, X.moderate),
        C("Draft a creative cover letter for a product design role at a growth-stage startup.", D.creative, X.moderate),
        C("Write a comprehensive FAQ for an enterprise SaaS migration project.", D.creative, X.moderate),
        C("Create a detailed LinkedIn post series about building high-performing remote teams.", D.creative, X.moderate),
        C("Draft a comprehensive announcement email for a product deprecation with migration steps.", D.creative, X.moderate),
        C("Write a 500-word blog post about trends in developer experience tooling.", D.creative, X.moderate),
        C("Create a detailed persona for a typical enterprise IT decision-maker.", D.creative, X.moderate),
        C("Draft a comprehensive marketing copy for a new cloud-native data platform.", D.creative, X.moderate),
        C("Write a feature article about the evolution of engineering productivity metrics.", D.creative, X.moderate),
        C("Create a comprehensive press release for a new series of developer tools.", D.creative, X.moderate),
    ]  # +15 = 60

    # creative / complex: 2+ creative keywords, score ≥ 0.55
    # IMP_WRITE(0.15)+WORD_COUNT(0.15)+SCALE_COMPLEX(0.20)+QUALITY_ADJ(0.08)+SECONDARY_REQ(0.08)=0.66
    # OR: IMP_DESIGN("design a content calendar") + SCALE_COMPLEX = 0.55
    _crc_wordcount = [  # "Write a comprehensive {N}-word {fmt} for {scale} users, including {X}."
        ("1000-word", "blog post", "100K", "marketing copy and a tagline"),
        ("2000-word", "feature article", "50K", "a detailed narrative and key insights"),
        ("1000-word", "press release", "10K", "marketing copy and a brand tagline"),
        ("1500-word", "newsletter", "100K", "a content calendar section and marketing copy"),
        ("1000-word", "product announcement", "50K", "marketing copy and a tagline"),
        ("2000-word", "pitch deck narrative", "10K", "a brand story and marketing copy"),
        ("1000-word", "cover letter and pitch", "1K", "a narrative introduction and tagline"),
        ("1500-word", "blog post series", "100K", "a content calendar and marketing copy"),
        ("1000-word", "case study", "50K", "key narrative highlights and a tagline"),
        ("2000-word", "marketing campaign brief", "10K", "marketing copy, taglines, and a content calendar"),
    ]
    _crc_design = [  # "Design a content calendar/drip campaign/narrative for {scale} users."
        ("content calendar featuring a newsletter and drip campaign", "100K users"),
        ("drip campaign and content calendar for a product launch", "50K users"),
        ("multi-channel marketing campaign with blog posts and marketing copy", "500K users"),
        ("quarterly content calendar with newsletter and press release strategy", "100K users"),
        ("brand storytelling framework with narrative blog posts and marketing copy", "50K users"),
        ("product launch campaign with pitch deck, blog post, and marketing copy", "10K users"),
        ("annual content calendar with drip campaigns and newsletter cadence", "100K users"),
        ("integrated marketing campaign combining blog posts, press releases, and taglines", "50K users"),
        ("content strategy covering newsletters, blog posts, and marketing copy", "100K users"),
        ("multi-format marketing plan with pitch deck, blog posts, and product descriptions", "10K users"),
    ]
    creative_c = (
        [C(f"Write a comprehensive {wc} {fmt} targeting {scale} users, including {inc}.", D.creative, X.complex) for wc, fmt, scale, inc in _crc_wordcount]
        + [C(f"Design a {plan} targeting {scale}.", D.creative, X.complex) for plan, scale in _crc_design]
        + [
            C("Write a comprehensive 1000-word drip campaign targeting 100K users, including marketing copy, blog posts, and a tagline.", D.creative, X.complex),
            C("Write a 2000-word comprehensive content calendar for a newsletter reaching 50K users, including marketing copy and a pitch narrative.", D.creative, X.complex),
            C("Create a comprehensive pitch deck and marketing copy for a Series B targeting 10K enterprise users, including a brand tagline.", D.creative, X.complex),
            C("Write a 1000-word comprehensive product description and press release targeting 100K users, including a tagline and narrative.", D.creative, X.complex),
            C("Design a multi-channel drip campaign with content calendar and marketing copy for 500K users.", D.creative, X.complex),
            C("Write a comprehensive 1500-word feature article targeting 100K readers, including a narrative arc and tagline.", D.creative, X.complex),
            C("Design a quarterly newsletter and content calendar campaign for 50K users, including blog posts and marketing copy.", D.creative, X.complex),
            C("Write a 1000-word comprehensive pitch deck narrative for 10K investors, including marketing copy and brand tagline.", D.creative, X.complex),
            C("Create a comprehensive 2000-word drip campaign series targeting 100K users, including blog posts, marketing copy, and a content calendar.", D.creative, X.complex),
            C("Design a creative marketing content calendar with drip campaigns and blog posts for 50K users.", D.creative, X.complex),
            C("Write a comprehensive 1000-word press release targeting 100K readers, including marketing copy and a tagline.", D.creative, X.complex),
            C("Design a multi-format content calendar featuring newsletter, blog posts, and marketing copy for 100K users.", D.creative, X.complex),
            C("Write a comprehensive 1500-word brand narrative for a startup targeting 10K early adopters, including taglines and a pitch deck.", D.creative, X.complex),
            C("Create a comprehensive quarterly content calendar with drip campaign and newsletter for 50K users, including marketing copy.", D.creative, X.complex),
            C("Design a comprehensive integrated marketing campaign with blog posts, press releases, and content calendar for 100K users.", D.creative, X.complex),
            C("Write a 2000-word comprehensive brand story and pitch deck narrative for 10K investors, including marketing copy.", D.creative, X.complex),
            C("Create a comprehensive 1000-word product launch announcement targeting 50K users, including marketing copy and a tagline.", D.creative, X.complex),
            C("Design a multi-channel blog post series and content calendar for a newsletter reaching 100K subscribers.", D.creative, X.complex),
            C("Write a 1000-word comprehensive marketing copy and feature article targeting 50K enterprise users.", D.creative, X.complex),
            C("Create a comprehensive drip campaign strategy with blog posts and press releases targeting 100K users.", D.creative, X.complex),
            C("Design a full-year content calendar with newsletter cadence and marketing copy for 500K users.", D.creative, X.complex),
            C("Write a comprehensive 1500-word product description and pitch deck for a new platform targeting 10K businesses.", D.creative, X.complex),
            C("Create a comprehensive marketing campaign with 1000-word blog posts, marketing copy, and a tagline for 100K users.", D.creative, X.complex),
            C("Design a multi-format content calendar featuring drip campaign, blog posts, and newsletters for 50K users.", D.creative, X.complex),
            C("Write a comprehensive 2000-word case study and press release targeting 10K enterprise users, including marketing copy.", D.creative, X.complex),
            C("Create a comprehensive 1000-word brand voice guide with examples, taglines, and marketing copy for 100K users.", D.creative, X.complex),
            C("Design a product launch content calendar with drip campaigns, blog posts, and press releases for 50K users.", D.creative, X.complex),
            C("Write a comprehensive 1500-word investor pitch narrative and pitch deck for a startup targeting 1K investors.", D.creative, X.complex),
            C("Create a comprehensive quarterly content calendar with blog posts, newsletters, and marketing copy for 100K subscribers.", D.creative, X.complex),
            C("Design a multi-channel content campaign with marketing copy, blog posts, and taglines for 500K users.", D.creative, X.complex),
        ]
    )  # 10+10+40... let me check: _crc_wordcount=10, _crc_design=10, extra=40 → 60? Let me count extra: 11+3+4+3+3+3+3+3+3+3+3+2+2+2+2+2 hmm, let me count the list items: I see 30 items in the last list. Total = 10+10+30 = 50. Need 10 more.
    creative_c += [
        C("Design a comprehensive content strategy with blog posts, marketing copy, and newsletter for 100K users.", D.creative, X.complex),
        C("Write a comprehensive 1000-word blog post and marketing copy series for 50K enterprise users.", D.creative, X.complex),
        C("Create a comprehensive content calendar with drip campaigns and taglines for a product targeting 100K users.", D.creative, X.complex),
        C("Design a quarterly blog post series with newsletter and press release for 50K subscribers.", D.creative, X.complex),
        C("Write a 1000-word narrative pitch deck and marketing copy for a growth-stage startup targeting 10K investors.", D.creative, X.complex),
        C("Create a comprehensive annual content calendar with blog posts, newsletters, and marketing copy for 100K users.", D.creative, X.complex),
        C("Design a brand storytelling campaign with blog posts, marketing copy, and taglines for 500K users.", D.creative, X.complex),
        C("Write a 2000-word comprehensive marketing strategy with blog posts, press releases, and a content calendar for 50K users.", D.creative, X.complex),
        C("Create a comprehensive multi-channel content calendar with drip campaigns, blog posts, and newsletters for 100K users.", D.creative, X.complex),
        C("Design a product narrative with pitch deck, marketing copy, and taglines for a startup targeting 10K enterprise customers.", D.creative, X.complex),
    ]  # +10 → need to verify total = 60
    chat_s: list[Case] = [
        # Geography (15) — WH_SIMPLE at line start → −0.08, short → −0.05 = −0.13 → simple
        C("What is the capital of France?",                        D.chat, X.simple),
        C("What is the largest country by area?",                  D.chat, X.simple),
        C("Where is the Amazon River located?",                    D.chat, X.simple),
        C("Which ocean is the largest on Earth?",                  D.chat, X.simple),
        C("What is the tallest mountain in the world?",            D.chat, X.simple),
        C("Where is Mount Everest?",                               D.chat, X.simple),
        C("What country is Rome the capital of?",                  D.chat, X.simple),
        C("How many continents are there?",                        D.chat, X.simple),
        C("What is the smallest country in the world?",            D.chat, X.simple),
        C("Which country has the largest population?",             D.chat, X.simple),
        C("What language is spoken in Brazil?",                    D.chat, X.simple),
        C("Where is the Sahara Desert?",                           D.chat, X.simple),
        C("What is the capital of Australia?",                     D.chat, X.simple),
        C("How long is the Great Wall of China?",                  D.chat, X.simple),
        C("Which city is the Eiffel Tower in?",                    D.chat, X.simple),
        # History (15)
        C("When did World War II end?",                            D.chat, X.simple),
        C("Who was the first US president?",                       D.chat, X.simple),
        C("When was the Declaration of Independence signed?",      D.chat, X.simple),
        C("What year did the Berlin Wall fall?",                   D.chat, X.simple),
        C("Who was Napoleon Bonaparte?",                           D.chat, X.simple),
        C("When did the Roman Empire fall?",                       D.chat, X.simple),
        C("What century was the Renaissance?",                     D.chat, X.simple),
        C("Who discovered penicillin?",                            D.chat, X.simple),
        C("When did the Apollo 11 moon landing happen?",           D.chat, X.simple),
        C("What was the name of the first artificial satellite?",  D.chat, X.simple),
        C("Who painted the Mona Lisa?",                            D.chat, X.simple),
        C("When was the Great Fire of London?",                    D.chat, X.simple),
        C("Which century was Shakespeare born in?",                D.chat, X.simple),
        C("Who built the pyramids of Giza?",                       D.chat, X.simple),
        C("What year did the Titanic sink?",                       D.chat, X.simple),
        # Science / nature (15)
        C("How many planets are in the solar system?",             D.chat, X.simple),
        C("What is the closest star to Earth?",                    D.chat, X.simple),
        C("How long does light take to reach Earth from the sun?", D.chat, X.simple),
        C("What is the chemical symbol for gold?",                 D.chat, X.simple),
        C("How many bones are in the human body?",                 D.chat, X.simple),
        C("What is the speed of light?",                           D.chat, X.simple),
        C("What is water made of?",                                D.chat, X.simple),
        C("How hot is the surface of the sun?",                    D.chat, X.simple),
        C("What is the hardest natural substance on Earth?",       D.chat, X.simple),
        C("How many days are in one Earth year?",                  D.chat, X.simple),
        C("What is the largest planet in the solar system?",       D.chat, X.simple),
        C("Which gas do plants absorb from the air?",              D.chat, X.simple),
        C("What is the boiling point of water?",                   D.chat, X.simple),
        C("How many teeth does an adult human have?",              D.chat, X.simple),
        C("What is the most abundant gas in Earth's atmosphere?",  D.chat, X.simple),
        # Sports / entertainment (15)
        C("Which country won the first FIFA World Cup?",           D.chat, X.simple),
        C("How many players are on a basketball team?",            D.chat, X.simple),
        C("What sport is played at Wimbledon?",                    D.chat, X.simple),
        C("Who holds the record for most Olympic gold medals?",    D.chat, X.simple),
        C("How long is a marathon in kilometers?",                 D.chat, X.simple),
        C("What year were the first modern Olympics held?",        D.chat, X.simple),
        C("How many strings does a standard guitar have?",         D.chat, X.simple),
        C("What country does flamenco dancing come from?",         D.chat, X.simple),
        C("Who directed the movie Titanic?",                       D.chat, X.simple),
        C("How many cards are in a standard deck?",                D.chat, X.simple),
        C("What color is a ripe banana?",                          D.chat, X.simple),
        C("How many sides does a hexagon have?",                   D.chat, X.simple),
        C("What is the national bird of the USA?",                 D.chat, X.simple),
        C("What are the colors of the French flag?",               D.chat, X.simple),
        C("What is the longest-running Broadway show?",            D.chat, X.simple),
        # Food / culture / misc (15)
        C("What is sushi made of?",                                D.chat, X.simple),
        C("Which country is pizza originally from?",               D.chat, X.simple),
        C("How many days are in a leap year?",                     D.chat, X.simple),
        C("What time zone is New York City in?",                   D.chat, X.simple),
        C("What is the currency of Japan?",                        D.chat, X.simple),
        C("What is the national animal of Australia?",             D.chat, X.simple),
        C("How many hours are in a week?",                         D.chat, X.simple),
        C("What is the freezing point of water in Fahrenheit?",    D.chat, X.simple),
        C("Which animal is the fastest on land?",                  D.chat, X.simple),
        C("What is the average temperature of the human body?",    D.chat, X.simple),
        C("How long does it take for the Moon to orbit Earth?",    D.chat, X.simple),
        C("How many zeros are in one million?",                    D.chat, X.simple),
        C("What is the largest mammal in the world?",              D.chat, X.simple),
        C("What is the currency of the United Kingdom?",           D.chat, X.simple),
        C("How many legs does a spider have?",                     D.chat, X.simple),
    ]  # 75
    chat_m: list[Case] = [
        # Natural science — "explain" anywhere → WH_EXPLAIN +0.20, short → −0.05 = 0.15 → moderate
        # No "explain why" (reasoning domain keyword) — only "explain [what/how/...]"
        C("Can you explain how photosynthesis works?",             D.chat, X.moderate),
        C("Explain what the water cycle is.",                      D.chat, X.moderate),
        C("Please explain how rainbows are formed.",               D.chat, X.moderate),
        C("Please explain what causes the red sky at sunset.",     D.chat, X.moderate),
        C("Explain how tides work.",                               D.chat, X.moderate),
        C("Can you explain what causes earthquakes?",              D.chat, X.moderate),
        C("Explain the seasons to me.",                            D.chat, X.moderate),
        C("Please explain how volcanoes erupt.",                   D.chat, X.moderate),
        C("Can you explain what black holes are?",                 D.chat, X.moderate),
        C("Explain how the immune system works.",                  D.chat, X.moderate),
        C("Can you explain what DNA is?",                          D.chat, X.moderate),
        C("Please explain how the phases of the moon work.",       D.chat, X.moderate),
        C("Please explain how gravity works.",                     D.chat, X.moderate),
        C("Can you explain what causes thunder?",                  D.chat, X.moderate),
        C("Explain how the human digestive system works.",         D.chat, X.moderate),
        C("Can you explain what atoms are made of?",               D.chat, X.moderate),
        C("Explain how solar eclipses happen.",                    D.chat, X.moderate),
        C("Can you explain what causes the northern lights?",      D.chat, X.moderate),
        C("Explain how birds navigate during migration.",          D.chat, X.moderate),
        C("Please explain how the human brain stores memories.",   D.chat, X.moderate),
        # Social / historical / general concepts (20)
        C("Can you explain how inflation works?",                  D.chat, X.moderate),
        C("Explain what democracy means.",                         D.chat, X.moderate),
        C("Can you explain how stock markets work?",               D.chat, X.moderate),
        C("Explain how insurance works.",                          D.chat, X.moderate),
        C("Can you explain what compound interest is?",            D.chat, X.moderate),
        C("Explain how credit scores work.",                       D.chat, X.moderate),
        C("Can you explain what a constitution is?",               D.chat, X.moderate),
        C("Explain how elections work in a democracy.",            D.chat, X.moderate),
        C("Can you explain what the Industrial Revolution was?",   D.chat, X.moderate),
        C("Explain how the Roman Empire rose to power.",           D.chat, X.moderate),
        C("Can you explain what caused the Great Depression?",     D.chat, X.moderate),
        C("Explain how the Cold War started.",                     D.chat, X.moderate),
        C("Can you explain how vaccines work?",                    D.chat, X.moderate),
        C("Explain what a recession is.",                          D.chat, X.moderate),
        C("Can you explain how carbon dating works?",              D.chat, X.moderate),
        C("Explain what time zones are and how they work.",        D.chat, X.moderate),
        C("Can you explain how jet lag affects the body?",         D.chat, X.moderate),
        C("Explain how bread rises when you bake it.",             D.chat, X.moderate),
        C("Can you explain how fermentation works?",               D.chat, X.moderate),
        C("Explain what happens during sleep.",                    D.chat, X.moderate),
        # Everyday concepts (20)
        C("Can you explain how Wi-Fi works?",                      D.chat, X.moderate),
        C("Explain how a microwave heats food.",                   D.chat, X.moderate),
        C("Can you explain how GPS navigation works?",             D.chat, X.moderate),
        C("Explain how solar panels generate power.",              D.chat, X.moderate),
        C("Can you explain how batteries store energy?",           D.chat, X.moderate),
        C("Explain how the internet transmits data.",              D.chat, X.moderate),
        C("Can you explain how email is delivered?",               D.chat, X.moderate),
        C("Explain how a refrigerator keeps food cold.",           D.chat, X.moderate),
        C("Can you explain how airplanes stay in the air?",        D.chat, X.moderate),
        C("Explain how car engines work.",                         D.chat, X.moderate),
        C("Can you explain how cameras capture images?",           D.chat, X.moderate),
        C("Explain how touchscreens detect touch.",                D.chat, X.moderate),
        C("Can you explain how Bluetooth works?",                  D.chat, X.moderate),
        C("Explain how noise-canceling headphones work.",          D.chat, X.moderate),
        C("Can you explain how QR codes work?",                    D.chat, X.moderate),
        C("Explain how 3D printing works.",                        D.chat, X.moderate),
        C("Can you explain how electric cars work?",               D.chat, X.moderate),
        C("Explain what renewable energy is.",                     D.chat, X.moderate),
        C("Can you explain how wind turbines generate electricity?", D.chat, X.moderate),
        C("Explain how nuclear power plants generate energy.",     D.chat, X.moderate),
        # Food / health / misc (15)
        C("Can you explain how coffee affects the body?",          D.chat, X.moderate),
        C("Please explain what happens to our body during sleep.", D.chat, X.moderate),
        C("Can you explain how exercise builds muscle?",           D.chat, X.moderate),
        C("Explain what calories are.",                            D.chat, X.moderate),
        C("Can you explain what makes spicy food feel so hot?",    D.chat, X.moderate),
        C("Explain how the sense of smell works.",                 D.chat, X.moderate),
        C("Can you explain what causes hiccups?",                  D.chat, X.moderate),
        C("Explain how vision works.",                             D.chat, X.moderate),
        C("Can you explain what jet streams are?",                 D.chat, X.moderate),
        C("Explain what makes music sound pleasant to us.",        D.chat, X.moderate),
        C("Can you explain how languages evolve over time?",       D.chat, X.moderate),
        C("Explain what dreams are.",                              D.chat, X.moderate),
        C("Can you explain what causes aging?",                    D.chat, X.moderate),
        C("Explain how the stock market reflects economic health.", D.chat, X.moderate),
        C("Can you explain how maps are made?",                    D.chat, X.moderate),
    ]  # 75
    critical_x: list[Case] = [
        # Medical (15) — "diagnose"/"diagnosis"/"medical advice"/"prescribe" → critical veto
        C("Can you diagnose what is wrong with me based on these symptoms?",    D.chat, X.critical),
        C("I have chest pain and shortness of breath, can you diagnose this?",  D.chat, X.critical),
        C("What is the diagnosis for recurring migraines with aura?",           D.chat, X.critical),
        C("Should I prescribe ibuprofen or acetaminophen for my patient?",      D.chat, X.critical),
        C("Can you give me medical advice on managing type 2 diabetes?",        D.chat, X.critical),
        C("Provide medical advice on the best treatment for hypertension.",     D.chat, X.critical),
        C("What medication should I prescribe for a patient with anxiety?",     D.chat, X.critical),
        C("Is it safe to self-diagnose based on online symptoms?",              D.chat, X.critical),
        C("Can you help me diagnose a child with recurring fever?",             D.chat, X.critical),
        C("What does this diagnosis mean for my long-term health?",             D.chat, X.critical),
        C("I need medical advice on whether to have this surgery.",             D.chat, X.critical),
        C("Please prescribe something for my insomnia.",                        D.chat, X.critical),
        C("Give me medical advice on managing my arthritis pain.",              D.chat, X.critical),
        C("Can you diagnose whether this mole looks cancerous?",                D.chat, X.critical),
        C("What medical advice do doctors give for chronic back pain?",         D.chat, X.critical),
        # Legal (10)
        C("Give me legal advice on whether I can sue my landlord.",             D.chat, X.critical),
        C("I need legal advice about my employment contract rights.",            D.chat, X.critical),
        C("Can you provide legal advice on this intellectual property dispute?", D.chat, X.critical),
        C("I need legal advice on whether this contract is enforceable.",        D.chat, X.critical),
        C("Explain the legal liability I face if my app collects user data.",    D.chat, X.critical),
        C("What is my legal liability if an employee is injured at my business?", D.chat, X.critical),
        C("I need legal advice on how to respond to a cease and desist letter.", D.chat, X.critical),
        C("Does my company face legal liability for this data breach?",          D.chat, X.critical),
        C("What legal liability do I have as a director if the company fails?",  D.chat, X.critical),
        C("Provide legal advice on whether this NDA is legally binding.",        D.chat, X.critical),
        # Compliance / regulatory (10)
        C("Is our data handling policy compliant with GDPR regulatory requirements?", D.chat, X.critical),
        C("What are the regulatory requirements for launching a fintech product in the EU?", D.chat, X.critical),
        C("How do we ensure compliance with SOX regulatory standards?",          D.chat, X.critical),
        C("Our system must meet compliance standards for PCI-DSS — where do we start?", D.chat, X.critical),
        C("What compliance checks are required before launching a medical device?", D.chat, X.critical),
        C("Are we exposed to regulatory fines under this new data protection rule?", D.chat, X.critical),
        C("How do we achieve compliance with AML regulatory requirements?",      D.chat, X.critical),
        C("We need to pass a compliance audit — what regulatory documents are needed?", D.chat, X.critical),
        C("What are the compliance and regulatory steps to operate as a payment processor?", D.chat, X.critical),
        C("How do I ensure compliance with HIPAA regulatory requirements?",      D.chat, X.critical),
        # Financial risk / investment (10)
        C("What is my financial risk if I invest all savings in one stock?",     D.chat, X.critical),
        C("Give me investment advice on whether to buy Bitcoin now.",            D.chat, X.critical),
        C("I need investment advice on building a retirement portfolio.",        D.chat, X.critical),
        C("What is the financial risk of taking a second mortgage?",             D.chat, X.critical),
        C("Provide investment advice on hedging against inflation.",             D.chat, X.critical),
        C("What is the credit risk of lending to this startup?",                 D.chat, X.critical),
        C("Can you assess the credit risk of this loan application?",            D.chat, X.critical),
        C("What is the financial risk exposure from this derivatives position?", D.chat, X.critical),
        C("Give me investment advice on real estate versus equities.",           D.chat, X.critical),
        C("What is the credit risk of this corporate bond portfolio?",           D.chat, X.critical),
        # Safety-critical / life-critical (10)
        C("This is a life-critical system — what standards apply?",              D.chat, X.critical),
        C("We are building a life-critical pacemaker firmware — what standards apply?", D.chat, X.critical),
        C("Our autonomous vehicle is safety-critical — who sets the standards?", D.chat, X.critical),
        C("How do we test a safety-critical aviation system?",                   D.chat, X.critical),
        C("Is this a life or death situation requiring immediate attention?",    D.chat, X.critical),
        C("This is life or death — should I call emergency services right now?", D.chat, X.critical),
        C("We are developing safety-critical industrial control systems.",       D.chat, X.critical),
        C("What certification do life-critical medical devices need?",           D.chat, X.critical),
        C("How do we validate safety-critical software in aerospace?",           D.chat, X.critical),
        C("Our platform handles life-critical alerts for hospital emergency rooms.", D.chat, X.critical),
    ]  # 55
    ambiguous_x: list[Case] = [
        # 1-5 word vague messages — no domain keywords, score ≈ 0 → simple, domain = chat
        C("Help.",            D.chat, X.simple), C("Go on.",          D.chat, X.simple),
        C("What?",            D.chat, X.simple), C("Hmm.",            D.chat, X.simple),
        C("Ok.",              D.chat, X.simple), C("Sure.",           D.chat, X.simple),
        C("Yes?",             D.chat, X.simple), C("No.",             D.chat, X.simple),
        C("Really?",          D.chat, X.simple), C("Interesting.",    D.chat, X.simple),
        C("Thanks.",          D.chat, X.simple), C("Cool.",           D.chat, X.simple),
        C("Got it.",          D.chat, X.simple), C("I see.",          D.chat, X.simple),
        C("What now?",        D.chat, X.simple), C("Next step?",      D.chat, X.simple),
        C("Continue.",        D.chat, X.simple), C("More please.",    D.chat, X.simple),
        C("Tell me more.",    D.chat, X.simple), C("And?",            D.chat, X.simple),
        C("So?",              D.chat, X.simple), C("Then what?",      D.chat, X.simple),
        C("How so?",          D.chat, X.simple), C("Why?",            D.chat, X.simple),
        C("When?",            D.chat, X.simple), C("Who?",            D.chat, X.simple),
        C("Where?",           D.chat, X.simple), C("Which?",          D.chat, X.simple),
        C("Huh?",             D.chat, X.simple), C("Right?",          D.chat, X.simple),
        C("Is that so?",      D.chat, X.simple), C("Makes sense.",    D.chat, X.simple),
        C("Alright.",         D.chat, X.simple), C("Understood.",     D.chat, X.simple),
        C("Done.",            D.chat, X.simple), C("Perfect.",        D.chat, X.simple),
        C("Great.",           D.chat, X.simple), C("Noted.",          D.chat, X.simple),
        C("Agreed.",          D.chat, X.simple), C("Fine.",           D.chat, X.simple),
        C("OK got it.",       D.chat, X.simple), C("Sounds good.",    D.chat, X.simple),
        C("Maybe.",           D.chat, X.simple), C("Probably.",       D.chat, X.simple),
        C("I think so.",      D.chat, X.simple), C("I don't know.",   D.chat, X.simple),
        C("Kind of.",         D.chat, X.simple), C("I'm not sure.",   D.chat, X.simple),
        C("Let me think.",    D.chat, X.simple), C("I'm confused.",   D.chat, X.simple),
        C("I'm lost.",        D.chat, X.simple), C("Help me.",        D.chat, X.simple),
        C("Need help.",       D.chat, X.simple), C("Assist me.",      D.chat, X.simple),
        C("Can you help?",    D.chat, X.simple), C("Please help.",    D.chat, X.simple),
        C("What do I do?",    D.chat, X.simple), C("What's next?",    D.chat, X.simple),
        C("What's wrong?",    D.chat, X.simple), C("What happened?",  D.chat, X.simple),
        C("Tell me something.", D.chat, X.simple), C("Say something.", D.chat, X.simple),
        C("Anything?",        D.chat, X.simple), C("Something else?", D.chat, X.simple),
        C("What else?",       D.chat, X.simple), C("More.",           D.chat, X.simple),
        C("Less.",            D.chat, X.simple), C("Better.",         D.chat, X.simple),
        C("Faster.",          D.chat, X.simple), C("Slower.",         D.chat, X.simple),
        C("Bigger.",          D.chat, X.simple), C("Smaller.",        D.chat, X.simple),
        C("Simpler.",         D.chat, X.simple), C("Different.",      D.chat, X.simple),
        C("Again.",           D.chat, X.simple), C("Once more.",      D.chat, X.simple),
        C("Never mind.",      D.chat, X.simple), C("Skip it.",        D.chat, X.simple),
        C("Move on.",         D.chat, X.simple), C("Start over.",     D.chat, X.simple),
        C("Reset.",           D.chat, X.simple), C("Undo.",           D.chat, X.simple),
        C("Redo.",            D.chat, X.simple), C("Stop.",           D.chat, X.simple),
        C("Pause.",           D.chat, X.simple), C("Wait.",           D.chat, X.simple),
        C("Hold on.",         D.chat, X.simple), C("One moment.",     D.chat, X.simple),
        C("Coming.",          D.chat, X.simple), C("Ready.",          D.chat, X.simple),
        C("Here.",            D.chat, X.simple), C("There.",          D.chat, X.simple),
        C("Now.",             D.chat, X.simple), C("Later.",          D.chat, X.simple),
        C("Soon.",            D.chat, X.simple), C("Today.",          D.chat, X.simple),
        C("Tomorrow.",        D.chat, X.simple), C("Okay.",           D.chat, X.simple),
        C("Fair enough.",     D.chat, X.simple), C("Got that.",       D.chat, X.simple),
        C("All good.",        D.chat, X.simple), C("No idea.",        D.chat, X.simple),
    ]  # 100
    privacy_x: list[Case] = [
        # SSN → confidential (15) — valid prefixes (not 000/666/9xx), standalone 9-digit
        C("My SSN is 123-45-6789, please help me fill out this tax form.",      D.chat, X.simple, P.confidential),
        C("Social security number 234-56-7890 was used to verify identity.",    D.chat, X.simple, P.confidential),
        C("The account holder SSN is 345-67-8901 per the application.",         D.chat, X.simple, P.confidential),
        C("Please verify that SSN 456-78-9012 matches our records.",            D.chat, X.simple, P.confidential),
        C("I found SSN 567-89-0123 in the audit log.",                          D.chat, X.simple, P.confidential),
        C("Customer SSN: 111-22-3333. Name: John Smith.",                       D.chat, X.simple, P.confidential),
        C("File a return using SSN 222-33-4444.",                               D.chat, X.simple, P.confidential),
        C("My social security 333-44-5555 is linked to my pension.",            D.chat, X.simple, P.confidential),
        C("Applicant SSN 444-55-6666 failed the background check.",             D.chat, X.simple, P.confidential),
        C("The beneficiary SSN is 555-66-7777.",                                D.chat, X.simple, P.confidential),
        C("SSN 124-35-6789 was disclosed without consent.",                     D.chat, X.simple, P.confidential),
        C("Use my SSN 135-46-7890 to access the pension portal.",               D.chat, X.simple, P.confidential),
        C("My SSN is 246-57-8901 — is it safe to give to my employer?",         D.chat, X.simple, P.confidential),
        C("Patient SSN 357-68-9012 is in the intake form.",                     D.chat, X.simple, P.confidential),
        C("Enrollee 468-79-0123 needs to update their profile.",                D.chat, X.simple, P.confidential),
        # Credit card → confidential (8) — Luhn-valid test card numbers
        C("My card number is 4532015112830366 and it expires 04/27.",           D.chat, X.simple, P.confidential),
        C("Please charge card 4916338506082832 for this order.",                D.chat, X.simple, P.confidential),
        C("The billing card on file is 5425233430109903.",                      D.chat, X.simple, P.confidential),
        C("Card 5500005555555559 was declined — please retry.",                 D.chat, X.simple, P.confidential),
        C("Refund to credit card 4111111111111111.",                            D.chat, X.simple, P.confidential),
        C("My card 5555555555554444 was charged twice.",                        D.chat, X.simple, P.confidential),
        C("Please update my billing card to 4929420498610201.",                 D.chat, X.simple, P.confidential),
        C("Card 5136333468010119 has an expired billing address.",              D.chat, X.simple, P.confidential),
        # AWS key → confidential (5)
        C("The AWS key AKIAIOSFODNN7EXAMPLE was committed to the repository.",  D.chat, X.simple, P.confidential),
        C("Found AWS access key AKIAXYZ123ABCDEF56GH in the config file.",      D.chat, X.simple, P.confidential),
        C("My AWS credential AKIAM7O8BFAD2C3E4FG5 needs rotation.",             D.chat, X.simple, P.confidential),
        C("Revoke this leaked key: AKIAJ3WS2BFOCM12ABCD.",                     D.chat, X.simple, P.confidential),
        C("AWS key AKIAR3WX4YZ5ABC678DE should not be in source control.",      D.chat, X.simple, P.confidential),
        # GitHub token → confidential (5)
        C("Leaked GitHub token: ghp_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ.",    D.chat, X.simple, P.confidential),
        C("The token gho_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij was rotated.",   D.chat, X.simple, P.confidential),
        C("Revoke ghu_1234567890abcdefghijklmnopqrstuvwxyz0 immediately.",      D.chat, X.simple, P.confidential),
        C("This PR contains ghs_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789X.",        D.chat, X.simple, P.confidential),
        C("Found ghr_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789 in the logs.",       D.chat, X.simple, P.confidential),
        # API key / secret → confidential (2)
        C("Config: api_key=sk_live_abc123456789xyz",                            D.chat, X.simple, P.confidential),
        C("The env has password=MySecretPass123! stored in plaintext.",         D.chat, X.simple, P.confidential),
        # Medical → internal (12) — avoid "diagnose"/"prescribe"/"medical advice" (those are critical)
        C("The patient records show elevated glucose levels.",                  D.chat, X.simple, P.internal),
        C("PHI was inadvertently shared with an unauthorized vendor.",          D.chat, X.simple, P.internal),
        C("HIPAA requires that all patient data be encrypted at rest.",         D.chat, X.simple, P.internal),
        C("The medical record was updated after the last clinical visit.",      D.chat, X.simple, P.internal),
        C("Patient dosage was adjusted following the latest bloodwork.",        D.chat, X.simple, P.internal),
        C("The treatment plan includes weekly clinical assessments.",           D.chat, X.simple, P.internal),
        C("Clinical notes from the last session are in the patient file.",      D.chat, X.simple, P.internal),
        C("The patient's prescription needs to be renewed next month.",         D.chat, X.simple, P.internal),
        C("PHI data must be de-identified before sharing for research.",        D.chat, X.simple, P.internal),
        C("The clinical trial monitors patient dosage response over 12 weeks.", D.chat, X.simple, P.internal),
        C("Review the medical record before issuing a new treatment plan.",     D.chat, X.simple, P.internal),
        C("All patient information is subject to HIPAA at this facility.",      D.chat, X.simple, P.internal),
        # Legal → internal (8)
        C("The attorney-client privilege protects this memo from disclosure.",  D.chat, X.simple, P.internal),
        C("This document is privileged and must not be shared externally.",     D.chat, X.simple, P.internal),
        C("The filing is under seal and requires a court order to access.",     D.chat, X.simple, P.internal),
        C("The confidential settlement is under seal until the hearing.",       D.chat, X.simple, P.internal),
        C("The NDA prevents either party from disclosing the deal terms.",      D.chat, X.simple, P.internal),
        C("Our trade secret is protected by an NDA signed by all employees.",   D.chat, X.simple, P.internal),
        C("This trade secret must not leave the building.",                     D.chat, X.simple, P.internal),
        C("This attorney-client communication is privileged.",                  D.chat, X.simple, P.internal),
        # Financial → internal (5)
        C("Please send the wire transfer to account number 1234567890.",        D.chat, X.simple, P.internal),
        C("My W-2 shows income from three employers last year.",                D.chat, X.simple, P.internal),
        C("The 1099 form needs to be sent to all contractors by January.",      D.chat, X.simple, P.internal),
        C("Verify the routing number and account number before finalizing payroll.", D.chat, X.simple, P.internal),
        C("The bank statement shows a suspicious wire transfer last Tuesday.",  D.chat, X.simple, P.internal),
        # Plain messages → internal by default (15) — no PII or sensitive keywords
        C("What is the best way to greet someone in Japan?",                    D.chat, X.simple, P.internal),
        C("How do I make pasta from scratch?",                                  D.chat, X.simple, P.internal),
        C("What time does the New York Stock Exchange open?",                   D.chat, X.simple, P.internal),
        C("What is the population of Tokyo?",                                   D.chat, X.simple, P.internal),
        C("Can you recommend a good book to read?",                             D.chat, X.simple, P.internal),
        C("What is the average rainfall in Seattle?",                           D.chat, X.simple, P.internal),
        C("How do I grow tomatoes at home?",                                    D.chat, X.simple, P.internal),
        C("What is the tallest building in the world?",                         D.chat, X.simple, P.internal),
        C("How do I fix a leaky faucet?",                                       D.chat, X.simple, P.internal),
        C("What is the best time to visit Paris?",                              D.chat, X.simple, P.internal),
        C("How many steps should I walk per day?",                              D.chat, X.simple, P.internal),
        C("What is the lifespan of a golden retriever?",                        D.chat, X.simple, P.internal),
        C("What fruits are high in vitamin C?",                                 D.chat, X.simple, P.internal),
        C("What sets a latte apart from a cappuccino?",                         D.chat, X.simple, P.internal),
        C("How do I remove a stripped screw?",                                  D.chat, X.simple, P.internal),
    ]  # 75

    return (
        code_s + code_m + code_c
        + reasoning_s + reasoning_m + reasoning_c
        + extraction_s + extraction_m + extraction_c
        + classification_s + classification_m + classification_c
        + summarization_s + summarization_m + summarization_c
        + creative_s + creative_m + creative_c
        + chat_s + chat_m
        + critical_x + ambiguous_x + privacy_x
    )


# ---------------------------------------------------------------------------
# Scoring / reporting
# ---------------------------------------------------------------------------

def score_and_report(cases: list[Case], use_t2: bool, verbose: bool) -> bool:
    n = len(cases)
    ci = 1.96 * math.sqrt(0.25 / n)

    dom_ok   = cmplx_ok = priv_ok = 0
    esc_t2   = esc_llm  = 0
    lat_t1: list[float] = []
    lat_t2: list[float] = []

    dom_cm:   dict[tuple[str, str], int] = defaultdict(int)
    cmplx_cm: dict[tuple[str, str], int] = defaultdict(int)
    dom_total: dict[str, int] = defaultdict(int)
    dom_right: dict[str, int] = defaultdict(int)

    for case in cases:
        t0  = time.perf_counter()
        res = classify_t1(case.text)
        lat_t1.append((time.perf_counter() - t0) * 1000)

        if use_t2 and res.tier == "needs_embedding":
            esc_t2 += 1
            t0  = time.perf_counter()
            res = classify_t2(case.text, res)
            lat_t2.append((time.perf_counter() - t0) * 1000)

        if res.tier == "needs_llm":
            esc_llm += 1

        d_ok = res.domain     == case.expected_domain
        c_ok = res.complexity == case.expected_cmplx
        p_ok = res.privacy    == case.expected_privacy

        dom_ok   += int(d_ok)
        cmplx_ok += int(c_ok)
        priv_ok  += int(p_ok)

        dom_total[case.expected_domain.value]  += 1
        dom_right[case.expected_domain.value]  += int(d_ok)
        dom_cm[(case.expected_domain.value, res.domain.value)]    += 1
        cmplx_cm[(case.expected_cmplx.value,   res.complexity.value)] += 1

        if verbose and (not d_ok or not c_ok):
            tag = "[OK]" if (d_ok and c_ok) else "[NG]"
            print(
                f"  {tag} dom:{case.expected_domain.value[:4]}/{res.domain.value[:4]}"
                f"  cmp:{case.expected_cmplx.value[:4]}/{res.complexity.value[:4]}"
                f"  cf={res.domain_conf:.2f}  t={res.tier[:4]}"
                f"  | {case.text[:68]}"
            )

    sep = "-" * 64
    print(f"\n{sep}")
    print("  Tidus v1.2.0 POC -- Auto-Classifier Validation")
    print(f"  Dataset   : {n} labeled test cases")
    print(f"  Tier 2    : {'enabled' if use_t2 else 'disabled (--no-embedding)'}")
    print(f"  95%% CI    : +/-{ci*100:.1f}%%")
    print(sep)

    print("\n  Accuracy")
    print(f"    Domain     : {dom_ok:>4}/{n}  ({dom_ok/n*100:.1f}%)")
    print(f"    Complexity : {cmplx_ok:>4}/{n}  ({cmplx_ok/n*100:.1f}%)")
    print(f"    Privacy    : {priv_ok:>4}/{n}  ({priv_ok/n*100:.1f}%)")

    t1_only = n - esc_t2
    print("\n  Tier routing")
    print(f"    Tier 1 heuristic  : {t1_only:>4}/{n}  ({t1_only/n*100:.1f}%)")
    print(f"    Tier 2 embedding  : {esc_t2:>4}/{n}  ({esc_t2/n*100:.1f}%)")
    print(f"    Tier 3 LLM needed : {esc_llm:>4}/{n}  ({esc_llm/n*100:.1f}%)")

    print("\n  Per-domain accuracy")
    bar_w = 20
    for d in Domain:
        tot = dom_total.get(d.value, 0)
        ok  = dom_right.get(d.value, 0)
        pct = ok / tot * 100 if tot else 0.0
        bar = "#" * int(pct / 100 * bar_w)
        print(f"    {d.value:>14} : {ok:>3}/{tot:<3}  ({pct:5.1f}%)  {bar}")

    print("\n  Domain confusion (rows=actual, cols=predicted)")
    doms = [d.value for d in Domain]
    hdr  = "              " + "  ".join(f"{v[:5]:>5}" for v in doms)
    print(f"  {hdr}")
    for act in doms:
        row = f"  {act:>13}  "
        for prd in doms:
            c = dom_cm.get((act, prd), 0)
            row += f" {'['+str(c)+']':>5}" if c else "     ."
        print(row)

    print("\n  Complexity confusion (rows=actual, cols=predicted)")
    cmxs = [c.value for c in Complexity]
    hdr2 = "          " + "  ".join(f"{v[:5]:>6}" for v in cmxs)
    print(f"  {hdr2}")
    for act in cmxs:
        row = f"  {act:>9}  "
        for prd in cmxs:
            c = cmplx_cm.get((act, prd), 0)
            row += f" {'['+str(c)+']':>6}" if c else "      ."
        print(row)

    if lat_t1:
        p95 = sorted(lat_t1)[int(len(lat_t1) * 0.95)]
        avg = sum(lat_t1) / len(lat_t1)
        print(f"\n  Latency -- Tier 1: avg={avg:.3f}ms  p95={p95:.3f}ms")
    if lat_t2:
        p95 = sorted(lat_t2)[int(len(lat_t2) * 0.95)]
        avg = sum(lat_t2) / len(lat_t2)
        print(f"  Latency -- Tier 2: avg={avg:.1f}ms   p95={p95:.1f}ms")

    print(f"\n{sep}")
    print("  Success criteria")
    p95_t1 = sorted(lat_t1)[int(len(lat_t1) * 0.95)] if lat_t1 else 0.0
    checks = [
        ("Domain accuracy >= 70%",         dom_ok   / n >= 0.70),
        ("Complexity accuracy >= 72%",     cmplx_ok / n >= 0.72),
        ("Tier 2 escalations <= 30%",      esc_t2   / n <= 0.30),
        ("LLM escalations <= 10%",         esc_llm  / n <= 0.10),
        ("Tier 1 p95 latency < 2ms",       p95_t1   < 2.0),
    ]
    passed = all(ok for _, ok in checks)
    for label, ok in checks:
        tag = "[PASS]" if ok else "[FAIL]"
        print(f"    {tag} {label}")

    print()
    verdict = "POC PASSED -- proceed with Tidus integration." if passed else "POC FAILED -- tune heuristics before integrating."
    print(f"  {verdict}\n")
    return passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Tidus auto-classifier POC")
    parser.add_argument("--no-embedding", action="store_true", help="Tier 1 only")
    parser.add_argument("--verbose",      action="store_true", help="Print failures")
    args = parser.parse_args()

    use_t2 = not args.no_embedding
    if use_t2:
        print("Loading all-MiniLM-L6-v2 ...")
        if not load_embeddings():
            print("  sentence-transformers not installed -- running Tier 1 only.")
            print("  Install with: uv add sentence-transformers")
            use_t2 = False
        else:
            print("  Model ready.\n")

    cases = _build_cases()
    print(f"Dataset: {len(cases)} test cases\n")

    passed = score_and_report(cases, use_t2=use_t2, verbose=args.verbose)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
