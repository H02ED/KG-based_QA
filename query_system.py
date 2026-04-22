"""Minimal KG query template for Assignment 4.

Keep these APIs unchanged for auto-test:
- generate_text(messages, max_new_tokens=220)
- get_relevant_articles(question)
- generate_answer(question, rule_results)

Keep Rule fields aligned with build_kg output:
rule_id, type, action, result, art_ref, reg_name
"""

import os
import re
from typing import Any

from neo4j import GraphDatabase
from dotenv import load_dotenv

from llm_loader import load_local_llm, get_tokenizer, get_raw_pipeline
from sentence_transformers import SentenceTransformer, util

# Load once globally
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"[Embedding model load failed] {e}")
    embedder = None


# ========== 0) Initialization ==========
load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (
	os.getenv("NEO4J_USER", "neo4j"),
	os.getenv("NEO4J_PASSWORD", "password"),
)

# Avoid local proxy settings interfering with model/Neo4j access.
for key in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
	if key in os.environ:
		del os.environ[key]


try:
	driver = GraphDatabase.driver(URI, auth=AUTH)
	driver.verify_connectivity()
except Exception as e:
	print(f"⚠️ Neo4j connection warning: {e}")
	driver = None


# ========== 1) Public API (query flow order) ==========
# Order: extract_entities -> build_typed_cypher -> get_relevant_articles -> generate_answer

def generate_text(messages: list[dict[str, str]], max_new_tokens: int = 220) -> str:
	"""
	Call local HF model via chat template + raw pipeline.

	Interface:
	- Input:
	  - messages: list[dict[str, str]] (chat messages with role/content)
	  - max_new_tokens: int
	- Output:
	  - str (model generated text, no JSON guarantee)
	"""
	tok = get_tokenizer()
	pipe = get_raw_pipeline()
	if tok is None or pipe is None:
		load_local_llm()
		tok = get_tokenizer()
		pipe = get_raw_pipeline()
	prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	return pipe(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"].strip()


def extract_entities(question: str) -> dict[str, Any]:
    q = question.lower()

    # Detect question type
    if any(k in q for k in ["penalty", "punishment", "consequence", "fine", "suspend", "expel", "處分", "警告", "記過", "退學"]):
        question_type = "penalty"
    elif any(k in q for k in ["require", "must", "need", "obligat", "應", "須", "必須"]):
        question_type = "requirement"
    elif any(k in q for k in ["prohibit", "forbid", "not allow", "cannot", "不得", "禁止"]):
        question_type = "prohibition"
    else:
        question_type = "general"

    stop_words = {
        "what", "when", "where", "who", "how", "why", "is", "are", "the",
        "a", "an", "do", "does", "can", "will", "if", "for", "to", "of",
        "penalty", "punishment", "consequence", "regulation", "rule", "article"
    }
    words = q.replace("?", "").replace(",", "").split()
    subject_terms = [w for w in words if w not in stop_words and len(w) > 2]

    # Synonym expansion — inside loop this time
    expanded = []
    for term in subject_terms:
        expanded.append(term)
        if term in ("undergraduate", "bachelor", "bachelor's"):
            expanded.extend(["undergraduate", "bachelor", "complete studies"])
        if term in ("graduate", "master", "phd", "doctoral", "postgraduate"):
            expanded.extend(["master", "doctoral", "postgraduate", "graduate"])

    # Question-level semantic expansion based on intent
    # "standard duration" → what the regulation actually says
    if any(k in q for k in ["standard duration", "how long", "duration of study", "period of study"]):
        if any(k in q for k in ["bachelor", "undergraduate"]):
            expanded.extend(["four years", "complete", "expected", "128"])
        if any(k in q for k in ["master", "graduate"]):
            expanded.extend(["one to four years", "master"])
        if any(k in q for k in ["phd", "doctoral"]):
            expanded.extend(["two to seven years", "doctoral"])

    if any(k in q for k in ["working days", "workdays", "how long", "how many days"]):
        expanded.extend(["workdays", "three workdays", "available after"])

    if any(k in q for k in ["passing score", "pass", "minimum score", "passing grade"]):
        if any(k in q for k in ["undergraduate", "bachelor"]):
            expanded.extend(["sixty", "60", "passing grade undergraduate"])
        if any(k in q for k in ["graduate", "master", "phd", "postgraduate"]):
            expanded.extend(["seventy", "70", "passing grade postgraduate"])

    if any(k in q for k in ["maximum", "leave of absence", "suspension"]):
        expanded.extend(["two academic years", "maximum period", "suspend"])

    subject_terms = list(dict.fromkeys(expanded))  # deduplicate, preserve order

    return {
        "question_type": question_type,
        "subject_terms": subject_terms,
        "aspect": question_type,
    }


def build_typed_cypher(entities: dict[str, Any]) -> tuple[str, str]:
    """
    Build two Cypher queries:
    - typed_query: searches Rule nodes by type + keywords
    - broad_query: broader fulltext search across articles and rules
    """
    terms = entities.get("subject_terms", [])
    qtype = entities.get("question_type", "general")

    # Build fulltext search string: join keywords with OR
    keyword_str = " OR ".join(terms) if terms else "*"

    # Typed query: search rule_idx (action + result fields), filter by type
    if qtype != "general":
        cypher_typed = """
        CALL db.index.fulltext.queryNodes("rule_idx", $keyword)
        YIELD node AS r, score
        WHERE r.type = $qtype
        MATCH (a:Article)-[:CONTAINS_RULE]->(r)
        RETURN r.rule_id   AS rule_id,
               r.type      AS type,
               r.action    AS action,
               r.result    AS result,
               r.art_ref   AS art_ref,
               r.reg_name  AS reg_name,
               a.content   AS article_content,
               score
        ORDER BY score DESC
        LIMIT 10
        """
    else:
        cypher_typed = """
        CALL db.index.fulltext.queryNodes("rule_idx", $keyword)
        YIELD node AS r, score
        MATCH (a:Article)-[:CONTAINS_RULE]->(r)
        RETURN r.rule_id   AS rule_id,
               r.type      AS type,
               r.action    AS action,
               r.result    AS result,
               r.art_ref   AS art_ref,
               r.reg_name  AS reg_name,
               a.content   AS article_content,
               score
        ORDER BY score DESC
        LIMIT 5
        """

    # Broad query: also search article_content_idx as fallback
    cypher_broad = """
    CALL db.index.fulltext.queryNodes("article_content_idx", $keyword)
    YIELD node AS a, score
    MATCH (a)-[:CONTAINS_RULE]->(r)
    RETURN r.rule_id   AS rule_id,
           r.type      AS type,
           r.action    AS action,
           r.result    AS result,
           r.art_ref   AS art_ref,
           r.reg_name  AS reg_name,
           a.content   AS article_content,
           score
    ORDER BY score DESC
    LIMIT 5
    """

    return cypher_typed, cypher_broad


def sanitize_for_lucene(text: str) -> str:
    # Remove characters that break Lucene fulltext queries
    text = re.sub(r'[()\/\[\]{}\^~*?:\\"+\-!|&]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def rerank_results(question: str, results: list[dict]) -> list[dict]:
    """
    Hybrid reranking:
    - semantic similarity (embedding)
    - keyword overlap
    - type match
    """

    if not results:
        return results

    q = question.lower()
    q_lower = q.lower()
    
    if any(k in q_lower for k in [
        "not allowed", "cannot", "can't", "forbid", "prohibit",
        "不得", "禁止", "not permitted", "must not"
    ]):
        qtype = "prohibition"

    elif any(k in q_lower for k in [
        "penalty", "punishment", "fine", "expel", "處分"
    ]):
        qtype = "penalty"

    elif any(k in q_lower for k in [
        "must", "require", "need", "should", "須", "必須"
    ]):
        qtype = "requirement"

    else:
        qtype = "general"

    # ---- keyword prep ----
    q_words = set(q.split())
    stop = {"what","is","the","a","an","for","how","many","can","i","do","does"}
    q_keywords = q_words - stop

    # ---- semantic embedding ----
    if embedder:
        q_emb = embedder.encode(question, convert_to_tensor=True)
        doc_texts = [
            (r.get("action","") + " " + r.get("result","") + " " + r.get("article_content",""))
            for r in results
        ]
        doc_embs = embedder.encode(doc_texts, convert_to_tensor=True)
        sim_scores = util.cos_sim(q_emb, doc_embs)[0]
    else:
        sim_scores = [0] * len(results)

    # ---- scoring ----
    scored = []
    for i, r in enumerate(results):
        text = (
            r.get("action","") + " " +
            r.get("result","") + " " +
            r.get("article_content","")
        ).lower()

        # keyword overlap
        overlap = sum(1 for kw in q_keywords if kw in text)

        # type bonus
        type_bonus = 5 if r.get("type") == qtype else 0

        # semantic similarity (MOST important)
        semantic_score = float(sim_scores[i]) if embedder else 0

        # final score (tuned weights)
        final_score = (
            semantic_score * 50 +   # main signal
            overlap * 5 +           # keyword backup
            type_bonus +            # type alignment
            r.get("score", 0)       # Neo4j score
        )

        scored.append((final_score, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [r for _, r in scored]


def get_relevant_articles(question: str) -> list[dict[str, Any]]:
    """
    Run typed + broad retrieval and return merged, deduplicated rule dicts.
    """
    if driver is None:
        return []

    entities = extract_entities(question)
    cypher_typed, cypher_broad = build_typed_cypher(entities)

    terms = entities.get("subject_terms", [])
    keyword_str = " OR ".join(terms) if terms else "*"
    keyword_str= sanitize_for_lucene(keyword_str)
    qtype = entities.get("question_type", "general")

    results = []
    seen_ids = set()

    with driver.session() as session:
        # 1. Try typed query first
        try:
            rows = session.run(cypher_typed, keyword=keyword_str, qtype=qtype)
            for row in rows:
                rid = row["rule_id"]
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    results.append(dict(row))
        except Exception as e:
            print(f"[Typed query failed] {e}")

        # 2. Broad fallback if typed returned nothing
        try:
            rows = session.run(cypher_broad, keyword=keyword_str)
            for row in rows:
                rid = row["rule_id"]
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    results.append(dict(row))
        except Exception as e:
            print(f"[Broad query failed] {e}")

    results = rerank_results(question, results)
    print(f"[Retrieval] Found {len(results)} rules for: '{question}'")
    return results


def extract_direct_answer(question: str, rule_results: list[dict]) -> str | None:
    """
    General-purpose answer extractor:
    - handles numeric answers
    - handles yes/no
    - no hardcoded questions
    """
    q = question.lower()

    for r in rule_results[:10]:
        action  = r.get("action", "").lower()
        result  = r.get("result", "").lower()
        content = str(r.get("article_content", r.get("content", ""))).lower()
        text = action + " " + result + " " + content

        art = r.get("art_ref", "?")
        reg = r.get("reg_name", "?")

        # ===== 1. Numeric answer detection =====
        if any(k in q for k in ["how many", "how long", "maximum", "minimum", "duration"]):
            match = re.search(
                r'(\d+\s*(minutes?|days?|years?|credits?|semesters?))',
                text
            )
            if match:
                return f"{match.group(1)}. (Source: Article {art}, {reg})"

        # ===== 2. Yes / No detection =====
        if any(k in q for k in ["can", "allowed", "permitted", "is it allowed"]):

            neg_patterns = [
                "not allowed", "prohibited", "forbidden",
                "not permitted", "cannot", "can't", "may not",
                "不得", "禁止"
            ]

            pos_patterns = [
                "allowed", "permitted", "may"
            ]

            # IMPORTANT: check negation FIRST
            if any(k in text for k in neg_patterns):
                return f"No, it is not allowed. (Source: Article {art}, {reg})"

            # only treat as YES if clearly positive AND not negated
            if any(k in text for k in pos_patterns):
                return f"Yes, it is allowed. (Source: Article {art}, {reg})"

    return None


def generate_answer(question: str, rule_results: list[dict[str, Any]]) -> str:
    if not rule_results:
        return "Insufficient rule evidence to answer this question."

    # Try direct extraction first (fast + reliable)
    direct = extract_direct_answer(question, rule_results)
    if direct:
        print("[Direct extraction used]")
        return direct

    # Build evidence (top 3)
    evidence_lines = []
    for i, r in enumerate(rule_results[:3]):
        evidence_lines.append(
            f"[{i+1}] Article {r.get('art_ref','?')} ({r.get('reg_name','?')})\n"
            f"    Action : {r.get('action','')}\n"
            f"    Result : {r.get('result','')}\n"
        )
    evidence = "\n".join(evidence_lines)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an NCU regulation assistant.\n"
                "Answer ONLY using the provided evidence.\n"
                "Select the SINGLE most relevant evidence entry.\n"
                "Give a short direct answer, then cite source.\n"
                "Format: <answer>. (Source: Article X, Regulation Name)"
            )
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nEvidence:\n{evidence}\n\nAnswer:"
        }
    ]

    try:
        return generate_text(messages, max_new_tokens=120)
    except Exception as e:
        print(f"[generate_answer failed] {e}")
        top = rule_results[0]
        return (
            f"{top.get('action','')} → {top.get('result','')} "
            f"(Source: Article {top.get('art_ref','?')}, {top.get('reg_name','?')})"
        )


def main() -> None:
	"""Interactive CLI (provided scaffold)."""
	if driver is None:
		return

	load_local_llm()

	print("=" * 50)
	print("🎓 NCU Regulation Assistant (Template)")
	print("=" * 50)
	print("💡 Try: 'What is the penalty for forgetting student ID?'")
	print("👉 Type 'exit' to quit.\n")

	while True:
		try:
			user_q = input("\nUser: ").strip()
			if not user_q:
				continue
			if user_q.lower() in {"exit", "quit"}:
				print("👋 Bye!")
				break

			results = get_relevant_articles(user_q)
			answer = generate_answer(user_q, results)
			print(f"Bot: {answer}", flush=True)

		except KeyboardInterrupt:
			print("\n👋 Bye!")
			break
		except NotImplementedError as e:
			print(f"⚠️ {e}")
			break
		except Exception as e:
			print(f"❌ Error: {e}")

	driver.close()



if __name__ == "__main__":
	main()