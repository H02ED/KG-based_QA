"""Minimal KG builder template for Assignment 4.

Keep this contract unchanged:
- Graph: (Regulation)-[:HAS_ARTICLE]->(Article)-[:CONTAINS_RULE]->(Rule)
- Article: number, content, reg_name, category
- Rule: rule_id, type, action, result, art_ref, reg_name
- Fulltext indexes: article_content_idx, rule_idx
- SQLite file: ncu_regulations.db
"""

import os
import sqlite3
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

from llm_loader import load_local_llm, get_tokenizer, get_raw_pipeline
import time
import json
import re


# ========== 0) Initialization ==========
load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (
    os.getenv("NEO4J_USER", "neo4j"),
    os.getenv("NEO4J_PASSWORD", "password"),
)


def extract_entities(article_number: str, reg_name: str, content: str) -> dict:
    tok = get_tokenizer()
    pipe = get_raw_pipeline()

    if tok is None or pipe is None:
        load_local_llm()
        tok = get_tokenizer()
        pipe = get_raw_pipeline()

    content_trimmed = content[:500]

    # ✅ Minimal prompt — fewer input tokens = faster
    prompt = [
        {
            "role": "system",
            "content": (
                'Extract ALL rules from this regulation article. '
                'Return ONLY JSON: {"rules":[{"type":"requirement|penalty|prohibition|general",'
                '"action":"the situation or condition","result":"the consequence or outcome"}]}, '
                'type must be EXACTLY ONE of: requirement, penalty, prohibition, general'
            )
        },
        {
            "role": "user",
            "content": f"Article {article_number} from {reg_name}:\n{content_trimmed}"
        }
    ]

    input_text = tok.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    
    # Debug: see how long inference actually takes

    t0 = time.time()
    print(f"  [Input tokens: {len(tok.encode(input_text))}]")

    try:
        output = pipe(input_text)[0]["generated_text"]

        print(f"  [Inference time: {time.time()-t0:.1f}s]")
        print(f"  [Raw output: {output[:200]}]")

        start = output.find("{")
        end = output.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found")

        return json.loads(output[start:end])

    except Exception as e:
        print(f"  [LLM Extraction Failed: {e}]")
        return {"rules": []}

    except Exception as e:
        print(f"  [LLM Extraction Failed: {e}]")
        return {"rules": []}


def build_fallback_rules(article_number: str, content: str) -> list[dict]:
    """Split content into sentences and create one rule per sentence."""
    sentences = re.split(r'[.;]', content)
    rules = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 20:
            rules.append({
                "type": "general",
                "action": sent[:200],
                "result": "refer to article"
            })
    if not rules:
        rules = [{
            "type": "general",
            "action": content[:200],
            "result": "refer to article"
        }]
    return rules


def build_graph() -> None:
    """Build KG from SQLite into Neo4j using the fixed assignment schema."""
    sql_conn = sqlite3.connect("ncu_regulations.db")
    cursor = sql_conn.cursor()
    driver = GraphDatabase.driver(URI, auth=AUTH)

    load_local_llm()

    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

        # 1) Read regulations and create Regulation nodes.
        cursor.execute("SELECT reg_id, name, category FROM regulations")
        regulations = cursor.fetchall()
        reg_map: dict[int, tuple[str, str]] = {}

        for reg_id, name, category in regulations:
            reg_map[reg_id] = (name, category)
            session.run(
                "MERGE (r:Regulation {id:$rid}) SET r.name=$name, r.category=$cat",
                rid=reg_id,
                name=name,
                cat=category,
            )

        # 2) Read articles and create Article + HAS_ARTICLE.
        cursor.execute("SELECT reg_id, article_number, content FROM articles")
        articles = cursor.fetchall()

        for reg_id, article_number, content in articles:
            reg_name, reg_category = reg_map.get(reg_id, ("Unknown", "Unknown"))
            session.run(
                """
                MATCH (r:Regulation {id: $rid})
                CREATE (a:Article {
                    number:   $num,
                    content:  $content,
                    reg_name: $reg_name,
                    category: $reg_category
                })
                MERGE (r)-[:HAS_ARTICLE]->(a)
                """,
                rid=reg_id,
                num=article_number,
                content=content,
                reg_name=reg_name,
                reg_category=reg_category,
            )

        # 3) Create full-text index on Article content.
        session.run(
            """
            CREATE FULLTEXT INDEX article_content_idx IF NOT EXISTS
            FOR (a:Article) ON EACH [a.content]
            """
        )

        rule_counter = 0

        for reg_id, article_number, content in articles:
            reg_name, _ = reg_map.get(reg_id, ("Unknown", "Unknown"))

            print(f"\nProcessing: {article_number}")

            # 1. Extract rules using LLM
            extracted = extract_entities(article_number, reg_name, content)
            rules = extracted.get("rules", [])

            print(f"[Extracted]: {rules}")

            # 2. Fallback if empty
            if not rules:
                rules = build_fallback_rules(article_number, content)
                print("[Fallback used]")

            # 3. Deduplicate rules with identical actions within this article
            seen_actions = set()

            for r in rules:
                action = r.get("action", "").strip()
                result = r.get("result", "").strip()
                rtype  = r.get("type", "general")

                if not action:
                    action = content[:200]
                if not result:
                    result = "refer to article"

                # Skip duplicate actions within same article
                action_key = action[:80].lower()
                if action_key in seen_actions:
                    continue
                seen_actions.add(action_key)

                rule_id = f"R{rule_counter}"
                rule_counter += 1

                # 4. Create Rule node with full article content attached
                session.run(
                    """
                    MATCH (a:Article {number: $num, reg_name: $reg_name})
                    CREATE (r:Rule {
                        rule_id:  $rid,
                        type:     $type,
                        action:   $action,
                        result:   $result,
                        art_ref:  $num,
                        reg_name: $reg_name,
                        content:  $content
                    })
                    MERGE (a)-[:CONTAINS_RULE]->(r)
                    """,
                    num=article_number,
                    reg_name=reg_name,
                    rid=rule_id,
                    type=rtype,
                    action=action,
                    result=result,
                    content=content,  # full article content on every rule
                )

        # 4) Create full-text index on Rule fields including content.
        session.run(
            """
            CREATE FULLTEXT INDEX rule_idx IF NOT EXISTS
            FOR (r:Rule) ON EACH [r.action, r.result, r.content]
            """
        )

        # 5) Coverage audit.
        coverage = session.run(
            """
            MATCH (a:Article)
            OPTIONAL MATCH (a)-[:CONTAINS_RULE]->(r:Rule)
            WITH a, count(r) AS rule_count
            RETURN count(a) AS total_articles,
                   sum(CASE WHEN rule_count > 0 THEN 1 ELSE 0 END) AS covered_articles,
                   sum(CASE WHEN rule_count = 0 THEN 1 ELSE 0 END) AS uncovered_articles
            """
        ).single()

        total_articles    = int((coverage or {}).get("total_articles", 0) or 0)
        covered_articles  = int((coverage or {}).get("covered_articles", 0) or 0)
        uncovered_articles = int((coverage or {}).get("uncovered_articles", 0) or 0)

        print(
            f"\n[Coverage] covered={covered_articles}/{total_articles}, "
            f"uncovered={uncovered_articles}"
        )

    driver.close()
    sql_conn.close()



if __name__ == "__main__":
    
    start = time.time()
    build_graph()
    print("Total time:", time.time() - start)
