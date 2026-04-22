import os
import sys
from pathlib import Path
for key in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    if key in os.environ:
        del os.environ[key]

import json
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT_DIR = Path(__file__).resolve().parent
TEST_DATA_PATH = ROOT_DIR / "test_data.json"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from query_system import get_relevant_articles, generate_answer, generate_text

load_dotenv()


def preflight_checks() -> bool:
    """Fail fast when grading environment is not ready."""
    if not (ROOT_DIR / "query_system.py").exists():
        print(f"[X] Error: query_system.py not found in {ROOT_DIR}")
        return False

    if not TEST_DATA_PATH.exists():
        print(f"[X] Error: test data not found: {TEST_DATA_PATH}")
        return False

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))

    try:
        driver = GraphDatabase.driver(uri, auth=auth)
        driver.verify_connectivity()
        with driver.session() as session:
            count = session.run("MATCH (r:Rule) RETURN count(r) AS c").single()["c"]
        driver.close()
    except Exception as e:
        print(f"[X] Error: Neo4j preflight failed: {e}")
        print("    Hint: Start Neo4j and run build_kg.py before auto-test.")
        return False

    if count == 0:
        print("[X] Error: Neo4j has 0 Rule nodes. Please run setup_data.py and build_kg.py first.")
        return False

    print(f"[OK] Preflight passed: Neo4j connected, Rule nodes = {count}")
    return True

def ask_bot_no_metadata(question):
    """Retrieve relevant articles without relying on metadata, then generate an answer."""
    try:
        # Directly retrieve articles without metadata dependency
        articles = get_relevant_articles(question)
        print("[?] Retrieved Articles for Question (No Metadata):", articles)
        final_answer = generate_answer(question, articles)
        return final_answer
    except Exception as e:
        return f"Error: {str(e)}"

def evaluate_with_llm(question, expected, actual):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an impartial judge evaluating a Q&A system for university regulations. "
                "Respond with exactly one word: PASS or FAIL."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Expected Answer: {expected}\n"
                f"Actual Answer from Bot: {actual}\n\n"
                "Does the Actual Answer convey the same key information as the Expected Answer?\n"
                "Rules:\n"
                "1. If the bot says it cannot find information or gives a wrong number/fact → FAIL.\n"
                "2. Minor wording differences (e.g. '20 mins' vs 'twenty minutes') → PASS.\n"
                "3. More detail than expected but the core fact is correct → PASS.\n"
                "Answer with one word only: PASS or FAIL."
            ),
        },
    ]

    try:
        text = generate_text(messages).strip()
        if "PASS" in text.upper():
            return "PASS"
        return "FAIL"
    except Exception as e:
        return f"FAIL (Judge Error: {str(e)})"

def run_llm_evaluation_no_metadata():
    if not preflight_checks():
        return

    try:
        with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print("[X] Error: test_data.json not found!")
        return

    print(f"[*] Starting LLM-based Evaluation (No Metadata) for {len(test_cases)} Questions...\n")
    
    passed_count = 0
    results_log = []

    for i, case in enumerate(test_cases):
        qid = case["id"]
        question = case["question"]
        expected_answer = case["answer"]
        
        print(f"Testing Q{qid}: {question}")
        
        start_time = time.time()
        bot_answer = ask_bot_no_metadata(question)
        
        verdict = evaluate_with_llm(question, expected_answer, bot_answer)
        duration = time.time() - start_time
        
        status_icon = "[OK]" if "PASS" in verdict else "[FAIL]"
        if "PASS" in verdict:
            passed_count += 1
            
        print(f"  -> Bot Says: {bot_answer.strip()}")
        print(f"  -> Judge: {status_icon} {verdict} (Time: {duration:.2f}s)")
        print("-" * 50)
        
        results_log.append({
            "id": qid,
            "question": question,
            "expected": expected_answer,
            "bot_response": bot_answer,
            "result": verdict
        })

    print("\n" + "="*30)
    print(f"=== Evaluation Summary (No Metadata) ===")
    print(f"Total: {len(test_cases)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {len(test_cases) - passed_count}")
    if len(test_cases) > 0:
        print(f"Accuracy: {(passed_count / len(test_cases)) * 100:.1f}%")
    print("="*30)
    
    
    # Save log file
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = ROOT_DIR / f"{timestamp}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        for entry in results_log:
            f.write(f"Q{entry['id']}: {entry['question']}\n")
            f.write(f"Expected: {entry['expected']}\n")
            f.write(f"Bot: {entry['bot_response']}\n")
            f.write(f"Result: {entry['result']}\n")
            f.write("-" * 50 + "\n")
        f.write(f"\nTotal: {len(test_cases)}\n")
        f.write(f"Passed: {passed_count}\n")
        f.write(f"Failed: {len(test_cases) - passed_count}\n")
        if len(test_cases) > 0:
            f.write(f"Accuracy: {(passed_count / len(test_cases)) * 100:.1f}%\n")
    print(f"[Log saved] {log_path}")

if __name__ == "__main__":
    run_llm_evaluation_no_metadata()