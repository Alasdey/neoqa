# mock_prereqs.py - Combined Mock for NLI + Question Filter (Bypasses Both Steps)
import argparse
import json
import random  # Optional: For realistic % fails/neutrals
from pathlib import Path
from typing import List, Dict
from data_gen.util.file_util import read_json  # Your util (or json.load)
from data_gen.util.ids import generate_id

def idfy_news_articles(storyline_dir: Path):
    news_path = storyline_dir / "news-articles.json"
    idfy_path = storyline_dir / "news-articles-idfy.json"
    articles = read_json(news_path)
    for key in articles["articles"]:
        for article in articles["articles"][key]:
            article["article_id"] = generate_id(article)
    with idfy_path.open("w") as f:
        f.write(json.dumps(articles)+'\n')
    return idfy_path


def mock_nli(storyline_dir: Path, fail_pct: float = 0.0) -> Path:
    """Mock nli-predictions.jsonl: Mostly entailment (configurable fail %)."""
    # news_path = storyline_dir / "news-articles.json"
    news_path = storyline_dir / "news-articles-idfy.json"
    print("The news path searched is", news_path)
    # news_path = idfy_path if idfy_path.exists() else news_path
    if not news_path.exists():
        print(f"No news file in {news_path.parent} → Skipping NLI")
        return None
    
    articles = read_json(news_path)
    preds = []
    for event_articles in articles['articles'].values():
        for art in event_articles:
            for sent_id in art['used_items']:
                # Mock: Mostly entail (tune fail_pct for ~unsures)
                nli_pred = "entailment" if random.random() > fail_pct else random.choice(["neutral", "contradiction"])
                preds.append({
                    "article_id": art['article_id'],
                    "sentence_id": sent_id,
                    "label": "entailment",  # Gold
                    "nli_prediction": nli_pred  # Mock pred → unsure if != label
                })
    
    out_path = storyline_dir / "nli-predictions.jsonl"
    with out_path.open('w') as f:
        for p in preds:
            f.write(json.dumps(p) + '\n')
    unsure_pct = sum(1 for p in preds if p['nli_prediction'] != 'entailment') / len(preds)
    print(f"Mocked {len(preds)} NLI ({unsure_pct:.1%} unsure) → {out_path}")
    return out_path

def mock_question_filter(q_dir: Path, fail_pct: float = 0.0) -> Path:
    """Mock filter-evaluated-outputs.jsonl: Mostly success (configurable fail %)."""
    raw_files = ['multiv2-bridge-series.json', 'timespan-questions_v2complete.json']
    questions = []
    for f_name in raw_files:
        f_path = q_dir / f_name
        if f_path.exists():
            data = read_json(f_path)
            questions.extend(data.get('questions', []))
    
    if not questions:
        print("No raw questions → Skipping filter")
        return None
    
    for q in questions:
        is_success = random.random() > fail_pct
        q['filtered'] = 'success' if is_success else 'fail'
        q['filter_reason'] = 'Mock: Valid' if is_success else 'Mock: Invalid (e.g., incoherent)'
    
    out_path = q_dir / 'filter-evaluated-outputs.jsonl'
    with out_path.open('w') as f:
        for q in questions:
            f.write(json.dumps(q) + '\n')
    
    success_pct = sum(1 for q in questions if q['filtered'] == 'success') / len(questions)
    print(f"Mocked {len(questions)} questions ({success_pct:.1%} success) → {out_path}")
    return out_path

def main(storyline_dir: str, fail_pct: float = 0.0):
    storyline_dir = Path(storyline_dir)
    print(f"🔧 Mocking NLI + Q-Filter for {storyline_dir} (fail_pct={fail_pct})")
    
    idfy_path = idfy_news_articles(storyline_dir / "news")
    nli_path = mock_nli(storyline_dir / "news", fail_pct)
    q_path = mock_question_filter(storyline_dir / "questions", fail_pct)
    
    if nli_path and q_path:
        print("✅ Prereqs mocked → Export ready!")
    else:
        print("⚠️  Skipped (missing raw files)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock BOTH NLI + Question Filter (bypass prereqs)")
    parser.add_argument("storyline_dir", help="Storyline path (e.g., outputs/storylines-final4/story_id)")
    parser.add_argument("--fail-pct", type=float, default=0.0, help="Mock fail/neutral % (0=keep all)")
    args = parser.parse_args()
    main(args.storyline_dir, args.fail_pct)