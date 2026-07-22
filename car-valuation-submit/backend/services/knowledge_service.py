import json
from functools import lru_cache

from config import settings


@lru_cache(maxsize=1)
def load_knowledge() -> list[dict]:
    with settings.knowledge_path.open("r", encoding="utf-8") as knowledge_file:
        records = json.load(knowledge_file)
    return [record for record in records if record.get("source_id") and record.get("content")]


def retrieve_knowledge(query: str, limit: int = 3) -> list[dict]:
    query_lower = query.lower()
    scored = []
    for record in load_knowledge():
        keywords = record.get("keywords", [])
        score = sum(1 for keyword in keywords if keyword.lower() in query_lower)
        if score:
            scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return load_knowledge()[:limit]
    return [record for _, record in scored[:limit]]
