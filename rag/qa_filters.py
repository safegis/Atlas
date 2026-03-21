"""Qdrant filters so QA RAG never leaks one conversation's memory into another."""

from __future__ import annotations

from qdrant_client.http import models as qm


def build_qa_retrieval_filter(conversation_id: str | None) -> qm.Filter:
    """
    - With ``conversation_id``: OR of
        (1) ``scope=conversation_memory`` **and** ``conversation_id`` matches — past turns in *this* thread;
        (2) any point that is **not** ``scope=conversation_memory`` — knowledge_base uploads + legacy ingests.
    - Without ``conversation_id``: exclude all ``conversation_memory`` points (KB + legacy only).
    """
    mem = qm.FieldCondition(
        key="scope",
        match=qm.MatchValue(value="conversation_memory"),
    )
    if conversation_id and conversation_id.strip():
        cid = conversation_id.strip()
        return qm.Filter(
            should=[
                qm.Filter(
                    must=[
                        mem,
                        qm.FieldCondition(
                            key="conversation_id",
                            match=qm.MatchValue(value=cid),
                        ),
                    ]
                ),
                qm.Filter(must_not=[mem]),
            ]
        )
    return qm.Filter(must_not=[mem])
