# backend/core/orchestrator_agent.py
import asyncio
from typing import TypedDict,List, Dict, Any, Optional
from .revision_agent import RevisionAgent
from backend.core.quiz_agent import QuizAgent
from backend.core.feedback_agent import FeedbackAgent
from backend.core.qa_agent import QAAgent
from backend.core.conclusion_agent import ConclusionAgent
from .mongodb_client import MongoDBClient
from backend.models.schemas import RevisionSessionData, SessionState
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# LangGraph import - we use it to express the orchestrator flow
from langgraph.graph import StateGraph

class OrchestratorState(TypedDict, total=False):
    user_message: Optional[str]
    assistant_message: Optional[str]
    stage: Optional[str]
    conversation_history: Optional[List[Dict[str, Any]]]
    current_chunk_index: Optional[int]
    current_question_concept: Optional[str]
    current_expected_keywords: Optional[List[str]]
    expecting_answer: Optional[bool]

class OrchestratorAgent:
    """
    Main orchestrator that implements the flow using langgraph.
    It stores conversation turns and session progress in MongoDB (via MongoDBClient).
    """
    def __init__(self, mongodb: Optional[MongoDBClient] = None):
        self.rev_agent = RevisionAgent()
        self.quiz_agent = QuizAgent()
        self.feedback_agent = FeedbackAgent()
        self.qa_agent = QAAgent()
        self.conclusion_agent = ConclusionAgent()
        self.mongo = mongodb or MongoDBClient()

        # Build LangGraph flow
        self.graph = StateGraph(OrchestratorState)
        self._build_graph()

    def _build_graph(self):
        g = self.graph

        # Add nodes (no logic yet, just placeholders)
        g.add_node("EXPLAIN", lambda state: state)
        g.add_node("ASK_CHECK", lambda state: state)
        g.add_node("WAIT_ANSWER", lambda state: state)
        g.add_node("GIVE_FEEDBACK", lambda state: state)
        g.add_node("NEXT_CONCEPT", lambda state: state)

        # Add edges between them
        g.add_edge("EXPLAIN", "ASK_CHECK")
        g.add_edge("ASK_CHECK", "WAIT_ANSWER")
        g.add_edge("WAIT_ANSWER", "GIVE_FEEDBACK")
        g.add_edge("GIVE_FEEDBACK", "NEXT_CONCEPT")
        g.add_edge("NEXT_CONCEPT", "EXPLAIN")
        g.add_edge("NEXT_CONCEPT", "END")


    # utility to stringify conversation history for prompts (latest first)
    def _format_conversation_history(self, session_doc: Dict[str, Any], limit: int = 10) -> str:
        if not session_doc:
            return ""
        conv = session_doc.get("conversation_history", [])[-limit:]
        # reverse for latest-first display
        conv = list(reversed(conv))
        lines = []
        for i, turn in enumerate(conv):
            user = turn.get("user_message", "")
            assistant = turn.get("assistant_message", "")
            ts = turn.get("timestamp", "")
            lines.append(f"[{i}] user: {user} | assistant: {assistant}")
        return "\n".join(lines)

    async def start_revision_session(self, topic: str, student_id: str, session_id: str) -> Dict[str, Any]:
        """
        Create/initialize session, fetch topic content from MongoDB (subtopics), and return first explanation + check question.
        """
        # prepare or fetch session doc
        session_doc = self.mongo.get_revision_session(session_id) or {}
        if not session_doc:
            # create a new revision session doc structure
            session_doc = {
                "session_id": session_id,
                "student_id": student_id,
                "topic": topic,
                "started_at": datetime.utcnow(),
                "conversation_count": 0,
                "is_complete": False,
                "max_conversations": 999,
                "completion_threshold": 0,
                "conversation_history": []
            }
            self.mongo.save_revision_session(session_doc)

        # fetch subtopics for the topic
        # topic in your db may be stored as 'topic_title' - adapt as your schema
        # Accept both "Chapter - Topic #: title" or plain title
        topic_title = topic.split(": ")[-1] if ": " in topic else topic
        subtopics = self.mongo.get_topic_subtopics(topic_title)
        # if not found, try get_topic_content fallback
        if not subtopics:
            subtopic_chunks = self.mongo.get_topic_content(topic)
            subtopics = [{"subtopic_number": c["id"], "subtopic_title": c.get("subtopic_title", ""), "content": c["text"]} for c in subtopic_chunks]

        # store concept chunks into session_doc
        session_doc["concept_chunks"] = subtopics
        session_doc["current_chunk_index"] = 0
        self.mongo.save_revision_session(session_doc)

        # start first concept
        return await self._present_current_concept(session_doc)

    async def _present_current_concept(self, session_doc: Dict[str, Any]) -> Dict[str, Any]:
        idx = session_doc.get("current_chunk_index", 0)
        chunks = session_doc.get("concept_chunks", [])
        if idx >= len(chunks):
            # Finished all concepts
            session_doc["is_complete"] = True
            self.mongo.save_revision_session(session_doc)
            summary = await self.conclusion_agent.summary(
                correct=len([p for p in session_doc.get("concepts_learned", []) if p]),
                total=len(chunks),
                conversation_history=self._format_conversation_history(session_doc)
            )
            return {"response": summary, "is_session_complete": True, "conversation_count": session_doc.get("conversation_count", 0)}
        current = chunks[idx]
        title = current.get("subtopic_title") or f"Concept {current.get('subtopic_number')}"
        content = current.get("content", "")
        # generate explanation steps and check question
        conv_hist = self._format_conversation_history(session_doc)
        steps = await self.rev_agent.generate_explanation_steps(title, content, conversation_history=conv_hist, steps=3)
        check_q = await self.rev_agent.make_check_question(title, content, conversation_history=conv_hist)
        # Append assistant message with explanation
        assistant_message = "\n".join(steps) + "\n\nCheck question: " + check_q
        turn = {
            "turn": session_doc.get("conversation_count", 0) + 1,
            "user_message": None,
            "assistant_message": assistant_message,
            "stage": "explain",
            "timestamp": datetime.utcnow(),
            "concept_covered": title,
            "question_asked": True,
        }
        session_doc.setdefault("conversation_history", []).append(turn)
        session_doc["conversation_count"] = session_doc.get("conversation_count", 0) + 1
        # set expecting answer
        session_doc["expecting_answer"] = True
        session_doc["current_question_concept"] = title
        session_doc["current_expected_keywords"] = [w for w in (title.split()[:3])]  # naive default; we store real keywords later if available
        self.mongo.save_revision_session(session_doc)

        return {"response": assistant_message, "is_session_complete": False, "conversation_count": session_doc["conversation_count"], "current_stage": "explain", "current_concept": title}

    async def handle_user_input(self, session_id: str, user_query: str) -> Dict[str, Any]:
        session_doc = self.mongo.get_revision_session(session_id) or {}
        if not session_doc:
            return {"response":"Session not found. Start a new revision session.", "is_session_complete": True, "conversation_count": 0}

        # basic question detection: if user asks question -> route to QAAgent
        is_question = user_query.strip().endswith('?') or any(user_query.lower().startswith(w) for w in ("why","what","how","when","where","who"))
        conv_hist = self._format_conversation_history(session_doc)
        # Save user turn
        user_turn = {
            "turn": session_doc.get("conversation_count", 0) + 1,
            "user_message": user_query,
            "assistant_message": None,
            "stage": "user_input",
            "timestamp": datetime.utcnow(),
            "concept_covered": session_doc.get("current_question_concept")
        }
        session_doc.setdefault("conversation_history", []).append(user_turn)
        session_doc["conversation_count"] = session_doc.get("conversation_count", 0) + 1
        self.mongo.save_revision_session(session_doc)

        if is_question:
            # route to QAAgent
            answer = await self.qa_agent.answer_question(user_query, conversation_history=conv_hist, content="")
            # save assistant turn
            assistant_turn = {
                "turn": session_doc["conversation_count"],
                "user_message": user_query,
                "assistant_message": answer,
                "stage": "qa",
                "timestamp": datetime.utcnow(),
            }
            session_doc.setdefault("conversation_history", []).append(assistant_turn)
            self.mongo.save_revision_session(session_doc)
            return {"response": answer, "conversation_count": session_doc["conversation_count"], "is_session_complete": False, "current_stage": "qa"}

        # otherwise, treat it as answer to the check question
        expected_keywords = session_doc.get("current_expected_keywords", [])
        eval_result = await self.rev_agent.evaluate_answer(user_query, expected_keywords, conversation_history=conv_hist)
        verdict = eval_result.get("verdict", "WRONG")
        assistant_feedback = self.feedback_agent.feedback_for(verdict, {"correction": eval_result.get("correction", eval_result.get("justification", ""))})
        # save feedback as assistant message
        assistant_turn = {
            "turn": session_doc["conversation_count"],
            "user_message": user_query,
            "assistant_message": assistant_feedback,
            "stage": "feedback",
            "timestamp": datetime.utcnow(),
            "correct_answer": (verdict == "CORRECT")
        }
        session_doc.setdefault("conversation_history", []).append(assistant_turn)

        # store progress flags
        session_doc.setdefault("concepts_covered", []).append(session_doc.get("current_question_concept"))
        if verdict == "CORRECT":
            session_doc.setdefault("concepts_learned", []).append(session_doc.get("current_question_concept"))
        else:
            # for partial/wrong we still record but mark remedial
            session_doc.setdefault("needs_remedial", True)

        # Move to next chunk
        session_doc["current_chunk_index"] = session_doc.get("current_chunk_index", 0) + 1
        session_doc["expecting_answer"] = False
        session_doc["current_question_concept"] = None
        self.mongo.save_revision_session(session_doc)

        # prepare next concept or conclusion
        next_payload = await self._present_current_concept(session_doc)
        # combine feedback + next concept explanation in response to frontend (so user sees feedback then next explanation)
        combined_response = assistant_feedback + "\n\n---\n\n" + next_payload.get("response", "")
        return {
            "response": combined_response,
            "conversation_count": session_doc["conversation_count"],
            "is_session_complete": next_payload.get("is_session_complete", False),
            "current_stage": "feedback_and_next"
        }
