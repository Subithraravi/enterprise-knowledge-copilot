from unittest.mock import patch

import pytest

from app.services.agent_service import run_agent


def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        run_agent(
            task="question",
            question="",
        )


def test_unsupported_task_is_rejected():
    with pytest.raises(ValueError):
        run_agent(
            task="unknown",
            question="What is the leave policy?",
        )


@patch("app.services.agent_service.answer_question")
def test_question_task_returns_answer(mock_answer):
    mock_answer.return_value = {
        "answer": "Employees should follow the documented policy.",
        "sources": [
            {
                "filename": "HR-Policy.pdf",
                "text": "Sample policy text",
                "score": 0.9,
            }
        ],
    }

    result = run_agent(
        task="question",
        question="What is the leave policy?",
    )

    assert result["task"] == "question"
    assert "answer" in result
    assert len(result["sources"]) == 1