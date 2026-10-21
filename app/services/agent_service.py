from app.services.rag_service import answer_question


def run_agent(
    task: str,
    question: str,
    top_k: int = 3,
) -> dict:
    """
    Run a simple enterprise knowledge agent workflow.
    """

    if not question or not question.strip():
        raise ValueError("Please enter a question.")

    task = task.lower().strip()

    # Step 1: Select workflow based on task
    if task not in {
        "question",
        "summarize",
        "recommend",
        "action_items",
    }:
        raise ValueError(
            "Unsupported task. Choose question, summarize, "
            "recommend, or action_items."
        )

    # Step 2: Retrieve context and generate an answer
    result = answer_question(
        question=question,
        top_k=top_k,
    )

    # Step 3: Add task-specific instructions
    instructions = {
        "question": (
            "Answer the question using the available "
            "document evidence."
        ),
        "summarize": (
            "Provide a concise summary of the key points "
            "from the answer and its document evidence."
        ),
        "recommend": (
            "Provide practical recommendations based only "
            "on the document evidence. Clearly label any "
            "suggestions that are not explicitly stated "
            "in the documents."
        ),
        "action_items": (
            "Extract actionable steps from the document "
            "evidence. Do not invent company requirements."
        ),
    }

    # Step 4: Return workflow result
    return {
        "task": task,
        "instruction": instructions[task],
        "answer": result["answer"],
        "sources": result["sources"],
    }