
EXPLANATION_TEMPLATE = """
You are a friendly step-by-step tutor. Explain the concept titled: "{title}" using the provided content.
Keep each message extremely short (one or two sentences). Split the explanation into {steps} numbered messages.
Include only the step lines in the response.

Context / Content:
{content}

Conversation history (latest first):
{conversation_history}
"""

CHECK_QUESTION_TEMPLATE = """
Create a *very simple* check question (one short question) for the concept '{title}'.
The question should be answerable in 1-3 words or a single sentence.
Return only the question text.

Conversation history (latest first):
{conversation_history}
"""

EVAL_PROMPT_TEMPLATE = """
You are an objective grader. Expected keywords: {keywords}
User answer: {user_answer}

Using the conversation history (latest first):
{conversation_history}

Decide whether the user's answer is: CORRECT, PARTIAL, or WRONG.
Return JSON-like string: VERDICT: <CORRECT|PARTIAL|WRONG>\\nJUSTIFICATION: <one short sentence>\\nCORRECTION: <one short sentence correction to teach the learner>
"""
