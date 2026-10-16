from tau_airline import WIKI

# Agent: the tau-bench airline policy plus ReAct-style working rules.
AGENT_SYSTEM_PROMPT = WIKI + """

# How you work
- Work step by step: reason about what you know, call ONE tool, read its result, then decide the next step.
- Never invent ids, flight numbers or prices -- look them up with tools.
- When you need something from the user (or their confirmation), reply in plain text without a tool call.
"""

# User simulator: same prompt tau-bench uses for its LLM-simulated customer.
USER_SIM_PROMPT = """You are a user interacting with an agent.

Instruction: {instruction}

Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""
