from pulse.agent.tools import package

# ReAct-style working rules added after the domain's tau-bench policy (wiki), per domain.
_RULES = {
    "airline": """
- When you need several lookups that do not depend on each other (e.g. the details of every reservation
  in the user's profile, or flights on several dates or routes), request them ALL in the same reply as
  parallel tool calls, not one per reply. Only batch calls whose arguments you already know.
- Make changes (book, cancel, update, send) one at a time, each after the user confirms.
- Never invent ids, flight numbers or prices -- look them up with tools.""",
    "retail": """
- When you need several lookups that do not depend on each other (e.g. the details of every order
  in the user's profile, or several products), request them ALL in the same reply as
  parallel tool calls, not one per reply. Only batch calls whose arguments you already know.
- Make changes (cancel, modify, return, exchange, update address) one at a time, each after the user confirms.
- Never invent ids, item ids or prices -- look them up with tools.""",
}


def agent_system_prompt(domain: str | None = None) -> str:
    """The agent's system prompt: the domain's tau-bench policy plus how to work (domain None = env DOMAIN)."""
    pkg = package(domain)
    rules = _RULES["retail" if pkg.__name__.endswith("retail") else "airline"]
    return pkg.WIKI + """

# How you work
- Work step by step: reason about what you know, call tools, read their results, then decide the next step.""" \
        + rules + """
- When you need something from the user (or their confirmation), reply in plain text without a tool call.
"""


AGENT_SYSTEM_PROMPT = agent_system_prompt()

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
