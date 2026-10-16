"""
ReAct-style agent for airline customer service.
Loop: Thought → Action → Observation → repeat until Final Answer.
"""

import json
import re
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from tools import execute_tool, get_tool_descriptions, TOOLS
from prompts import SYSTEM_PROMPT
from llm import chat

MAX_STEPS = 15  # safety limit to prevent infinite loops


def parse_llm_output(text: str) -> dict:
    """
    Parse the LLM output into a structured dict.
    Returns one of:
        {"type": "action",       "tool": str, "args": dict}
        {"type": "final_answer", "answer": str}
        {"type": "error",        "raw": str}
    """
    # Check for Final Answer
    fa_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if fa_match:
        return {"type": "final_answer", "answer": fa_match.group(1).strip()}

    # Check for Action + Action Input
    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match = re.search(r"Action Input:\s*(\{.*?\}|\[.*?\])", text, re.DOTALL)

    if action_match:
        tool_name = action_match.group(1).strip()
        args = {}
        if input_match:
            try:
                args = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                return {"type": "error", "raw": f"Could not parse Action Input JSON: {input_match.group(1)}"}
        return {"type": "action", "tool": tool_name, "args": args}

    return {"type": "error", "raw": text}


def run(task: str, verbose: bool = True) -> str:
    """
    Run the ReAct agent on a task.
    Returns the final answer string.
    """
    tool_descriptions = get_tool_descriptions()
    system_prompt = SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": task},
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

    for step in range(1, MAX_STEPS + 1):
        if verbose:
            print(f"\n--- Step {step} ---")

        # Call LLM
        response_text = chat(messages)

        if verbose:
            print(response_text)

        # Append LLM response to history
        messages.append({"role": "assistant", "content": response_text})

        # Parse LLM output
        parsed = parse_llm_output(response_text)

        if parsed["type"] == "final_answer":
            if verbose:
                print(f"\n{'='*60}")
                print(f"Final Answer: {parsed['answer']}")
                print(f"{'='*60}\n")
            return parsed["answer"]

        elif parsed["type"] == "action":
            tool_name = parsed["tool"]
            args = parsed["args"]

            if verbose:
                print(f"\n[tool call] {tool_name}({json.dumps(args)})")

            # Execute tool
            observation = execute_tool(tool_name, args)

            if verbose:
                print(f"[observation] {observation}")

            # Feed observation back as user message
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}",
            })

        else:
            # Parsing error — nudge the LLM
            if verbose:
                print(f"[parse error] Could not parse LLM output, nudging...")
            messages.append({
                "role": "user",
                "content": (
                    "I could not parse your last response. "
                    "Please respond with exactly:\n"
                    "Thought: <reasoning>\n"
                    "Action: <tool_name>\n"
                    "Action Input: <json>\n"
                    "or:\n"
                    "Thought: <reasoning>\n"
                    "Final Answer: <answer>"
                ),
            })

    return "Agent reached maximum steps without a final answer."


if __name__ == "__main__":
    # Quick smoke test
    tasks = [
        "My customer ID is C001. Can you show me all my bookings?",
        "I have booking BK101. I want to change it to flight 6E-301. What's the fee and can you make the change?",
        "Cancel booking BK101 for customer C001 and tell me the refund amount.",
    ]

    for task in tasks:
        answer = run(task, verbose=True)
