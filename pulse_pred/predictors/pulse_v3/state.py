"""
Environment-state flags for every hint: facts about the conversation that come from the USER and the
TOOL RESULTS, never from how the agent model writes -- so they mean the same thing whatever LLM runs
the agent. These are the facts tau-bench's branches depend on (is the user identified? is the order
pending or delivered? did the user just confirm? did the last call fail?).

state_by_key(eps) -> {(episode index, phase, step index): vector}, keyed like predictors.contexts:
at a "call" hint only results of earlier steps are known; at a "result" hint the step's own results too.
"""

import json
import math
import re

import numpy as np

from ...tools import TOOL

INTENTS = {
    "cancel": r"\bcancel", "change": r"\b(change|modify|update|switch|move)\b", "exchange": r"\bexchange",
    "return": r"\breturn", "book": r"\bbook", "baggage": r"\b(bag|baggage|luggage)", "refund": r"\brefund",
    "address": r"\baddress", "payment": r"\b(payment|pay|gift card|credit card|certificate)",
    "passenger": r"\bpassenger", "compensation": r"\b(compensat|delay)", "human": r"\b(human|agent|representative)",
}
ORDER_STATUS = ("pending", "processed", "delivered", "cancelled")
CABIN = ("basic_economy", "economy", "business")
YES = re.compile(r"\b(yes|yeah|yep|confirm|go ahead|proceed|sure|correct)\b", re.I)
NO = re.compile(r"\b(no|don't|do not|not now|never mind|cancel that)\b", re.I)
NAMES = (["user_identified", "last_error", "n_errors", "n_steps", "write_done", "last_was_write",
          "user_yes", "user_no"]
         + [f"intent_{k}" for k in INTENTS] + [f"order_{s}" for s in ORDER_STATUS + ("other",)]
         + [f"cabin_{c}" for c in CABIN] + ["insurance_yes", "reservation_seen"])
DIM = len(NAMES)


class _State:
    def __init__(self):
        self.identified = self.write_done = self.last_was_write = False
        self.last_error = False
        self.n_errors = self.n_steps = 0
        self.order = self.cabin = None
        self.insurance = False
        self.reservation = False
        self.user_text = ""
        self.last_user = ""

    def see_user(self, text):
        self.user_text += " " + (text or "")
        self.last_user = text or ""

    def see_step(self, step):
        self.n_steps += 1
        self.last_was_write = False
        for tool, res in zip(step.tools, step.results_full):
            err = (res or "").startswith("Error")
            self.last_error = err
            self.n_errors += err
            name = tool.split(".", 1)[1]
            if not err and name in ("find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details"):
                self.identified = True
            if not err and not TOOL[tool].read_only and name != "transfer_to_human_agents":
                self.write_done = self.last_was_write = True
            try:
                obj = json.loads(res)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(obj, dict) and name == "get_order_details":
                st = str(obj.get("status", "")).lower()
                self.order = next((s for s in ORDER_STATUS if s in st), "other")
            if isinstance(obj, dict) and name == "get_reservation_details":
                self.reservation = True
                self.cabin = obj.get("cabin")
                self.insurance = str(obj.get("insurance", "")).lower() == "yes"

    def vector(self):
        v = [self.identified, self.last_error, math.log1p(self.n_errors), math.log1p(self.n_steps),
             self.write_done, self.last_was_write, bool(YES.search(self.last_user)), bool(NO.search(self.last_user))]
        text = self.user_text.lower()
        v += [bool(re.search(p, text)) for p in INTENTS.values()]
        v += [self.order == s for s in ORDER_STATUS + ("other",)]
        v += [self.cabin == c for c in CABIN]
        v += [self.insurance, self.reservation]
        return np.array(v, dtype=float)


def state_by_key(eps):
    out = {}
    for k, ep in enumerate(eps):
        st = _State()
        st.see_user(ep.first_user_text)
        out[(k, "start", 0)] = st.vector()
        for i, s in enumerate(ep.steps):
            if i > 0 and s.user_turn_before:
                st.see_user(s.user_text)
            out[(k, "call", i)] = st.vector()
            st.see_step(s)
            out[(k, "result", i)] = st.vector()
    return out
