"""
Tool registry for the tau-bench airline and retail domains.

Each tool is its own Knative service. Tool names are prefixed with the domain
("airline.get_user_details"), because both domains have tools with the same name
(get_user_details, calculate, think, transfer_to_human_agents) backed by different databases.

Resource profiles default to assumptions (light database tools hold 128 MiB and cold-start in ~2.2 s;
search and catalog tools are a bit heavier). Set PULSE_TOOL_PROFILES to a measured profile file
(scripts/measure_coldstart.py writes data/tool_profiles_measured.json) to replace them with cluster
numbers before anything imports this module's tables.

The class space every predictor scores and every policy reads: CLASSES = all tools + END, with IDX,
V and the per-domain MASK. A predictor that only produces {tool: p} converts with vectors().
"""

import json
import math
import os
from dataclasses import dataclass, replace

import numpy as np

END = "END"


@dataclass(frozen=True)
class Tool:
    name: str            # "domain.tool"
    domain: str
    mem_mib: int         # resource request of the warm pod
    cold_s: float        # scale-from-zero to ready (pod start + MCP server init)
    exec_s: tuple        # (min, max) execution time of one call
    read_only: bool      # safe to run in parallel with other reads
    desc: str            # MCP tool description (used for zero-shot text similarity)


_LIGHT = (128, 2.2, (0.05, 0.10))
_SEARCH = (256, 2.6, (0.08, 0.15))

_SPECS = {
    "airline": {
        "get_user_details": (_LIGHT, True, "Get the details of a user, including their reservations, membership and payment methods."),
        "get_reservation_details": (_LIGHT, True, "Get the details of a reservation: flights, cabin, passengers, baggage, insurance."),
        "search_direct_flight": (_SEARCH, True, "Search direct flights between two cities on a specific date."),
        "search_onestop_flight": (_SEARCH, True, "Search flights with one stop between two cities on a specific date."),
        "list_all_airports": (_LIGHT, True, "List all airports and their cities."),
        "calculate": (_LIGHT, True, "Calculate the result of a mathematical expression, e.g. a price difference or refund."),
        "think": (_LIGHT, True, "Think about something without changing the database."),
        "book_reservation": (_LIGHT, False, "Book a new reservation with flights, passengers, payment and baggage."),
        "cancel_reservation": (_LIGHT, False, "Cancel the whole reservation and refund it."),
        "update_reservation_flights": (_LIGHT, False, "Change the flights or cabin of a reservation."),
        "update_reservation_passengers": (_LIGHT, False, "Update the passenger information of a reservation."),
        "update_reservation_baggages": (_LIGHT, False, "Update the number of checked bags of a reservation."),
        "send_certificate": (_LIGHT, False, "Send a certificate to a user as compensation."),
        "transfer_to_human_agents": (_LIGHT, False, "Transfer the user to a human agent with a summary of the issue."),
    },
    "retail": {
        "find_user_id_by_email": (_LIGHT, True, "Find a user id by email address."),
        "find_user_id_by_name_zip": (_LIGHT, True, "Find a user id by first name, last name and zip code."),
        "get_user_details": (_LIGHT, True, "Get the details of a user, including their orders, address and payment methods."),
        "get_order_details": (_LIGHT, True, "Get the status and items of an order: pending, processed, delivered or cancelled."),
        "get_product_details": (_SEARCH, True, "Get the inventory details of a product and its item variants."),
        "list_all_product_types": (_SEARCH, True, "List the name and product id of all product types."),
        "calculate": (_LIGHT, True, "Calculate the result of a mathematical expression, e.g. a price difference or refund."),
        "think": (_LIGHT, True, "Think about something without changing the database."),
        "cancel_pending_order": (_LIGHT, False, "Cancel a pending order and refund it."),
        "modify_pending_order_address": (_LIGHT, False, "Modify the shipping address of a pending order."),
        "modify_pending_order_items": (_LIGHT, False, "Modify items in a pending order to other variants of the same product."),
        "modify_pending_order_payment": (_LIGHT, False, "Modify the payment method of a pending order."),
        "modify_user_address": (_LIGHT, False, "Modify the default address of a user."),
        "return_delivered_order_items": (_LIGHT, False, "Return some items of a delivered order for a refund."),
        "exchange_delivered_order_items": (_LIGHT, False, "Exchange items in a delivered order for other variants of the same product."),
        "transfer_to_human_agents": (_LIGHT, False, "Transfer the user to a human agent with a summary of the issue."),
    },
}

TOOLS = [Tool(f"{dom}.{name}", dom, mem, cold, ex, ro, desc)
         for dom, specs in _SPECS.items()
         for name, ((mem, cold, ex), ro, desc) in specs.items()]


def _measured(tools, path):
    """Measured cold start (image cached), warm latency (+-20%) and memory request (working set rounded
    up to 32 MiB, at least 64) per tool; tools missing from the file keep their assumed profile."""
    prof = json.loads(open(path).read())
    out = []
    for t in tools:
        p = prof.get(t.name) or {}
        cold = p.get("cold_s") or t.cold_s
        warm = (p.get("warm_ms") or 0) / 1000
        ex = (0.8 * warm, 1.2 * warm) if warm else t.exec_s
        mem = max(64, math.ceil(p["memory_mib"] / 32) * 32) if p.get("memory_mib") else t.mem_mib
        out.append(replace(t, cold_s=cold, exec_s=ex, mem_mib=mem))
    return out


PROFILE_SOURCE = os.getenv("PULSE_TOOL_PROFILES") or "assumed"
if PROFILE_SOURCE != "assumed":
    TOOLS = _measured(TOOLS, PROFILE_SOURCE)
TOOL = {t.name: t for t in TOOLS}
TOOL_NAMES = [t.name for t in TOOLS]
DOMAINS = tuple(_SPECS)
TOTAL_MEM_MIB = sum(t.mem_mib for t in TOOLS)


MEM = {t: TOOL[t].mem_mib for t in TOOL_NAMES}
COLD = {t: TOOL[t].cold_s for t in TOOL_NAMES}


def domain_tools(domain):
    return {t.name for t in TOOLS if t.domain == domain}


CLASSES = TOOL_NAMES + [END]
IDX = {c: i for i, c in enumerate(CLASSES)}
V = len(CLASSES)
MASK = {}
for _d in DOMAINS:
    MASK[_d] = np.zeros(V, dtype=bool)
    MASK[_d][[IDX[t] for t in domain_tools(_d)] + [IDX[END]]] = True


def vectors(p1, p2=None, p_end=0.0):
    """{tool: p} for the next step (and the step after) -> (P1, P2) over CLASSES; unknown tools are
    dropped, END gets p_end."""
    out = []
    for p in (p1, p2 or {}):
        v = np.zeros(V)
        for t, x in p.items():
            if t in IDX:
                v[IDX[t]] = x
        out.append(v)
    out[0][IDX[END]] = p_end
    return out[0], out[1]
