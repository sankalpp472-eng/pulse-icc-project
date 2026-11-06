"""
Tool dependency graph derived from tool schemas and sample results (no transition counts).

How it is built
  1. Input schemas: each tau-bench tool's get_info() (OpenAI function format, the same shape as an MCP
     tool's inputSchema), parsed with ast.literal_eval from data/tau-bench/schemas/ -- never imported.
  2. Entity types: parameter and output-field names are normalised to entity types
     (order_id, item_ids -> item_id, new_item_ids -> item_id, payments[].id -> payment_id, ...).
     One hand alias covers what naming cannot (ALIASES).
  3. Output types: the tools have no output schema, so each tool's outputs are typed from sample
     results (JSON keys; bare-string results by value shape, learned from typed input arguments).
  4. Edges: A -> B for every ID-like input type of B that A's outputs contain. If A returns it in a
     collection (list, or dict of records) the edge is a FAN-OUT: B may be called once per element.
  5. Entry tools: no required ID-like input (callable from what the user says: email, name+zip,
     cities, dates). Confirmation: the schema description asks for explicit user confirmation.

At runtime, ReadySet tracks which ID values a session knows (from user text and typed results):
a tool is ready when all its required ID-like inputs are known, and its fan-out estimate is the
number of known values of its key input type it has not been called with yet.
"""

import ast
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from pulse import paths as root_paths
from ...tools import DOMAINS, TOOL

SCHEMAS = root_paths.SCHEMAS
ALIASES = {"payment_method_id": "payment_id"}      # airline user records say payment_methods[].id; book_reservation says payment_id
PREFIXES = ("new_", "original_", "old_")


# ----------------------------------------------------------------- entity typing

def singular(word):
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("sses") or word.endswith("xes"):
        return word[:-2]
    return word[:-1] if word.endswith("s") else word


def etype(key, parent=None):
    """Field or parameter name -> entity type."""
    k = key.lower()
    for p in PREFIXES:
        if k.startswith(p):
            k = k[len(p):]
    if k == "id" and parent:
        k = singular(parent.lower()) + "_id"
    elif k.endswith("_ids"):
        k = k[:-1]
    return ALIASES.get(k, k)


def id_like(t):
    return t.endswith("_id") or t.endswith("_number")


def shape(value):
    """'#W1234567' -> '#A9999999', 'sara_doe_496' -> 'a_a_999': an ID's format (letter runs collapsed,
    digit runs kept at their length, so user ids and 'credit_card_1234567' payment ids differ)."""
    s = re.sub(r"[a-z]+", "a", str(value))
    s = re.sub(r"[A-Z]+", "A", s)
    return re.sub(r"[0-9]", "9", s)


# ----------------------------------------------------------------- schemas

def load_schemas():
    """{'retail.get_order_details': {'name', 'description', 'parameters'}} from the downloaded tool files."""
    out = {}
    for dom in DOMAINS:
        for f in sorted((SCHEMAS / dom).glob("*.py.txt")):
            for node in ast.walk(ast.parse(f.read_text())):
                if isinstance(node, ast.FunctionDef) and node.name == "get_info":
                    ret = next(n for n in ast.walk(node) if isinstance(n, ast.Return))
                    info = ast.literal_eval(ret.value)["function"]
                    out[f"{dom}.{info['name']}"] = info
    return out


def input_fields(params, parent=None, required=True):
    """Leaf parameters of a JSON schema: [(entity_type, is_list, required)]."""
    out = []
    req = set(params.get("required", []))
    for name, spec in params.get("properties", {}).items():
        r = required and name in req
        if spec.get("type") == "object":
            out += input_fields(spec, name, r)
        elif spec.get("type") == "array" and spec.get("items", {}).get("type") == "object":
            out += input_fields(spec["items"], name, r)
        else:
            out.append((etype(name, parent), spec.get("type") == "array", r))
    return out


def arg_values(args, parent=None):
    """Concrete call arguments -> [(entity_type, value)] for string leaves."""
    out = []
    if isinstance(args, dict):
        for k, v in args.items():
            if isinstance(v, (dict, list)):
                out += arg_values(v, k)
            elif isinstance(v, str):
                out.append((etype(k, parent), v))
    elif isinstance(args, list):
        for v in args:
            if isinstance(v, str) and parent:
                out.append((etype(parent), v))
            else:
                out += arg_values(v, parent)
    return out


def output_values(obj, parent=None, in_coll=False):
    """A parsed JSON result -> [(entity_type, value, in_collection)] for ID-like fields."""
    out = []
    if isinstance(obj, dict):
        records = len(obj) > 1 and all(isinstance(v, dict) for v in obj.values())   # dict keyed by id
        for k, v in obj.items():
            t = etype(k, parent)
            if isinstance(v, str) and id_like(t):
                out.append((t, v, in_coll))
            elif isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                t = t if id_like(t) else singular(k.lower()) + "_id"            # reservations: [...] -> reservation_id
                out += [(ALIASES.get(t, t), x, True) for x in v]
            elif isinstance(v, (dict, list)):
                out += output_values(v, k if not records else parent, in_coll or records or isinstance(v, list))
    elif isinstance(obj, list):
        for x in obj:
            out += output_values(x, parent, True)
    return out


def parse_result(content):
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None


# ----------------------------------------------------------------- the graph

@dataclass
class ToolGraph:
    schemas: dict
    inputs: dict = field(default_factory=dict)        # tool -> [(type, is_list, required)]
    produces: dict = field(default_factory=dict)      # tool -> {type: in_collection share}
    shapes: dict = field(default_factory=dict)        # type -> set of value shapes seen as arguments
    edges: dict = field(default_factory=dict)         # (A, B) -> {type: fan_out}
    confirm: set = field(default_factory=set)         # tools whose schema asks for user confirmation

    def required_ids(self, tool):
        return sorted({t for t, _, r in self.inputs[tool] if r and id_like(t)})

    def key_type(self, tool):
        ids = self.required_ids(tool)
        return ids[0] if ids else None

    def entry_tools(self, domain):
        return sorted(t for t in self.inputs if t.startswith(domain + ".") and not self.required_ids(t))

    def producers(self, etype_):
        return sorted(a for a, ts in self.produces.items() if etype_ in ts)

    def type_bare(self, value):
        """Type a bare-string result (e.g. find_user_id_by_email -> 'sara_doe_496') by its shape."""
        hits = [t for t, s in self.shapes.items() if shape(value) in s]
        return hits[0] if len(hits) == 1 else None

    def typed_outputs(self, content):
        obj = parse_result(content)
        if obj is None:
            t = self.type_bare(content.strip()) if content and len(content) < 80 else None
            return [(t, content.strip(), False)] if t else []
        return output_values(obj)


def build(records_by_domain):
    """records_by_domain: {domain: [raw tau-bench records]} (training split only)."""
    g = ToolGraph(load_schemas())
    for tool, info in g.schemas.items():
        g.inputs[tool] = input_fields(info["parameters"])
        if "confirmation" in info["description"].lower():
            g.confirm.add(tool)
    # value shapes of typed input arguments (to type bare-string results and user text)
    shapes = defaultdict(set)
    calls = []
    for dom, recs in records_by_domain.items():
        for rec in recs:
            names = {}
            for m in rec["traj"]:
                for c in m.get("tool_calls") or []:
                    args = c["function"]["arguments"]
                    args = json.loads(args) if isinstance(args, str) else args
                    names[c["id"]] = f"{dom}.{c['function']['name']}"
                    for t, v in arg_values(args):
                        if id_like(t):
                            shapes[t].add(shape(v))
                if m["role"] == "tool" and m.get("tool_call_id") in names:
                    calls.append((names[m["tool_call_id"]], m.get("content") or ""))
    g.shapes = dict(shapes)
    seen, coll = defaultdict(Counter), defaultdict(Counter)
    for tool, content in calls:
        for t, _, in_coll in g.typed_outputs(content):
            seen[tool][t] += 1
            coll[tool][t] += in_coll
    g.produces = {a: {t: coll[a][t] / n for t, n in ts.items()} for a, ts in seen.items()}
    for b in g.inputs:
        dom = b.split(".")[0]
        for t in g.required_ids(b):
            for a in g.producers(t):
                if a.startswith(dom + "."):
                    g.edges.setdefault((a, b), {})[t] = g.produces[a][t] > 0.5
    return g


# ----------------------------------------------------------------- runtime: what a session can call now

class ReadySet:
    """Per-session ID values known so far, and which tools they make callable."""

    def __init__(self, graph, domain):
        self.g, self.domain = graph, domain
        self.known = defaultdict(dict)          # type -> {value: source}
        self.used = defaultdict(set)            # tool -> argument values already passed to it
        self.tools = sorted(t for t in graph.inputs if t.startswith(domain + "."))

    def see_user(self, text):
        for tok in re.findall(r"[#\w]+", text or ""):
            for t, s in self.g.shapes.items():
                if shape(tok) in s and len(tok) >= 4:
                    self.known[t].setdefault(tok, "user")

    def see_result(self, tool, content):
        for t, v, _ in self.g.typed_outputs(content):
            self.known[t].setdefault(v, tool)

    def see_call(self, tool, args):
        for t, v in arg_values(args):
            self.used[tool].add(v)

    def ready(self):
        return [b for b in self.tools if all(self.known[t] for t in self.g.required_ids(b))]

    def fan_out(self, tool):
        """How many calls of `tool` the known-but-unused values of its key input suggest (>= 1).
        Only read-only tools with a single required ID fan out: writes are never run in parallel
        (tau.py keeps them as their own step), and a multi-ID input has no single key to count."""
        t = self.g.key_type(tool)
        if t is None or not TOOL[tool].read_only or len(self.g.required_ids(tool)) != 1:
            return 1
        return max(1, len([v for v in self.known[t] if v not in self.used[tool]]))


def fanout_by_key(g, eps, classes):
    """Fan-out vector (expected calls if the tool is called, >= 1) for every hint of tau episodes,
    keyed like predictors.contexts: (episode index, phase, step index). Uses only what the session
    has seen by that hint: user messages, the calls it made, and the results returned so far."""
    idx = {c: i for i, c in enumerate(classes)}
    out = {}
    for k, ep in enumerate(eps):
        rs = ReadySet(g, ep.domain)

        def vec():
            v = np.ones(len(classes))
            for t in rs.tools:
                v[idx[t]] = rs.fan_out(t)
            return v

        rs.see_user(ep.first_user_text)
        out[(k, "start", 0)] = vec()
        for i, s in enumerate(ep.steps):
            if i > 0 and s.user_turn_before:
                rs.see_user(s.user_text)
            for tool, args in zip(s.tools, s.args):
                rs.see_call(tool, args)
            out[(k, "call", i)] = vec()
            for tool, res in zip(s.tools, s.results_full):
                rs.see_result(tool, res)
            out[(k, "result", i)] = vec()
    return out


def ready_by_key(g, eps, classes):
    """Environment-state features for every hint, keyed like predictors.contexts:
    [ready flag per class, log2 fan-out per class]. Built only from user messages, the session's own
    calls and their results -- nothing about how the agent model writes."""
    fans = fanout_by_key(g, eps, classes)
    idx = {c: i for i, c in enumerate(classes)}
    out = {}
    for k, ep in enumerate(eps):
        rs = ReadySet(g, ep.domain)

        def vec(key):
            r = np.zeros(len(classes))
            r[[idx[t] for t in rs.ready()]] = 1.0
            return np.concatenate([r, np.log2(fans[key])])

        rs.see_user(ep.first_user_text)
        out[(k, "start", 0)] = vec((k, "start", 0))
        for i, s in enumerate(ep.steps):
            if i > 0 and s.user_turn_before:
                rs.see_user(s.user_text)
            for tool, args in zip(s.tools, s.args):
                rs.see_call(tool, args)
            out[(k, "call", i)] = vec((k, "call", i))
            for tool, res in zip(s.tools, s.results_full):
                rs.see_result(tool, res)
            out[(k, "result", i)] = vec((k, "result", i))
    return out


def attach(ctxs, feats):
    """Put per-hint feature vectors (from ready_by_key) on the contexts as Ctx.extra."""
    for c in ctxs:
        c.extra = feats[c.key]
    return ctxs


def build_from_tau(train_eps):
    """Build the graph from the training episodes' raw tau-bench records (same split as the predictors)."""
    from ...traces import tau
    keep = {(e.domain, e.task_id) for e in train_eps}
    recs = {d: [] for d in DOMAINS}
    for f, (dom, _) in tau.FILES.items():
        for rec in json.loads((tau.DATA / f).read_text()):
            if (dom, rec["task_id"]) in keep:
                recs[dom].append(rec)
    return build(recs)
