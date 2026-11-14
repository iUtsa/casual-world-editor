"""Causal explanation of a simulated intervention.

Input is evidence only: the intervention, the measured support graph before
and after (with load shares), the event log, the causal edges the simulator
attributed, per-object outcomes and the modelling assumptions. Two writers
share one output schema:
  * Claude (when credentials exist) for fluent, physics-literate prose that
    must cite event ids as evidence;
  * a deterministic writer that renders the same causal graph as text, so the
    app is fully functional offline and the two can be compared.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from . import config

log = logging.getLogger(__name__)

SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "causal_chain": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cause": {"type": "string"},
                    "effect": {"type": "string"},
                    "mechanism": {"type": "string"},
                    "principle": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["cause", "effect", "mechanism", "principle", "evidence"],
                "additionalProperties": False,
            },
        },
        "unaffected": {"type": "string"},
        "graph_vs_physics": {"type": "string"},
        "counterfactual": {"type": "string"},
        "confidence": {"type": "number"},
        "confidence_reason": {"type": "string"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["headline", "causal_chain", "unaffected", "graph_vs_physics", "counterfactual",
                 "confidence", "confidence_reason", "assumptions"],
    "additionalProperties": False,
}

SYSTEM = (
    "You explain the outcome of a rigid-body physics simulation of a real tabletop scene that was reconstructed "
    "from a single camera frame. You receive structured evidence: the user's intervention, measured support graphs "
    "(which object rests on which, with the share of its weight each supporter carries), a timestamped event log, "
    "causal edges attributed by the simulator, per-object outcomes and modelling assumptions.\n"
    "Rules: every causal step must cite event ids from the log as evidence. Do not invent events, forces or objects "
    "that are not in the evidence. Use object labels, not ids. Name the physical principle at work (loss of support "
    "/ unbalanced gravity, torque about a pivot, friction limiting sliding, momentum transfer in a collision, etc.). "
    "In graph_vs_physics, compare what the static support graph predicted (predicted_dependents) with what the "
    "dynamics actually did. In counterfactual, give one concrete, testable 'what if' the user could try next in this "
    "scene. Confidence (0-1) must reflect reconstruction uncertainty (primitive shapes, estimated masses, monocular "
    "depth) and how decisive the dynamics were. Keep every string short and concrete."
)


def _label(ids_to_label: dict[str, str], x: str | None) -> str:
    if x is None:
        return "-"
    if x == "intervention":
        return "your intervention"
    return ids_to_label.get(x, x)


def evidence_packet(scene, sim: dict, seg_index: int) -> dict[str, Any]:
    seg = sim["segments"][seg_index]
    labels = {o.id: o.label for o in scene.objects}
    labels.update(sim.get("labels", {}))
    labels.update({"table": "table", "floor": "floor"})
    by_id = {o.id: o for o in scene.objects}

    def graph(g):
        return {labels[k]: [{"on": labels.get(e["on"], e["on"]), "share": e["share"], "force_N": e["force_N"]}
                            for e in v] for k, v in g.items() if k in labels}

    iv = seg["intervention"]
    return {
        "intervention": {"action": iv["type"], "target": labels.get(iv["target"], iv["target"]), "params": iv["params"]},
        "earlier_interventions": [
            {"action": s["intervention"]["type"], "target": labels.get(s["intervention"]["target"], s["intervention"]["target"])}
            for s in sim["segments"][:seg_index]],
        "summoned_objects": {v["label"]: {"asset": v["asset"], "appeared_at_s": v["t"]} for v in sim.get("spawned", {}).values()},
        "objects": {labels[o.id]: {"shape": o.shape, "size_cm": [round(v * 100, 1) for v in o.size],
                                   "mass_kg": o.mass, "mass_source": o.mass_source, "friction": o.friction,
                                   "static": o.static} for o in scene.objects},
        "support_before": graph(seg["support_before"]),
        "support_after": graph(seg["support_after"]),
        "predicted_dependents": [labels.get(x, x) for x in seg["predicted_dependents"]],
        "events": [{**e, "subject": labels.get(e["subject"], e["subject"]), "other": labels.get(e["other"], e["other"])}
                   for e in seg["events"]],
        "causal_edges": [{"cause": _label(labels, c["cause"]), "effect": labels.get(c["effect"]), "kind": c["kind"],
                          "t": c["t"], "evidence": c["evidence"]} for c in seg["causal_edges"]],
        "outcomes": {v["label"]: {k: v[k] for k in ("displacement_m", "drop_m", "rotation_deg", "max_speed_mps",
                                                     "toppled", "fell_off_table", "moved")}
                     for v in seg["summary"].values()},
        "gravity_mps2": sim["gravity"],
        "assumptions": sorted({n for o in scene.objects for n in o.notes} |
                              {"collision shapes are boxes/cylinders fitted to the visible surface",
                               "depth is monocular (scale may be off unless calibrated)"}),
    }


def explain(scene, sim: dict, seg_index: int, use_claude: bool = True) -> dict:
    packet = evidence_packet(scene, sim, seg_index)
    if use_claude and config.CLAUDE_ENABLED:
        out = _claude(packet)
        if out is not None:
            out["source"] = f"claude ({config.CLAUDE_MODEL})"
            out["evidence_packet"] = packet
            return out
    out = deterministic(packet)
    out["source"] = "deterministic (from causal graph)"
    out["evidence_packet"] = packet
    return out


def _claude(packet: dict) -> dict | None:
    import anthropic
    from . import llm
    client = llm.client()
    if client is None:
        return None
    try:
        resp = client.beta.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=8000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM,
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": "Evidence:\n" + json.dumps(packet, indent=1)}],
        )
    except anthropic.AuthenticationError as e:
        log.warning("Claude explanation skipped (auth): %s", e)
        return None
    except anthropic.APIConnectionError as e:
        log.warning("Claude explanation skipped (network): %s", e)
        return None
    except TypeError as e:          # credentials vanished between check and call
        log.warning("Claude call skipped (no credentials): %s", e)
        return None
    except anthropic.APIStatusError as e:
        log.warning("Claude explanation failed (%s): %s", e.status_code, e)
        return None
    if resp.stop_reason == "refusal":
        return None
    text = next((b.text for b in resp.content if b.type == "text"), None)
    try:
        return json.loads(text) if text else None
    except json.JSONDecodeError:
        return None


KIND_TEXT = {
    "support_removed": ("lost its support when {cause_obj} was removed", "Unsupported weight: with the normal force gone, gravity is unbalanced"),
    "lost_support": ("lost support from {cause}", "Unsupported weight: the normal force from below vanished"),
    "pushed": ("was pushed directly", "Impulse: applied force changes momentum (F dt = m dv)"),
    "struck_by": ("was struck by {cause}", "Momentum transfer in a collision"),
    "pushed_by": ("was pushed by {cause}", "Contact force from a moving neighbour"),
    "carried_by": ("was carried along by {cause}, which it rests on", "Static friction couples an object to its moving support"),
    "gravity_change": ("responded to the new gravity", "Weight scales with g"),
    "driven": ("was driven by its motors", "Wheel torque against friction propels the vehicle"),
    "launched": ("was launched", "Initial momentum, then friction and rolling resistance"),
    "released": ("was released and fell", "Free fall under gravity until contact"),
}


def deterministic(p: dict) -> dict:
    iv = p["intervention"]
    chain = []
    for c in p["causal_edges"]:
        kind = c["kind"]
        if kind.startswith("property_change"):
            mech, prin = "started moving after its properties changed", "Force balance changed (friction / inertia)"
        else:
            mech, prin = KIND_TEXT.get(kind, ("started moving (cause unclear)", "Unattributed"))
            mech = mech.format(cause=c["cause"], cause_obj=iv["target"])
        chain.append({"cause": c["cause"], "effect": c["effect"], "mechanism": f"{c['effect']} {mech} at t={c['t']:.2f}s",
                      "principle": prin, "evidence": c["evidence"]})
    seen_pairs = set()
    for e in p["events"]:
        pair = frozenset((e["subject"], e["other"]))
        if e["type"] == "impact" and e.get("speed", 0) > 0.3 and pair not in seen_pairs and "floor" not in pair:
            seen_pairs.add(pair)
            chain.append({"cause": e["subject"], "effect": e["other"],
                          "mechanism": f"{e['subject']} hit {e['other']} at {e['speed']:.2f} m/s (t={e['t']:.2f}s)",
                          "principle": "Collision: momentum exchanged through contact", "evidence": [e["id"]]})
    outcomes = p["outcomes"]
    moved = [k for k, v in outcomes.items() if v["moved"]]
    still = [k for k, v in outcomes.items() if not v["moved"]]
    fell = [k for k, v in outcomes.items() if v["fell_off_table"]]
    toppled = [k for k, v in outcomes.items() if v["toppled"]]
    verb = {"remove": "Removing", "push": "Pushing", "set_mass": "Changing the mass of", "set_friction": "Changing the friction of",
            "set_static": "Pinning", "set_gravity": "Changing gravity", "spawn": "Placing", "drive": "Driving",
            "launch": "Launching"}[iv["action"]]
    tgt = f" {iv['target']}" if iv["target"] else ""
    if moved:
        headline = f"{verb}{tgt} set {len(moved)} object(s) in motion: {', '.join(moved)}."
    else:
        headline = f"{verb}{tgt} changed nothing visible: every object stayed in equilibrium."
    if fell:
        headline += f" {', '.join(fell)} fell off the table."
    elif toppled:
        headline += f" {', '.join(toppled)} toppled."
    pred = set(p["predicted_dependents"])
    act = {c["effect"] for c in p["causal_edges"]} - {iv["target"]}
    if not pred and not act:
        gvp = "The support graph predicted no dependents and none moved."
    elif pred == act:
        gvp = f"The support graph predicted exactly what moved ({', '.join(sorted(pred))})."
    else:
        parts = []
        if act - pred:
            parts.append(f"dynamics also moved {', '.join(sorted(act - pred))} (not resting on the target: collisions or shared supports)")
        if pred - act:
            parts.append(f"{', '.join(sorted(pred - act))} rested on the target but stayed put (other supports carried the load)")
        gvp = "Support graph vs physics: " + "; ".join(parts) + "."
    target = iv["target"]
    if iv["action"] == "remove" and moved:
        cf = f"Make {moved[0]} heavier or raise its friction, then remove {target} again to see whether it still falls."
    elif iv["action"] == "push":
        cf = f"Push {target} with half the strength, or lower its friction, and compare how far it travels."
    else:
        cf = f"Remove the object that carries the largest share of {moved[0] if moved else 'the stack'}'s weight."
    n_unstable = sum(1 for a in p["assumptions"] if "unstable" in a)
    conf = 0.75 - 0.05 * n_unstable - (0.1 if "monocular" in " ".join(p["assumptions"]) else 0)
    return {
        "headline": headline,
        "causal_chain": chain,
        "unaffected": (", ".join(still) + " did not move.") if still else "Every remaining object moved.",
        "graph_vs_physics": gvp,
        "counterfactual": cf,
        "confidence": round(max(0.3, conf), 2),
        "confidence_reason": "Rule-based summary of the simulator's own causal attribution; quantitative outcomes depend on the reconstructed shapes and masses.",
        "assumptions": p["assumptions"],
    }
