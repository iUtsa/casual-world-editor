"""Transformer lab: watch a real language model (GPT-2, 124M parameters, running on this Mac) process a sentence.

Everything shown is read out of the model during the forward pass — nothing is illustrative:
  * tokens and their embeddings,
  * every layer's attention pattern for every head (who looks at whom),
  * each head's measured behaviour (previous-token head, "attention sink" on the first token, self, broad),
  * the "logit lens": the model's best guess for the next word after each layer, showing the answer forming,
  * how far each token's representation moves per layer, and how much attention vs the MLP changes it,
  * the final next-word probabilities (with temperature).
Experiments: switch off attention heads (head ablation), hide a word from the others (masking), change the
temperature — each compared against the unmodified run, with the measured change explained.
"""
from __future__ import annotations

import logging
import math
import threading

import numpy as np

log = logging.getLogger(__name__)
_lock = threading.Lock()
_M: dict = {}
MAX_TOKENS = 24


def _load():
    with _lock:
        if _M:
            return _M
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
        m = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True, attn_implementation="eager").eval()
        _M.update(tok=tok, m=m, torch=torch)
        log.info("GPT-2 ready")
        return _M


def _forward(ids, head_mask=None, attn_mask=None):
    M = _load()
    torch, m = M["torch"], M["m"]
    caps: dict[str, list] = {"attn": [], "mlp": []}
    hooks = []
    if head_mask is not None:
        # exact head ablation: zero the switched-off heads' outputs right before the attention output projection
        dh = m.config.n_embd // m.config.n_head
        for l, blk in enumerate(m.transformer.h):
            off = [h for h in range(m.config.n_head) if float(head_mask[l, h]) == 0.0]
            if not off:
                continue
            def pre(mod, args, off=off):
                x = args[0].clone()
                for h in off:
                    x[..., h * dh:(h + 1) * dh] = 0
                return (x,) + tuple(args[1:])
            hooks.append(blk.attn.c_proj.register_forward_pre_hook(pre))
    for blk in m.transformer.h:
        hooks.append(blk.attn.register_forward_hook(lambda mod, i, o: caps["attn"].append(o[0].detach())))
        hooks.append(blk.mlp.register_forward_hook(lambda mod, i, o: caps["mlp"].append(o.detach())))
    try:
        with torch.no_grad():
            out = m(input_ids=ids, attention_mask=attn_mask, output_attentions=True, output_hidden_states=True)
    finally:
        for h in hooks:
            h.remove()
    return out, caps


def _top(logits, k, temperature=1.0):
    M = _load()
    torch, tok = M["torch"], M["tok"]
    p = torch.softmax(logits.detach().float() / max(1e-3, temperature), -1)
    v, i = torch.topk(p, k)
    return [{"token": tok.decode([int(j)]), "p": round(float(x), 5)} for x, j in zip(v, i)]


def run(text: str, ablate: list[list[int]] | None = None, mask: list[int] | None = None, temperature: float = 1.0) -> dict:
    M = _load()
    torch, tok, m = M["torch"], M["tok"], M["m"]
    text = (text or "").strip() or "The cat sat on the"
    ids = tok(text, return_tensors="pt").input_ids[:, :MAX_TOKENS]
    T = ids.shape[1]
    L, H = m.config.n_layer, m.config.n_head
    tokens = [tok.decode([int(t)]) for t in ids[0]]
    ablate = [[int(l), int(h)] for l, h in (ablate or []) if 0 <= int(l) < L and 0 <= int(h) < H]
    mask = sorted({int(x) for x in (mask or []) if 0 <= int(x) < T - 1})       # never hide the word being predicted from
    hm = None
    if ablate:
        hm = torch.ones(L, H)
        for l, h in ablate:
            hm[l, h] = 0.0
    am = None
    if mask:
        am = torch.ones(1, T, dtype=torch.long)
        for x in mask:
            am[0, x] = 0
    out, caps = _forward(ids, hm, am)
    base = None
    if hm is not None or am is not None or abs(temperature - 1.0) > 1e-3:
        base, _ = _forward(ids)

    att = torch.stack(out.attentions)[:, 0]                      # [L, H, T, T]
    hs = torch.stack(out.hidden_states)[:, 0]                    # [L+1, T, 768]
    # attention patterns (rounded) and head behaviour, measured
    A = att.numpy()
    heads = []
    for l in range(L):
        for h in range(H):
            a = A[l, h]
            prev = float(np.mean([a[t, t - 1] for t in range(1, T)])) if T > 1 else 0.0
            first = float(np.mean(a[1:, 0])) if T > 1 else 1.0
            selfp = float(np.mean(np.diag(a)))
            ent = float(np.mean([-(a[t, :t + 1] * np.log(a[t, :t + 1] + 1e-12)).sum() / max(1e-9, math.log(t + 1)) for t in range(1, T)])) if T > 1 else 0.0
            kind = ("previous word" if prev > 0.5 else "first token (attention sink)" if first > 0.6 else
                    "itself" if selfp > 0.5 else "spread out (averaging)" if ent > 0.75 else "specific words")
            heads.append({"layer": l, "head": h, "prev": round(prev, 3), "first": round(first, 3), "self": round(selfp, 3),
                          "entropy": round(ent, 3), "kind": kind, "off": [l, h] in ablate})
    # logit lens: the model's guess after every layer (final layer norm + unembedding applied to each layer's state)
    with torch.no_grad():
        last = hs[:, -1].clone()                                  # [L+1, 768]
        normed = torch.cat([m.transformer.ln_f(last[:L]), last[L:]], 0)
        lens_logits = normed @ m.transformer.wte.weight.T         # [L+1, vocab]
    lens = [{"layer": l, "top": _top(lens_logits[l], 3)} for l in range(L + 1)]
    final = _top(out.logits[0, -1], 10, temperature)
    # how each token's vector moves: 2-D PCA of all layers' states, and the size of each layer's update
    X = hs.numpy().reshape(-1, hs.shape[-1])
    Xc = X - X.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    P = (Xc @ vt[:2].T).reshape(L + 1, T, 2)
    P = P / (np.abs(P).max() + 1e-9)
    norms = np.linalg.norm(hs.numpy(), axis=-1)
    attn_upd = [float(c[0, -1].norm()) for c in caps["attn"]]
    mlp_upd = [float(c[0, -1].norm()) for c in caps["mlp"]]
    res = {
        "text": text, "tokens": tokens, "n_layers": L, "n_heads": H, "temperature": temperature,
        "attention": np.round(A, 3).tolist(), "heads": heads, "lens": lens, "next": final,
        "pca": np.round(P, 4).tolist(), "norms": np.round(norms, 2).tolist(),
        "attn_update": [round(x, 2) for x in attn_upd], "mlp_update": [round(x, 2) for x in mlp_upd],
        "ablate": ablate, "mask": mask, "params": "124M", "model": "GPT-2 small (OpenAI, 2019) — running on this Mac",
    }
    res["baseline_next"] = _top(base.logits[0, -1], 10) if base is not None else None
    res["story"] = _explain(res)
    return res


def _explain(r: dict) -> list[str]:
    toks, L = r["tokens"], r["n_layers"]
    s = []
    nxt = r["next"][0]
    s.append(f"The sentence is split into {len(toks)} tokens ({' · '.join(repr(t) for t in toks)}). Each becomes a list of 768 numbers (an embedding) "
             f"plus a position code, then flows up through {L} identical layers. Each layer has two parts: attention (each word gathers information "
             f"from the words before it) and a feed-forward network (each word's vector is transformed on its own).")
    kinds = {}
    for h in r["heads"]:
        kinds.setdefault(h["kind"], []).append(h)
    prev = sorted(kinds.get("previous word", []), key=lambda h: -h["prev"])
    sink = kinds.get("first token (attention sink)", [])
    if prev:
        h = prev[0]
        s.append(f"{len(prev)} of the {L * r['n_heads']} attention heads mostly look at the previous word — the strongest is layer {h['layer'] + 1}, "
                 f"head {h['head'] + 1}, putting {h['prev'] * 100:.0f}% of its attention there. That's how the model keeps track of word order and short phrases.")
    if sink:
        s.append(f"{len(sink)} heads park most of their attention on the first token. That's a known trick: when a head has nothing useful to look at, "
                 f"it 'rests' on the first token instead of adding noise (an attention sink).")
    first_right = next((x["layer"] for x in r["lens"] if x["top"][0]["token"] == nxt["token"]), None)
    if first_right is not None:
        s.append(f"Reading the model's mind layer by layer (the 'logit lens'): its final answer '{nxt['token'].strip()}' first becomes its top guess after "
                 f"layer {first_right}" + (" — right from the embeddings." if first_right == 0 else f"; before that it was guessing '{r['lens'][max(0, first_right - 1)]['top'][0]['token'].strip()}'.")
                 + " The answer is built up gradually, not looked up in one step.")
    a, mlp = r["attn_update"], r["mlp_update"]
    if a and mlp:
        la, lm = int(np.argmax(a)), int(np.argmax(mlp))
        s.append(f"For the last word, the biggest single change comes from layer {lm + 1}'s feed-forward network ({mlp[lm]:.0f} units) — MLPs are where much "
                 f"of the model's stored knowledge lives; attention's biggest contribution is in layer {la + 1} ({a[la]:.0f} units).")
    s.append(f"Final prediction: '{nxt['token'].strip()}' with {nxt['p'] * 100:.1f}% probability" +
             (f" (then '{r['next'][1]['token'].strip()}' {r['next'][1]['p'] * 100:.1f}%, '{r['next'][2]['token'].strip()}' {r['next'][2]['p'] * 100:.1f}%)." if len(r["next"]) > 2 else ".")
             + " It never 'knows' one answer — it outputs a probability for all 50,257 tokens in its vocabulary.")
    b = r.get("baseline_next")
    if b:
        bp = {x["token"]: x["p"] for x in b}
        top_b = b[0]
        now_p = next((x["p"] for x in r["next"] if x["token"] == top_b["token"]), None)
        what = []
        if r["ablate"]:
            what.append(f"switching off {len(r['ablate'])} head{'s' if len(r['ablate']) > 1 else ''} ("
                        + ", ".join(f"L{l + 1}.H{h + 1}" for l, h in r["ablate"][:6]) + ")")
        if r["mask"]:
            what.append("hiding " + ", ".join(repr(r["tokens"][i]) for i in r["mask"]) + " from the other words")
        if abs(r["temperature"] - 1) > 1e-3:
            what.append(f"temperature {r['temperature']:g}")
        change = (f"'{top_b['token'].strip()}' went from {top_b['p'] * 100:.1f}% to " + (f"{now_p * 100:.1f}%" if now_p is not None else "out of the top 10"))
        s.insert(0, f"Experiment — {' and '.join(what)}: {change}; the top prediction is now '{nxt['token'].strip()}'."
                 + (" Temperature divides the scores before the softmax: below 1 the favourite gets even more likely, above 1 the choices flatten out."
                    if abs(r["temperature"] - 1) > 1e-3 else ""))
    return s
