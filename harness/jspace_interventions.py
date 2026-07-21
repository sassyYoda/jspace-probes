"""J-space interventions: concept swap and directional ablation on the residual stream.

j_v^l = normalize(J_l^T u_v). Hooks go on the decoder blocks, so the tensor we
edit at index l is the same one ActivationRecorder (and hence the fitted lens)
reads at layer l. Vector math in fp32, cast back to the model dtype at the
boundary.
"""

import torch


def find_blocks(model):
    for path in ("model.layers", "model.language_model.layers", "transformer.h"):
        obj = model
        try:
            for name in path.split("."):
                obj = getattr(obj, name)
            return obj
        except AttributeError:
            continue
    raise ValueError(f"no decoder blocks found in {type(model).__name__}")


def layer_band(lens, lo, hi):
    last = lens.source_layers[-1]
    return [l for l in lens.source_layers if lo <= l / last <= hi]


class JVecs:
    def __init__(self, lens, model, tokenizer):
        self.lens = lens
        self.tokenizer = tokenizer
        self._U = model.get_output_embeddings().weight
        self._cache = {}

    def token_id(self, word):
        # add_special_tokens=False: BOS-prepending tokenizers (gemma) put BOS
        # first, which silently made every j-vector the BOS unembedding row
        ids = self.tokenizer.encode(" " + word.strip(), add_special_tokens=False)
        for t in ids:
            if self.tokenizer.decode([t]).strip():
                return t
        return ids[0]

    def vec(self, layer, word=None, token_id=None):
        if token_id is None:
            token_id = self.token_id(word)
        key = (layer, token_id)
        if key not in self._cache:
            u = self._U[token_id].detach().float().cpu()
            j = self.lens.jacobians[layer].T @ u
            self._cache[key] = j / j.norm()
        return self._cache[key]

    def swap_pairs(self, word_from, word_to, layers):
        return {l: (self.vec(l, word_from), self.vec(l, word_to)) for l in layers}


def _qr_q(t):
    # torch.linalg.qr needs CPU LAPACK, absent in some ROCm builds; numpy always works
    import numpy as np
    q, _ = np.linalg.qr(t.detach().cpu().double().numpy())
    return torch.from_numpy(q).to(t.dtype)


def random_pairs(layers, real_pairs, d_model, seed):
    """Random orthonormal-frame pairs matching each real pair's mutual angle,
    so the applied delta norm per unit coefficient matches."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for l in layers:
        q = _qr_q(torch.randn(d_model, 2, generator=g))
        cos = (real_pairs[l][0] @ real_pairs[l][1]).clamp(-1, 1)
        sin = (1 - cos**2).sqrt()
        out[l] = (q[:, 0], cos * q[:, 0] + sin * q[:, 1])
    return out


class InterventionHooks:
    """Context manager registering forward hooks on decoder blocks.

    with InterventionHooks(blocks) as iv:
        iv.swap(pairs, positions=None, alpha=1.0)
        logits = model(ids).logits
    """

    def __init__(self, blocks):
        self.blocks = blocks
        self.ops = {}
        self._handles = {}
        self._active = False
        self._moved = {}

    def _dev(self, key, t, device):
        if key not in self._moved:
            self._moved[key] = t.to(device)
        return self._moved[key]

    def add(self, layer, op):
        self.ops.setdefault(layer, []).append(op)
        if self._active and layer not in self._handles:
            self._handles[layer] = self.blocks[layer].register_forward_hook(
                self._hook(layer)
            )

    def noop(self, layers):
        for l in layers:
            self.add(l, lambda h: h)

    def swap(self, pairs, positions=None, alpha=1.0):
        for l, (j_from, j_to) in pairs.items():
            self.add(l, self._swap_op(l, j_from, j_to, positions, alpha))

    def _swap_op(self, l, j_from, j_to, positions, alpha):
        def op(h):
            jf = self._dev(("f", l, id(j_from)), j_from, h.device)
            jt = self._dev(("t", l, id(j_to)), j_to, h.device)
            p = slice(*positions) if positions else slice(None)
            hs = h[:, p].float()
            c = hs @ jf
            hs = hs + alpha * c.unsqueeze(-1) * (jt - jf)
            out = h.clone()
            out[:, p] = hs.to(h.dtype)
            return out

        return op

    def ablate(self, dirs, positions=None):
        """dirs: {layer: [d, k] or [d]} — orthonormalized via QR before projecting out."""
        for l, v in dirs.items():
            v = v.reshape(v.shape[0], -1).float()
            q = _qr_q(v)
            self.add(l, self._ablate_op(l, q, positions))

    def _ablate_op(self, l, q, positions):
        def op(h):
            qd = self._dev(("q", l, id(q)), q, h.device)
            p = slice(*positions) if positions else slice(None)
            hs = h[:, p].float()
            hs = hs - (hs @ qd) @ qd.T
            out = h.clone()
            out[:, p] = hs.to(h.dtype)
            return out

        return op

    def _hook(self, layer):
        def fn(module, inputs, output):
            h = output if torch.is_tensor(output) else output[0]
            for op in self.ops.get(layer, []):
                h = op(h)
            return h if torch.is_tensor(output) else (h, *output[1:])

        return fn

    def __enter__(self):
        self._active = True
        for l in self.ops:
            if l not in self._handles:
                self._handles[l] = self.blocks[l].register_forward_hook(self._hook(l))
        return self

    def __exit__(self, *exc):
        for h in self._handles.values():
            h.remove()
        self._handles = {}
        self._active = False


@torch.no_grad()
def noop_check(model, input_ids, layers):
    """Hook placement test: an identity intervention must fire on every layer
    and reproduce baseline logits bit-exactly."""
    blocks = find_blocks(model)
    base = model(input_ids).logits
    fired = set()
    with InterventionHooks(blocks) as iv:
        for l in layers:
            iv.add(l, lambda h, l=l: fired.add(l) or h)
        hooked = model(input_ids).logits
    return fired == set(layers) and torch.equal(base, hooked)
