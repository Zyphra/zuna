import os
import torch

try:
    from torch.distributed.tensor import DTensor, distribute_tensor
except Exception:  # older torch
    try:
        from torch.distributed._tensor import DTensor, distribute_tensor
    except Exception:
        DTensor = ()  # isinstance(x, ()) is always False -> treated as plain tensor
        distribute_tensor = None


class EMA:
    """Exponential moving average of model parameters, stored in fp32.

    Args:
        model:   the (already-initialized, post-checkpoint-load) training model.
        decay:   target EMA decay (0.9999 recommended for long runs).
        warmup:  if True, ramp decay up early via min(decay, (1+t)/(10+t)) so
                 the noisy first steps don't dominate the average.
        skip_substrings: param-name fragments to exclude from EMA (rarely needed).
    """

    def __init__(self, model, decay: float = 0.9999, warmup: bool = True,
                 skip_substrings=()):
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.num_updates = 0
        self.skip_substrings = tuple(skip_substrings)

        # fp32 shadow of every FLOATING-POINT param (skip int/bool buffers/params).
        # Keyed by NORMALIZED name (see _norm) so it matches compiled targets later.
        self.shadow = {}
        for name, p in model.named_parameters():
            if not p.is_floating_point():
                continue
            if any(s in name for s in self.skip_substrings):
                continue
            self.shadow[self._norm(name)] = p.detach().clone().float()  # fp32; keeps DTensor layout
        self._n_tracked = len(self.shadow)

    @staticmethod
    def _norm(name: str) -> str:
        # torch.compile wraps a MODULE as OptimizedModule and prefixes its params
        # with "_orig_mod.". Strip it so a shadow built from the (clean-named)
        # training model matches a target whose .encoder/.sample were module-compiled.
        return name.replace("._orig_mod.", ".").replace("_orig_mod.", "")

    def _effective_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))

    @torch.no_grad()
    def update(self, model):
        """Call ONCE per optimizer step, AFTER optimizer.step()."""
        self.num_updates += 1
        d = self._effective_decay()
        for name, p in model.named_parameters():
            s = self.shadow.get(self._norm(name))
            if s is None:
                continue
            # fp32 accumulation. For no_shard, p is a replicated DTensor and these
            # ops stay local & consistent across ranks.
            s.mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model):
        """Load EMA weights INTO `model` (cast back to the model's dtype).
        Matches by NORMALIZED name, so it works whether or not the target's
        submodules have been module-compiled (_orig_mod prefix). Safe on a
        throwaway eval/inference copy; never call on the live training model."""
        tgt = {self._norm(n): p for n, p in model.named_parameters()}
        missing = []
        for name, s in self.shadow.items():
            p = tgt.get(name)  # name already normalized at shadow-build time
            if p is None:
                missing.append(name)
                continue
            # Source (shadow) and target may disagree on DTensor-ness: the shadow
            # keeps the sharded layout of the training model, while model_inference
            # is a plain-tensor replica (built via convert_dtensor_model_to_tensor_model,
            # which maps each DTensor to its LOCAL shard). Reduce both sides to their
            # local tensor so copy_ never mixes DTensor and plain Tensor. Shard shapes
            # match by the same to_local() mapping the replica build relies on.
            dst = p.data
            src = s
            if isinstance(dst, DTensor):
                dst = dst.to_local()
            if isinstance(src, DTensor):
                src = src.to_local()
            dst.copy_(src.to(dst.dtype))
        if missing:
            raise KeyError(
                f"[EMA] {len(missing)} params not found on target model "
                f"(e.g. {missing[:3]}). Names normalized for _orig_mod; check that "
                f"the target is the same architecture as the EMA source model."
            )

    # ---- checkpointing (saved as a sidecar ema.pt next to the DCP checkpoint) ----
    def _full_cpu(self, t):
        if isinstance(t, DTensor):
            t = t.full_tensor()  # gather to full (no-op-ish under replicate)
        return t.detach().to("cpu", torch.float32)

    def cpu_state_dict(self):
        """Plain (non-DTensor) fp32 state for torch.save on the master rank.
        Shadow keys are already normalized (no _orig_mod)."""
        return {
            "decay": self.decay,
            "warmup": self.warmup,
            "num_updates": self.num_updates,
            "shadow": {k: self._full_cpu(v) for k, v in self.shadow.items()},
        }

    @torch.no_grad()
    def load_cpu_state_dict(self, sd):
        self.decay = sd.get("decay", self.decay)
        self.warmup = sd.get("warmup", self.warmup)
        self.num_updates = sd.get("num_updates", 0)
        loaded = sd["shadow"]
        for k, s in self.shadow.items():
            src = loaded.get(k, loaded.get(self._norm(k)))  # tolerate legacy keys
            if src is None:
                continue
            if isinstance(s, DTensor):
                # `src` is the FULL (gathered) fp32 tensor saved by _full_cpu.
                # Re-distribute it to the shadow's mesh/placements so copy_ stays
                # DTensor<-DTensor. Correct for both replicate (no_shard) and shard.
                src = src.to(s.device, s.dtype)
                s.copy_(distribute_tensor(src, s.device_mesh, s.placements))
            else:
                s.copy_(src.to(s.dtype if isinstance(s, torch.Tensor) else torch.float32))

    def save(self, ckpt_step_dir: str):
        """Save EMA to <ckpt_step_dir>/ema.pt. Call on MASTER rank only."""
        os.makedirs(ckpt_step_dir, exist_ok=True)
        torch.save(self.cpu_state_dict(), os.path.join(ckpt_step_dir, "ema.pt"))

    def maybe_load(self, ckpt_step_dir: str) -> bool:
        path = os.path.join(ckpt_step_dir, "ema.pt")
        if not os.path.exists(path):
            return False
        self.load_cpu_state_dict(torch.load(path, map_location="cpu"))
        return True