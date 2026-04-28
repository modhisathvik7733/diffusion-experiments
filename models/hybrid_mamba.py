"""
Phase 2 placeholder. NOT IMPLEMENTED yet — do not import in training code.

Plan: bidirectional Mamba-2 block sized to drop-in replace a Qwen2DecoderLayer in
Open-dCoder 0.5B (hidden_size=896, num_layers=24). Output projection zero-init so
the layer starts as identity in the residual stream.

References:
- DiffuMamba (arXiv:2511.15927): bidirectional Mamba-2 for masked diffusion LMs.
- mamba-ssm: forward-only; bidirectionality via concat(forward, backward) scan.

Intended API:

    block = BidirectionalMamba2Block(hidden_size=896, ssm_state_size=128, ...)
    # block(x: [B, T, H]) -> [B, T, H], starts as identity (x + 0).

    patched = graft_mamba_into_qwen2(model, layer_indices=[8, 16])
"""
raise NotImplementedError(
    "models.hybrid_mamba is a Phase 2 stub. Reproduce Phase 1 baseline first."
)
