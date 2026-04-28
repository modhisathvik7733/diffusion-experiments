"""
Strict experiment: try YaRN RoPE scaling on DreamOn-v0-7B to extend its 2048
context window to 8192. Untested for diffusion — this script measures whether
the technique even works before we commit to anything.

What it tests:
  1. Loads the model with rope_scaling={"type": "yarn", "factor": 4.0} via
     trust_remote_code path. May error if custom modeling ignores the config.
  2. Runs short-context infilling (sanity: must still match ~88 Pass@1 quality).
  3. Runs long-context infilling at ~3k and ~5k tokens (in vanilla territory
     this is impossible). Output is qualitative — we just want to see if it
     produces COHERENT code or garbage.

Run: python scripts/08_yarn_extend_test.py
"""
import time
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODEL = "Dream-org/DreamOn-v0-7B"
EXTEND_FACTOR = 4.0
TARGET_MAX_POS = 8192  # 2048 * 4

print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0)}")

# Inspect config first so we know what we're modifying
cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
print(f"\noriginal max_position_embeddings: {getattr(cfg, 'max_position_embeddings', 'unset')}")
print(f"original rope_scaling:             {getattr(cfg, 'rope_scaling', None)}")
print(f"original rope_theta:               {getattr(cfg, 'rope_theta', None)}")

# Apply YaRN scaling.
cfg.rope_scaling = {
    "type": "yarn",
    "factor": EXTEND_FACTOR,
    "original_max_position_embeddings": cfg.max_position_embeddings,
}
cfg.max_position_embeddings = TARGET_MAX_POS

print(f"\nafter override: max_position_embeddings={cfg.max_position_embeddings}, "
      f"rope_scaling={cfg.rope_scaling}")

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
try:
    model = (
        AutoModel.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16,
                                  trust_remote_code=True)
        .to("cuda")
        .eval()
    )
    load_ok = True
except Exception as e:
    print(f"\n!!! Load with YaRN config FAILED: {type(e).__name__}: {e}")
    print("DreamOn's custom modeling likely doesn't honor standard rope_scaling.")
    print("Falling back to: load vanilla, manually bump max_position_embeddings.")
    cfg.rope_scaling = None
    cfg.max_position_embeddings = TARGET_MAX_POS
    model = (
        AutoModel.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16,
                                  trust_remote_code=True)
        .to("cuda")
        .eval()
    )
    load_ok = False
print(f"loaded in {time.time() - t0:.1f}s; "
      f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB; "
      f"YaRN_applied={load_ok}")


def encode_infill(prefix: str, suffix: str, n_mask: int):
    pids = [tokenizer.bos_token_id] + tokenizer.encode(prefix, add_special_tokens=False)
    mids = [tokenizer.mask_token_id] * n_mask
    sids = tokenizer.encode(suffix, add_special_tokens=False) + [tokenizer.eos_token_id]
    return torch.LongTensor([pids + mids + sids]).to("cuda"), len(pids)


GEN_KWARGS = dict(
    temperature=0.2, alg="entropy", alg_temp=0, top_p=0.9,
    max_new_tokens=64, return_dict_in_generate=True,
    output_history=False, number_transfer_tokens=1,
)


def long_python_blob(target_tokens: int) -> str:
    """Generate a fake but plausible Python file approximating target token count."""
    template = '''
def utility_fn_{i}(x: int, y: int) -> int:
    """Compute a transformed value combining x and y."""
    if x < 0 or y < 0:
        return 0
    z = (x * y) % 97
    while z > 50:
        z = (z * 3 + 1) // 2
    return z


class DataProcessor_{i}:
    def __init__(self, config: dict):
        self.config = config
        self.cache = {{}}

    def process(self, data: list) -> list:
        result = []
        for item in data:
            key = str(item)[:8]
            if key in self.cache:
                result.append(self.cache[key])
            else:
                v = utility_fn_{i}(len(key), item if isinstance(item, int) else 0)
                self.cache[key] = v
                result.append(v)
        return result
'''
    text = ""
    i = 0
    while True:
        text += template.format(i=i)
        if len(tokenizer.encode(text, add_special_tokens=False)) >= target_tokens:
            break
        i += 1
    return text


SUFFIX_FRAGMENT = "\n    return cache_value\n"


def run(name: str, prefix: str, suffix: str, n_mask: int):
    ids, plen = encode_infill(prefix, suffix, n_mask)
    total_input = ids.shape[1]
    print(f"\n{'-' * 70}\n{name}: input_tokens={total_input}\n{'-' * 70}")
    if total_input > TARGET_MAX_POS:
        print(f"SKIPPED — exceeds TARGET_MAX_POS={TARGET_MAX_POS}")
        return
    torch.cuda.synchronize()
    t = time.perf_counter()
    try:
        with torch.no_grad():
            out = model.diffusion_generate(ids, **GEN_KWARGS)
    except Exception as e:
        print(f"GENERATION FAILED: {type(e).__name__}: {e}")
        return
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) * 1000
    generated = tokenizer.decode(out.sequences[0][plen:plen + 64], skip_special_tokens=True)
    print(f"latency: {dt:.0f} ms")
    print(f"[generated]\n{generated.strip()[:400]}")


# ---- Test cases at increasing context length ----
short_prefix = "def add(a: int, b: int) -> int:\n    "
run("Short (~30 tokens)", short_prefix, "\n", 4)

med_prefix = (
    long_python_blob(1500)
    + "\n\ndef compute_score(items: list, threshold: float) -> float:\n"
    + "    cache_value = 0\n"
    + "    for item in items:\n        "
)
run("Medium (~1500 tokens, in original range)", med_prefix, SUFFIX_FRAGMENT, 8)

long_prefix = (
    long_python_blob(3000)
    + "\n\ndef compute_score(items: list, threshold: float) -> float:\n"
    + "    cache_value = 0\n"
    + "    for item in items:\n        "
)
run("Long (~3000 tokens, BEYOND original 2048)", long_prefix, SUFFIX_FRAGMENT, 8)

very_long_prefix = (
    long_python_blob(5500)
    + "\n\ndef compute_score(items: list, threshold: float) -> float:\n"
    + "    cache_value = 0\n"
    + "    for item in items:\n        "
)
run("Very long (~5500 tokens, ~2.7x original)", very_long_prefix, SUFFIX_FRAGMENT, 8)

print(f"\n{'=' * 70}")
print(f"YaRN_applied={load_ok}; peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print("Verdict criteria:")
print("  - Short stays coherent: baseline still works ✓")
print("  - Medium stays coherent: no regression in original range ✓")
print("  - Long produces sensible code: YaRN extension actually working ✓")
print("  - Long produces gibberish: YaRN didn't apply or DreamOn ignores it ✗")
