"""Full-prefix oracle must release transient memory without invalidating its KV."""

from types import SimpleNamespace
import weakref
import torch
from scripts.ar_v02_nano_validation import reference


@torch.no_grad()
def test_reference_releases_memory_wrapper_cycles_and_keeps_returned_kv():
    refs = []

    class Memory:
        pass

    def build():
        memory = Memory()
        memory._clean_gen_kv = [torch.ones(3)]
        memory.init = lambda *args: memory
        memory.read_for_layer = lambda *args: memory
        memory.write_for_layer = lambda *args: memory
        refs.append(weakref.ref(memory))
        return memory

    model = SimpleNamespace(build_memory_state=build)

    def forward(*args, **kwargs):
        model.build_memory_state()
        return torch.zeros(1), torch.zeros(1)

    sampler = SimpleNamespace(model=model, forward=forward)
    out, kv = reference(sampler, None, None, SimpleNamespace(latent_start=1, latent_stop=2), None, None)
    assert model.build_memory_state is build
    assert refs[0]() is None  # No retained cyclic MemoryState after a completed call.
    torch.testing.assert_close(kv[0], torch.ones(3))
