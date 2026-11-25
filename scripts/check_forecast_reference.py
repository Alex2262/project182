"""CPU-check the HF adapter without importing CUDA-only FLA kernels."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
import yaml
from forecast_reference import ReferenceForecaster
from forecast_runtime import Forecaster as RuntimeForecaster


def main():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / 'src/conf/forecast_reference_benchmark.yaml').read_text())['model']
    cfg.update(family='transformer_nope', precision='float32')
    torch.manual_seed(182)
    model = ReferenceForecaster(cfg).eval()
    assert isinstance(RuntimeForecaster(cfg), ReferenceForecaster)
    x = torch.randn(2, 48, 2)
    prediction = model(x)
    assert prediction.shape == (2, 48)
    changed = x.clone(); changed[:, 24:] += 5
    torch.testing.assert_close(model(changed)[:, :24], prediction[:, :24], atol=1e-6, rtol=0)
    with torch.no_grad():
        a, state = model(x[:, :24], return_state=True)
        b, _ = model(x[:, 24:], state, return_state=True)
    torch.testing.assert_close(torch.cat([a, b], 1), prediction, atol=1e-6, rtol=0)
    cos, sin = model.backbone.rotary_emb(x, torch.tensor([[0, 100000]]))
    assert (cos == 1).all() and (sin == 0).all()
    prediction.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print({'status': 'passed', **model.description()})


if __name__ == '__main__':
    main()
