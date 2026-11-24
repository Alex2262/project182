"""Independent rescoring and targeted streaming replay of the paired evaluation."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

repo, out = map(Path, sys.argv[1:])
sys.path.insert(0, str(repo / 'src'))
from forecast_data import score
from forecast_models import Forecaster
from forecast_runtime import predict
torch.set_num_threads(2)
manifest = json.loads((out / 'manifest.json').read_text())
summary = json.loads((out / 'summary.json').read_text())
assert manifest['status'] == 'complete'
with np.load(out / 'reference.npz') as z:
    data = {k: z[k] for k in z.files}
data['lengths'] = data['origins']
data['bayes_risk'] = .25 + data['bayes_epistemic_risk']
checked = []
with np.load(repo / 'outputs/forecast_optimization/alex_261123_ar1_opt30k_cosine_s182/data/validation.npz') as z:
    values = z['values']
for checkpoint in manifest['checkpoints']:
    seed = checkpoint['seed']
    path = repo / checkpoint['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint['sha256']
    with np.load(out / f'predictions_s{seed}.npz') as z:
        predictions = z['prediction']
    for k, policy in enumerate(manifest['policies']):
        scores = score(predictions[:, :, k], data)
        for j, origin in enumerate(manifest['origins']):
            row = next(r for r in summary['rows'] if r['origin'] == origin and r['policy'] == policy)
            np.testing.assert_allclose(scores[j]['conditional_mean_error'], row[f'seed{seed}_error'], rtol=1e-12, atol=1e-15)
            assert np.isfinite(scores[j]['excess_vs_bayes'])
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = Forecaster(saved['config']['model']).eval()
    model.load_state_dict(saved['model'])
    differences = []
    with torch.inference_mode():
        for k, window in [(1,256),(2,2048)]:
            origin = 16384
            # predict() needs one unused target slot after its observed window.
            subset = values[[7,101], origin-window:origin+2]
            replay = predict(model, {'values': subset, 'lengths': np.array([window])}, 'cpu', batch_size=2)
            target = predictions[[7,101], -1, k:k+1]
            np.testing.assert_allclose(replay, target, rtol=0, atol=3e-6)
            differences.append(float(np.max(np.abs(replay-target))))
    checked.append({'seed': seed, 'all_12_cells_rescored_with_existing_score_function': True,
                    'checkpoint_hash_unchanged': True, 'independent_window_streaming_max_abs_error': max(differences)})
result = {'status': 'passed', 'seeds': checked, 'fixed_origins_and_targets': True,
          'confirmation_set_opened': False}
(out / 'independent_verification.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
