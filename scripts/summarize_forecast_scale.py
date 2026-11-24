"""Recompute and visualize the approved three-seed, 10k AR(1) batch."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GROUPS = [('gated_deltanet', 'short'), ('gated_deltanet', 'mixed'),
          ('transformer_nope', 'short'), ('transformer_nope', 'mixed')]
LABELS = ['GDN · train 64', 'GDN · train 64/256',
          'Transformer · train 64', 'Transformer · train 64/256']
COLORS = ['#146d9c', '#146d9c', '#cc642d', '#cc642d']
STYLES = ['-', '--', '-', '--']
STEPS = [2000, 5000, 10000]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def summarize(raw, output, manifest_path, repo):
    sys.path.insert(0, str(repo / 'src'))
    from forecast_data import score
    output.mkdir(parents=True, exist_ok=True)
    for folder in ['configs', 'predictions']:
        (output / folder).mkdir(exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest['runs']) == 12
    runs, all_rows, common = [], [], None
    for item in manifest['runs']:
        assert item['status'] == 'finished'
        directory = raw / item['run_name']
        config = yaml.safe_load((directory / 'config.yaml').read_text())
        assert config['training']['steps'] == 10000
        assert config['training']['total_training_tokens'] == 20480000
        assert config['training']['validation_steps'] == STEPS
        assert config['training']['seed'] == item['seed']
        assert config['model']['family'] == item['family']
        assert config['training']['regime'] == item['regime']
        assert not (directory / 'data/confirmation.npz').exists()
        with np.load(directory / 'data/validation.npz') as saved:
            data = {k: saved[k] for k in saved.files}
        if common is None:
            common = data
        else:
            for key in ['values', 'theta', 'latent_mean', 'target', 'spec', 'lengths']:
                np.testing.assert_array_equal(common[key], data[key])
            for key in ['bayes_mean', 'bayes_risk', 'bayes_epistemic_risk']:
                np.testing.assert_allclose(common[key], data[key], rtol=1e-13, atol=1e-13)
        history = json.loads((directory / 'training.json').read_text())
        assert history[-1]['step'] == 10000
        assert [h['step'] for h in history] == list(range(25, 10001, 25))
        predictions, results = {}, {}
        for step in STEPS:
            with np.load(directory / f'validation_predictions_step_{step}.npz') as saved:
                predictions[step] = saved['prediction']
            results[step] = score(predictions[step], common)
            recorded = json.loads((directory / f'validation_step_{step}.json').read_text())
            assert len(recorded) == len(results[step]) == 9
            for calculated, stored in zip(results[step], recorded):
                for key in ['conditional_mean_error', 'excess_vs_bayes', 'observed_mse', 'expected_predictive_mse']:
                    np.testing.assert_allclose(calculated[key], stored[key], rtol=1e-10, atol=1e-12)
                all_rows.append({'run_name': item['run_name'], 'family': item['family'],
                                 'regime': item['regime'], 'training_seed': item['seed'],
                                 'training_steps': step, **calculated})
            shutil.copyfile(directory / f'validation_predictions_step_{step}.npz',
                            output / 'predictions' / f'{item["run_name"]}_step_{step}.npz')
        shutil.copyfile(directory / 'config.yaml', output / 'configs' / f'{item["run_name"]}.yaml')
        item['config_file'] = f'configs/{item["run_name"]}.yaml'
        item.pop('log_file', None)
        item.pop('launcher_pid', None)
        item['final_checkpoint_sha256'] = hashlib.sha256((directory / 'state.pt').read_bytes()).hexdigest()
        runs.append({**item, 'config': config, 'history': history, 'results': results,
                     'predictions': predictions})
    with (output / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    np.savez_compressed(output / 'validation_reference.npz', **{k:v for k,v in common.items() if k != 'values'})
    lengths = common['lengths']
    ix64, ix4096 = lengths.tolist().index(64), lengths.tolist().index(4096)
    bayes = common['bayes_epistemic_risk'].mean(0)
    sigma2 = json.loads(str(common['spec']))['noise_std'] ** 2
    groups, group_summary, run_summary, contrasts = [], [], [], []
    for key, label in zip(GROUPS, LABELS):
        members = sorted([r for r in runs if (r['family'], r['regime']) == key], key=lambda r:r['seed'])
        assert [r['seed'] for r in members] == [182,183,184]
        groups.append(members)
        for step in STEPS:
            for index in [ix64, ix4096]:
                means = np.array([r['results'][step][index]['conditional_mean_error'] for r in members])
                gap = np.array([r['results'][step][index]['excess_vs_bayes'] for r in members])
                group_summary.append({'family': key[0], 'regime': key[1], 'label': label,
                                      'training_steps': step, 'length': int(lengths[index]),
                                      'conditional_mean_error': float(means.mean()),
                                      'conditional_mean_error_seed_sd': float(means.std(ddof=1)),
                                      'per_seed_errors': means.tolist(),
                                      'excess_vs_bayes': float(gap.mean()),
                                      'excess_vs_bayes_seed_sd': float(gap.std(ddof=1)),
                                      'expected_predictive_mse': float(sigma2 + means.mean())})
        for run in members:
            e64 = run['results'][10000][ix64]['conditional_mean_error']
            e4096 = run['results'][10000][ix4096]['conditional_mean_error']
            run_summary.append({'run_name':run['run_name'], 'family':key[0], 'regime':key[1],
                                'seed':run['seed'], 'error_64':e64, 'error_4096':e4096,
                                'history_error_reduction_percent':100*(1-e4096/e64),
                                'long_error_by_step': {step:run['results'][step][ix4096]['conditional_mean_error'] for step in STEPS}})
    for family in ['gated_deltanet', 'transformer_nope']:
        for step in STEPS:
            differences = []
            per_seed = []
            for seed in [182,183,184]:
                short = next(r for r in runs if r['family']==family and r['regime']=='short' and r['seed']==seed)
                mixed = next(r for r in runs if r['family']==family and r['regime']=='mixed' and r['seed']==seed)
                s = (short['predictions'][step][:, ix4096]-common['latent_mean'][:, ix4096])**2
                m = (mixed['predictions'][step][:, ix4096]-common['latent_mean'][:, ix4096])**2
                difference = m-s
                differences.append(float(difference.mean()))
                per_seed.append({'seed':seed, 'paired_error_difference':float(difference.mean()),
                                 'paired_series_se':float(difference.std(ddof=1)/np.sqrt(len(difference))),
                                 'error_reduction_percent':float(100*(1-m.mean()/s.mean()))})
            contrasts.append({'family':family, 'training_steps':step, 'length':4096,
                              'comparison':'mixed minus short', 'mean_difference':float(np.mean(differences)),
                              'difference_seed_sd':float(np.std(differences,ddof=1)), 'per_seed':per_seed})

    fig, axes = plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    for members, label, color, style in zip(groups,LABELS,COLORS,STYLES):
        for axis, metric in zip(axes,['conditional_mean_error','excess_vs_bayes','expected_predictive_mse']):
            values = np.array([[row[metric] for row in r['results'][10000]] for r in members])
            mean, sd = values.mean(0), values.std(0,ddof=1)
            axis.plot(lengths,mean,style,color=color,marker='o',markersize=3,label=label)
            axis.fill_between(lengths,np.maximum(mean-sd,1e-8),mean+sd,color=color,alpha=.12)
    axes[0].plot(lengths,bayes,'k:',lw=2,label='Exact Bayes reference')
    ridge=((common['ridge_mean']-common['latent_mean'])**2).mean(0)
    axes[0].plot(lengths,ridge,color='#777777',linestyle='-.',label='Online ridge reference')
    axes[2].plot(lengths,sigma2+bayes,'k:',lw=2)
    axes[2].axhline(sigma2,color='#777777',linestyle='-.')
    axes[0].set_yscale('log'); axes[1].set_yscale('log')
    for axis,title,ylabel in zip(axes,['Reducible forecast error','Gap to available Bayes forecast','Total expected forecast error'],
                                ['MSE relative to a × current value','MSE relative to Bayes mean','Expected next-value MSE']):
        axis.set(xscale='log',xlabel='Observed transitions',ylabel=ylabel,title=title)
        axis.set_xscale('log',base=2)
        axis.axvline(64,color='gray',alpha=.4,lw=1)
        axis.axvline(256,color='gray',alpha=.4,lw=1,linestyle='--')
        axis.grid(alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('AR(1) streaming · 10,000 updates · three training seeds\nLines: seed means; bands: ±1 seed SD · same 512 validation series',fontsize=12)
    for ext in ['png','svg']:
        fig.savefig(output/f'forecast_curves.{ext}',dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for members,label,color,style in zip(groups,LABELS,COLORS,STYLES):
        for axis,index in zip(axes,[ix64,ix4096]):
            values=np.array([[r['results'][step][index]['conditional_mean_error'] for step in STEPS] for r in members])
            mean,sd=values.mean(0),values.std(0,ddof=1)
            for seed_values in values:
                axis.plot(STEPS,seed_values,style,color=color,alpha=.2,lw=.8)
            axis.plot(STEPS,mean,style,color=color,marker='o',label=label)
            axis.fill_between(STEPS,np.maximum(mean-sd,1e-8),mean+sd,color=color,alpha=.1)
    for axis,index in zip(axes,[ix64,ix4096]):
        axis.axhline(bayes[index],color='black',linestyle=':')
        axis.set(yscale='log',xlabel='Training updates',ylabel='Reducible forecast MSE',title=f'Validation after {int(lengths[index]):,} observations')
        axis.set_xticks(STEPS)
        axis.grid(alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,fontsize=9)
    fig.suptitle('Checkpoint trajectories · thick lines: seed means; faint lines: individual seeds\nBands: ±1 seed SD; dotted line: exact Bayes conditional-risk reference',fontsize=11)
    for ext in ['png','svg']:
        fig.savefig(output/f'learning_curves.{ext}',dpi=180)
    plt.close(fig)

    summary={'groups':group_summary,'runs':run_summary,'paired_contrasts':contrasts,
             'training_seeds':[182,183,184], 'confirmation_set_opened':False,
             'bayes_epistemic_at_64':float(bayes[ix64]),'bayes_epistemic_at_4096':float(bayes[ix4096]),
             'generated_at_utc':datetime.now(timezone.utc).isoformat()}
    save_json(output/'summary.json',summary)
    save_json(output/'manifest.json',manifest)
    save_json(output/'training_histories.json',{r['run_name']:r['history'] for r in runs})
    return [row for row in group_summary if row['training_steps']==10000]


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    print(json.dumps(summarize(args.raw,args.output,args.manifest,args.repo),indent=2))
