import json

from pathlib import Path
from jinja2 import Template

# define root and splits
# jinja_template_path = Path(__file__).parent/'jinja_templates'/'alf_tex_table_template_6.jinja'
# jinja_template_path = Path(__file__).parent/'jinja_templates'/'all_tex_table_template_6.jinja'
jinja_template_path = Path(__file__).parent/'jinja_templates'/'all_tex_table_template.jinja'

base_dir = Path('base-models/non-iid/VGGs/CIFAR-10')

splits = [2, 4, 8]
KEY = 'f1'  # or 'f1' or 'loss'
data = {}
add_other_algorithms = True

# collect data
for split in splits:
    json_path = base_dir / f'split-by-{split}'/'averaged_results.json'
    with open(json_path) as f:
        results = json.load(f)

    def format_metric(entry):
        return {
            'mean': f'{entry["mean"] * 100:.2f}',
            'std': f'{entry["std"] * 100:.2f}'
        }

    alf_prefix = f'alf-{split}-'
    alf_methods = {
        k: format_metric(v) for k, v in results[KEY].items() if k.startswith(alf_prefix)
    }

    data[split] = {
        'individual_models': [format_metric(results[KEY][f'model_{i}']) for i in range(split)],
        'vanilla_averaging': format_metric(results[KEY]['vanilla_averaging']),
        'ensemble': format_metric(results[KEY]['ensemble']),
        'alf_methods': alf_methods,
    }

    # add methods for other algorithms (OTF, GR, HF, KF)
    if add_other_algorithms:
        for prefix in ['otf', 'git-rebasin', 'hf', 'kf']:
            methods = {
                k: format_metric(v) for k, v in results[KEY].items() if k.startswith(prefix)
            }
            prefix_for_dict = prefix + '_methods'
            data[split][prefix_for_dict] = methods

# data['caption'] = r'\textbf{Accuracy} of ViT (7 layers) on CIFAR10, for \textbf{non-IID splits}. ' \
#                   r'Results are averaged over 5 seeds. Mean and one std are shown.} \label{tab:acc-results-vit-cifar10-noniid'
data['caption'] = r'\textbf{F1 Score} of VGGs on CIFAR10, for \textbf{non-IID splits}. ' \
                  r'Results are averaged over 5 seeds. Mean and one std are shown.} \label{tab:acc-results-vgg-cifar10-noniid'

# load jinja template
with open(jinja_template_path) as f:
    template = Template(f.read())

# render LaTeX
latex_code = template.render(data=data)

# save or print
print(latex_code)
