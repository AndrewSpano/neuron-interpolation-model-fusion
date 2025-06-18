import itertools
import matplotlib.pyplot as plt

from collections import defaultdict


PREFIX_TO_COLOR = {
    'hf': 'purple',
    'kf': 'green',
    'ot': 'red',
    'alf': 'blue',
    'git': 'orange',
    'model': 'gray',
}
HATCH_PATTERNS = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']


def get_prefix(name):
    """Assumes prefix is before the first '-'."""
    return name.split('-')[0]


def get_name_to_style(model_names):
    """Assign colors and hatches to model names based on their prefixes."""
    # group model names by prefix
    prefix_to_models = defaultdict(list)
    for name in model_names:
        prefix = get_prefix(name)
        prefix_to_models[prefix].append(name)

    # assign hatches within each prefix
    NAME_TO_STYLE = {}
    for prefix, names in prefix_to_models.items():
        hatch_cycle = itertools.cycle(HATCH_PATTERNS)
        for name in sorted(names):  # sort for consistency
            NAME_TO_STYLE[name] = (PREFIX_TO_COLOR.get(prefix, 'black'), next(hatch_cycle))

    return NAME_TO_STYLE

def bar_plot(ax: plt.Axes, metric, stats, model_names, is_01_metric=True):
    """Plot a bar plot with hatches for different models."""
    name_to_style = get_name_to_style(model_names)
    values = [stat[metric] for stat in stats]

    y_lim_min = min(values) - 0.02
    y_lim_max = max(values) + 0.02
    if is_01_metric:
        y_lim_min = max(0, y_lim_min)
        y_lim_max = min(1, y_lim_max)

    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    names_sorted = [model_names[i] for i in sorted_indices]
    values_sorted = [values[i] for i in sorted_indices]

    # Now use individual bar plotting to apply hatches
    for i, (name, value) in enumerate(zip(names_sorted, values_sorted)):
        color, hatch = name_to_style[name]
        ax.bar(i, value, color=color, hatch=hatch)

        # Label value
        increment_above = (y_lim_max - y_lim_min) / 100
        ax.text(i, value + increment_above, f'{value:.4f}', ha='center', va='bottom', fontsize=12)

    ax.set_xticks(range(len(names_sorted)))
    ax.set_xticklabels(names_sorted, rotation=45, ha='right')
    ax.set_ylim(y_lim_min, y_lim_max)
    ax.set_title(f'{metric} comparison')
    ax.set_ylabel(metric)
    ax.grid(axis='y')
