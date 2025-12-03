import os
import pstats

PROJECT_ROOT = '/home/jovyan/project/NeoQA/dataset-generation'  # adjust

stats = pstats.Stats('myLog.profile')

# filter out entries whose filename is not under PROJECT_ROOT
filtered = pstats.Stats()
for func, func_stats in stats.stats.items():
    filename = func[0]
    # normalize paths (for comparison)
    norm = os.path.normpath(filename)
    if norm.startswith(os.path.normpath(PROJECT_ROOT)):
        filtered.stats[func] = func_stats

filtered.dump_stats('project_only.profile')
