from cluster_utils import config_to_cmd_flags, generate_run_configs


t = generate_run_configs("pred_invention_vlm.yaml")

for cfg in t:
    cmd_flags = config_to_cmd_flags(cfg)
    approach = cfg.experiment_id.split("-")[-1]
    if approach != "ours":
        continue
    print(cmd_flags)
