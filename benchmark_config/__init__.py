"""Per-model benchmark configs. Root benchmark.ipynb imports one of these.

Switch models by changing import:
    from benchmark_config import bert   as cfg
    from benchmark_config import llama  as cfg
    from benchmark_config import vision as cfg
    from benchmark_config import yolo   as cfg

    cfg.run_all()
"""
