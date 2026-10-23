"""Live camera tracking only (no map points, no detection)."""
from live_run import run

if __name__ == '__main__':
    run(save_map=False, detect=False)
