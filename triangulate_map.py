"""Live camera tracking plus 3D map points saved to the database (no detection)."""
from live_run import run

if __name__ == '__main__':
    run(save_map=True, detect=False)
