"""Plot the 3D map of one session as a density heatmap.

    python visualize_pointcloud.py        latest session that has points
    python visualize_pointcloud.py 3      session 3
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from tracker import connect

conn = connect()
if len(sys.argv) > 1:
    session_id = int(sys.argv[1])
else:
    row = conn.execute('SELECT MAX(session_id) FROM point_cloud').fetchone()
    session_id = row[0] if row and row[0] is not None else 0
points = np.array(conn.execute('SELECT x, y, z FROM point_cloud WHERE session_id = ?',
                               (session_id,)).fetchall())
conn.close()
if len(points) == 0:
    sys.exit(f'No points stored for session {session_id}. Run triangulate_map.py or live_run.py first.')

# 1. Drop the farthest 10% from the median (stray outliers).
dist = np.linalg.norm(points - np.median(points, axis=0), axis=1)
kept = points[dist < np.percentile(dist, 90)]

# 2. Density per voxel; drop isolated points, color the rest by how crowded their voxel is.
extent = float(np.ptp(kept, axis=0).max()) or 1.0
voxel = extent / 40.0
keys = np.floor(kept / voxel).astype(np.int64)
_, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
density = counts[inverse.ravel()]
dense = kept[density >= 2]
density = density[density >= 2]

print(f'Session {session_id}: {len(points)} points, {len(kept)} after outlier filter, {len(dense)} after density filter')

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(dense[:, 0], dense[:, 2], -dense[:, 1], s=3, c=np.log1p(density), cmap='inferno')
ax.set_xlabel('X (right)')
ax.set_ylabel('Z (forward)')
ax.set_zlabel('Up')
ax.set_title(f'3D map, session {session_id} ({len(dense)} points)')
fig.colorbar(sc, label='log density', shrink=0.6)
plt.savefig('pointcloud_filtered.png', dpi=150)
print('Saved pointcloud_filtered.png')
plt.show()
