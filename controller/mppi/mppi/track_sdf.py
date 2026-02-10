import numpy as np
import cv2
import matplotlib.pyplot as plt


class TrackSDF:
    """Binary drivable-area map: 0.0 = free, 1.0 = wall/outside."""

    def __init__(self, wx, wy, wyaw, w_left, w_right, resolution=0.05, map_padding=2.0):
        self.resolution = resolution
        self.wx = wx
        self.wy = wy
        self.wyaw = wyaw
        self.w_left = w_left
        self.w_right = w_right

        # --- Boundary points for grid extent ---
        waypoints = np.stack([wx, wy], axis=1)
        gradients = np.gradient(waypoints, axis=0)
        norms = np.linalg.norm(gradients, axis=1, keepdims=True)
        tangents = gradients / (norms + 1e-6)
        normals = np.zeros_like(tangents)
        normals[:, 0] = -tangents[:, 1]
        normals[:, 1] = tangents[:, 0]

        left_bound = waypoints + normals * w_left[:, np.newaxis]
        right_bound = waypoints - normals * w_right[:, np.newaxis]

        all_x = np.concatenate([wx, left_bound[:, 0], right_bound[:, 0]])
        all_y = np.concatenate([wy, left_bound[:, 1], right_bound[:, 1]])

        self.min_x = float(np.min(all_x) - map_padding)
        self.max_x = float(np.max(all_x) + map_padding)
        self.min_y = float(np.min(all_y) - map_padding)
        self.max_y = float(np.max(all_y) + map_padding)

        self.width_cells = int((self.max_x - self.min_x) / resolution)
        self.height_cells = int((self.max_y - self.min_y) / resolution)
        self.origin = np.array([self.min_x, self.min_y])

        print(f"[TrackSDF] Building binary map: {self.width_cells}(W) x {self.height_cells}(H) @ {resolution}m", flush=True)

        self._bake_binary_map(waypoints, normals, left_bound, right_bound)

    # ------------------------------------------------------------------
    def _bake_binary_map(self, waypoints, normals, left_bound, right_bound):
        """Rasterise track boundaries → binary cost_map (W, H).
        
        cost_map[x_idx, y_idx]:  0.0 = drivable,  1.0 = wall/outside
        """
        H, W = self.height_cells, self.width_cells

        def world_to_pix(pts):
            """World coords → pixel coords (col, row) for OpenCV."""
            return ((pts - self.origin) / self.resolution).astype(np.int32)

        left_pix = world_to_pix(left_bound)
        right_pix = world_to_pix(right_bound)

        # Fill drivable area segment-by-segment (handles self-intersecting tracks)
        drivable_mask = np.zeros((H, W), dtype=np.uint8)
        for i in range(len(left_pix) - 1):
            quad = np.array([
                left_pix[i], left_pix[i + 1],
                right_pix[i + 1], right_pix[i],
            ], dtype=np.int32)
            cv2.fillConvexPoly(drivable_mask, quad, 1)

        # Binary: 0 = free, 1 = wall
        # drivable_mask: 1 = drivable, 0 = non-drivable
        # Invert: wall = 1 - drivable
        binary = (1 - drivable_mask).astype(np.float32)

        # Transpose to (W, H) so indexing is cost_map[x_idx, y_idx]
        self.cost_map = binary.T

        print(f"[TrackSDF] Binary map baked. shape={self.cost_map.shape} "
              f"(free={int(np.sum(binary == 0))}, wall={int(np.sum(binary == 1))})", flush=True)

    # ------------------------------------------------------------------
    def get_jax_maps(self):
        """Return JAX arrays for the MPPI solver."""
        import jax.numpy as jnp

        map_data = {
            'cost_map': jnp.array(self.cost_map, dtype=jnp.float32),  # (W, H): 0=free, 1=wall
        }
        map_metadata = {
            'origin_x': float(self.min_x),
            'origin_y': float(self.min_y),
            'resolution': float(self.resolution),
        }
        return map_data, map_metadata

    # ------------------------------------------------------------------
    def show_debug_map(self):
        """Quick matplotlib visualisation of the binary map."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # cost_map is (W, H) → transpose to (H, W) for imshow (rows=Y, cols=X)
        vis = self.cost_map.T
        ax.imshow(vis, origin='lower',
                  extent=[self.min_x, self.max_x, self.min_y, self.max_y],
                  cmap='gray_r', vmin=0, vmax=1)
        ax.set_title("Binary Cost Map (black=wall, white=free)")
        ax.plot(self.wx, self.wy, 'r--', linewidth=1, alpha=0.7, label='centerline')

        # Track boundaries
        yaw = self.wyaw
        nx, ny = -np.sin(yaw), np.cos(yaw)
        ax.plot(self.wx + nx * self.w_left, self.wy + ny * self.w_left,
                'c-', linewidth=0.8, label='left')
        ax.plot(self.wx - nx * self.w_right, self.wy - ny * self.w_right,
                'b-', linewidth=0.8, alpha=0.7, label='right')
        ax.legend()
        plt.tight_layout()
        plt.show(block=True)
        print("[TrackSDF] Debug map displayed.", flush=True)
