import numpy as np
import cv2
import matplotlib.pyplot as plt


class TrackSDF:
    """Cost map for track: lower is better, high values near walls/outside."""

    def __init__(
        self,
        wx,
        wy,
        wyaw,
        w_left,
        w_right,
        resolution=0.05,
        map_padding=2.0,
        safety_margin=0.0,
        contour_half_width=0.50,
        contour_exp=4.0,
        contour_scale=1.0,
        wall_k=5.0,
        wall_scale=20.0,
        collision_cost=10000.0,
    ):
        self.resolution = resolution
        self.wx = wx
        self.wy = wy
        self.wyaw = wyaw
        self.w_left = w_left
        self.w_right = w_right
        self.safety_margin = float(safety_margin)
        self.contour_half_width = float(contour_half_width)
        self.contour_exp = float(contour_exp)
        self.contour_scale = float(contour_scale)
        self.wall_k = float(wall_k)
        self.wall_scale = float(wall_scale)
        self.collision_cost = float(collision_cost)

        # Apply safety margin by shrinking drivable widths
        self.w_left_eff = np.maximum(self.w_left - self.safety_margin, 0.0)
        self.w_right_eff = np.maximum(self.w_right - self.safety_margin, 0.0)

        # --- Boundary points for grid extent ---
        waypoints = np.stack([wx, wy], axis=1)
        gradients = np.gradient(waypoints, axis=0)
        norms = np.linalg.norm(gradients, axis=1, keepdims=True)
        tangents = gradients / (norms + 1e-6)
        normals = np.zeros_like(tangents)
        normals[:, 0] = -tangents[:, 1]
        normals[:, 1] = tangents[:, 0]

        left_bound = waypoints + normals * self.w_left_eff[:, np.newaxis]
        right_bound = waypoints - normals * self.w_right_eff[:, np.newaxis]

        all_x = np.concatenate([wx, left_bound[:, 0], right_bound[:, 0]])
        all_y = np.concatenate([wy, left_bound[:, 1], right_bound[:, 1]])

        self.min_x = float(np.min(all_x) - map_padding)
        self.max_x = float(np.max(all_x) + map_padding)
        self.min_y = float(np.min(all_y) - map_padding)
        self.max_y = float(np.max(all_y) + map_padding)

        self.width_cells = int((self.max_x - self.min_x) / resolution)
        self.height_cells = int((self.max_y - self.min_y) / resolution)
        self.origin = np.array([self.min_x, self.min_y])

        print(f"[TrackSDF] Building cost map: {self.width_cells}(W) x {self.height_cells}(H) @ {resolution}m", flush=True)

        self._bake_costmap_cv2(waypoints, left_bound, right_bound)

    # ------------------------------------------------------------------
    def _bake_costmap_cv2(self, waypoints, left_bound, right_bound):
        """
        Rasterize the track to create a cost map using OpenCV.

        cost_map[x_idx, y_idx]: lower is better, higher near walls/outside
        """
        H, W = self.height_cells, self.width_cells

        def world_to_pix(pts):
            """World coords → pixel coords (col, row) for OpenCV."""
            return ((pts - self.origin) / self.resolution).astype(np.int32)

        wp_pix = world_to_pix(waypoints)
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

        # Store mask for debugging/visualization
        self.drivable_mask = drivable_mask

        # Centerline mask (for contour cost)
        center_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.polylines(center_mask, [wp_pix.reshape(-1, 1, 2)], isClosed=False, color=1, thickness=1)

        # Distance from walls (boundary of drivable area)
        dist_from_wall_pix = cv2.distanceTransform(drivable_mask, cv2.DIST_L2, 5)
        dist_from_wall = dist_from_wall_pix * self.resolution

        # Distance from centerline (for contour cost)
        dist_from_center_pix = cv2.distanceTransform((1 - center_mask).astype(np.uint8), cv2.DIST_L2, 5)
        dist_from_center = dist_from_center_pix * self.resolution

        # Contour cost: normalized by track half-width, then exponentiated
        contour_normalized = dist_from_center / max(self.contour_half_width, 1e-6)
        contour_cost = (contour_normalized ** self.contour_exp) * self.contour_scale

        # Wall proximity cost: exponential increase near walls
        wall_cost = np.exp(-self.wall_k * dist_from_wall) * self.wall_scale

        # Combined cost = contour + wall
        cost_map = contour_cost + wall_cost

        # Infinite penalty for non-drivable area
        cost_map[drivable_mask == 0] = self.collision_cost

        # Transpose to (W, H) so indexing is cost_map[x_idx, y_idx]
        self.cost_map = cost_map.T

        print(f"[TrackSDF] Cost map baked. shape={self.cost_map.shape} "
              f"(min={float(np.min(cost_map)):.3f}, max={float(np.max(cost_map)):.3f})", flush=True)

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
        """
        Debug visualization.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # 1. Cost Map
        # self.cost_map is (W, H). imshow expects (H, W), so transpose.
        cost_vis = self.cost_map.T

        # Clip for visibility
        cost_vis = np.clip(cost_vis, 0, 10.0)

        im1 = axes[0].imshow(
            cost_vis,
            origin='lower',
            extent=[self.min_x, self.max_x, self.min_y, self.max_y],
            cmap='jet',
        )
        axes[0].set_title("Cost Map (Clipped 0-10)")
        fig.colorbar(im1, ax=axes[0])
        axes[0].plot(self.wx, self.wy, 'w--', linewidth=1, alpha=0.7)

        # 2. Drivable Mask
        mask = getattr(self, 'drivable_mask', None)
        if mask is not None:
            mask_vis = mask.astype(np.float32)
            im2 = axes[1].imshow(
                mask_vis,
                origin='lower',
                extent=[self.min_x, self.max_x, self.min_y, self.max_y],
                cmap='gray',
                vmin=0.0,
                vmax=1.0,
            )
            axes[1].set_title("Drivable Mask")
            fig.colorbar(im2, ax=axes[1])
            axes[1].plot(self.wx, self.wy, 'r--', linewidth=1, alpha=0.7)
        else:
            axes[1].set_title("Drivable Mask (N/A)")

        plt.tight_layout()
        plt.show(block=True)
        print("[TrackSDF] Debug map displayed.", flush=True)
