import numpy as np
import gdal

def compute_flow_direction(dem):
    """Compute flow direction using the D8 algorithm."""
    nrows, ncols = dem.shape
    flow_dir = np.zeros((nrows, ncols), dtype=np.int32)

    # Compute the slope and aspect of each cell
    dx = 30.0  # Cell size in meters
    dy = 30.0
    slope_x = (dem[:, :-1] - dem[:, 1:]) / dx
    slope_y = (dem[:-1, :] - dem[1:, :]) / dy
    slope_x[slope_x > 1.0] = 1.0
    slope_x[slope_x < -1.0] = -1.0
    slope_y[slope_y > 1.0] = 1.0
    slope_y[slope_y < -1.0] = -1.0
    slope = np.arctan(np.sqrt(slope_x ** 2 + slope_y ** 2))
    aspect = np.arctan2(slope_y, -slope_x)

    # Compute the flow direction of each cell
    for i in range(1, nrows - 1):
        for j in range(1, ncols - 1):
            neighbors = [(i-1, j-1), (i-1, j), (i-1, j+1),
                         (i, j-1), (i, j+1),
                         (i+1, j-1), (i+1, j), (i+1, j+1)]
            elev_diff = dem[i, j] - dem[neighbors][:, :, np.newaxis]
            slopes = np.sqrt(np.sum(elev_diff ** 2, axis=1)) / dx
            slopes[slopes > 1.0] = 1.0
            angles = np.abs(aspect[i, j] - np.arctan2(dem[neighbors] - dem[i, j], dx))
            angles[angles > np.pi] -= 2.0 * np.pi
            weights = slopes * np.cos(angles)
            max_idx = np.argmax(weights)
            if weights[max_idx] > 0.0:
                flow_dir[i, j] = max_idx + 1

    return flow_dir

def compute_flow_accumulation(flow_dir):
    """Compute flow accumulation using a recursive algorithm."""
    nrows, ncols = flow_dir.shape
    stack = np.column_stack(np.where(flow_dir == 0))
    flow_accum = np.zeros((nrows, ncols), dtype=np.int32)

    while stack.size > 0:
        i, j = stack[-1]
        stack = stack[:-1]
        for ni, nj in [(i-1, j-1), (i-1, j), (i-1, j+1),
                       (i, j-1), (i, j+1),
                       (i+1, j-1), (i+1, j), (i+1, j+1)]:
            if ni < 0 or ni >= nrows or nj < 0 or nj >= ncols:
                continue
            if flow_dir[ni, nj] == 0:
                continue
            if flow_accum[ni, nj] == 0:
                stack = np.vstack((stack, (ni, nj)))
            flow_accum[ni, nj] += flow_accum[i, j] + (flow_dir[ni, nj] == 5) * 1
            return flow_accum
def compute_drainage_basins(flow_accum):
    """Compute drainage basins from flow accumulation."""
    nrows, ncols = flow_accum.shape
    pour_points = np.column_stack(np.where(flow_accum == np.max(flow_accum)))
    basins = np.zeros((nrows, ncols), dtype=np.int32)
    basin_num = 0
    while pour_points.size > 0:
        i, j = pour_points[-1]
        pour_points = pour_points[:-1]
        if basins[i, j] != 0:
            continue
        basin_num += 1
        queue = [(i, j)]
        while queue:
            i, j = queue.pop(0)
            basins[i, j] = basin_num
            for ni, nj in [(i-1, j-1), (i-1, j), (i-1, j+1),                       (i, j-1), (i, j+1),                       (i+1, j-1), (i+1, j), (i+1, j+1)]:
                if ni < 0 or ni >= nrows or nj < 0 or nj >= ncols:
                    continue
                if basins[ni, nj] != 0 or flow_accum[ni, nj] < flow_accum[i, j]:
                    continue
                if flow_accum[ni, nj] == flow_accum[i, j]:
                    queue.append((ni, nj))
                else:
                    pour_points = np.vstack((pour_points, (ni, nj)))

    return basins
def delineate_drainage_basins(dem_file, basin_file):
    """Delineate drainage basins from a digital elevation model."""
    dem_ds = gdal.Open(dem_file)
    dem = dem_ds.ReadAsArray().astype(np.float32)
    flow_dir = compute_flow_direction(dem)
    flow_accum = compute_flow_accumulation(flow_dir)
    basins = compute_drainage_basins(flow_accum)
    driver = gdal.GetDriverByName('GTiff')
    basin_ds = driver.Create(basin_file, dem_ds.RasterXSize, dem_ds.RasterYSize, 1, gdal.GDT_Int32)
    basin_ds.SetGeoTransform(dem_ds.GetGeoTransform())
    basin_ds.SetProjection(dem_ds.GetProjection())
    basin_ds.GetRasterBand(1).WriteArray(basins)
    basin_ds.FlushCache()
    basin_ds = None

    dem_ds = None

delineate_drainage_basins(r"C:\Users\nwagle\lidar\lidar\data\dem.tif", r"H:\\My Drive\BrownUni\Basin\Basin_deliniation\Basin.tif")