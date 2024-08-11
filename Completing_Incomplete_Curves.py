import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

def load_polylines(file_path):
    # Load polylines from a CSV file
    data = pd.read_csv(file_path, header=None)
    polylines = []
    current_polyline = []

    for index, row in data.iterrows():
        if np.isnan(row[0]):
            polylines.append(np.array(current_polyline))
            current_polyline = []
        else:
            current_polyline.append(row.values)

    if current_polyline:
        polylines.append(np.array(current_polyline))

    return polylines

def save_polylines(polylines, file_path):
    # Save polylines to a CSV file
    with open(file_path, 'w') as f:
        for polyline in polylines:
            for point in polyline:
                f.write(f"{point[0]},{point[1]}\n")
            f.write("\n")

def interpolate_curve(points, num_points=100):
    # Fit a spline to the points and interpolate to get a smooth curve
    if len(points) < 3:
        return points  # Not enough points to interpolate
    tck, u = splprep([points[:,0], points[:,1]], s=0)
    unew = np.linspace(0, 1.0, num_points)
    out = splev(unew, tck)
    return np.stack(out, axis=-1)

def complete_connected_curve(curve):
    # Logic to complete a connected curve
    completed_curve = interpolate_curve(curve)
    return completed_curve

def complete_disconnected_curve(curves):
    # Logic to connect multiple curve segments
    all_points = np.vstack(curves)
    completed_curve = interpolate_curve(all_points)
    return completed_curve

def process_occlusion(file_path, output_path, occlusion_type):
    # Load polylines
    polylines = load_polylines(file_path)
    
    if occlusion_type == "connected":
        completed_curves = [complete_connected_curve(polyline) for polyline in polylines]
    elif occlusion_type == "disconnected":
        completed_curves = [complete_disconnected_curve(polylines)]
    
    # Save completed curves
    save_polylines(completed_curves, output_path)

# Example usage:
process_occlusion('examples/occlusion1.csv', 'examples/occlusion1_sol.csv', 'connected')
process_occlusion('examples/occlusion2.csv', 'examples/occlusion2_sol.csv', 'disconnected')
