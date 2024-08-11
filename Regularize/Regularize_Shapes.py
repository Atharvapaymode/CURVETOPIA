import numpy as np
import svgwrite
from scipy.interpolate import splprep, splev

def bezier_from_points(points, t_values):
    tck, _ = splprep([points[:, 0], points[:, 1]], s=0)
    bezier_points = np.array(splev(t_values, tck)).T
    return bezier_points

def regularize_path(path):
    t_values = np.linspace(0, 1, num=100)
    regularized = bezier_from_points(np.array(path), t_values)
    return regularized

def paths_to_svg(paths, filename):
    dwg = svgwrite.Drawing(filename, profile='tiny')
    for path in paths:
        start = path[0]
        path_data = f"M {start[0]},{start[1]} "
        for i in range(1, len(path), 3):
            p1, p2, p3 = path[i:i+3]
            path_data += f"C {p1[0]},{p1[1]} {p2[0]},{p2[1]} {p3[0]},{p3[1]} "
        dwg.add(dwg.path(d=path_data, fill="none", stroke="black"))
    dwg.save()

def process_paths(input_paths):
    processed_paths = [regularize_path(path) for path in input_paths]
    return processed_paths

if __name__ == "__main__":
    input_paths = [
        [[50, 150], [100, 50], [150, 150], [200, 50], [250, 150]],
        [[300, 100], [350, 200], [400, 100], [450, 200]]
    ]
    
    output_paths = process_paths(input_paths)
    paths_to_svg(output_paths, "output.svg")
