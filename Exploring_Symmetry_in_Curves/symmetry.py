import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from scipy.optimize import minimize

def bezier_curve(t, P):
    return (1-t)**3 * P[0] + 3*(1-t)**2 * t * P[1] + 3*(1-t) * t**2 * P[2] + t**3 * P[3]

def generate_points(bezier_curves, num_points=100):
    points = []
    for curve in bezier_curves:
        t_values = np.linspace(0, 1, num_points)
        curve_points = np.array([bezier_curve(t, curve) for t in t_values])
        points.append(curve_points)
    return np.vstack(points)

def find_symmetry(points):
    pca = PCA(n_components=2)
    pca.fit(points)
    symmetry_axes = pca.components_
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    mirrored_points = np.dot(centered_points, symmetry_axes.T)
    original_distances = cdist(points, points)
    mirrored_distances = cdist(mirrored_points, mirrored_points)
    symmetry_score = np.sum(np.abs(original_distances - mirrored_distances))
    return symmetry_axes, symmetry_score

def fit_bezier(points):
    def objective(P, points):
        t_values = np.linspace(0, 1, len(points))
        P = np.reshape(P, (-1, 2))
        fitted_points = np.array([bezier_curve(t, P) for t in t_values])
        return np.sum(np.linalg.norm(fitted_points - points, axis=1))

    P0 = np.linspace(points[0], points[-1], 4).flatten()
    res = minimize(objective, P0, args=(points,))
    P_opt = np.reshape(res.x, (-1, 2))
    return P_opt

bezier_curves = [
    np.array([[0, 0], [1, 2], [2, 2], [3, 0]]),
    np.array([[3, 0], [4, -2], [5, -2], [6, 0]])
]

points = generate_points(bezier_curves)
symmetry_axes, symmetry_score = find_symmetry(points)
mirrored_points = np.dot(points - np.mean(points, axis=0), symmetry_axes.T)
symmetric_bezier = fit_bezier(mirrored_points)
