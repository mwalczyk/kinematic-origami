import matplotlib
import matplotlib.pyplot as plt
from matplotlib import collections
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def plot_reference_configuration(axes_3d, crease_pattern):
	'''Draws the crease pattern in its intial (reference) configuration, where all fold
	angles are implicitly set to zero 

	'''
	fold_angles = np.array([0.0 for _ in range(crease_pattern.num_folds)])

	plot_custom_configuration(axes_3d, crease_pattern, fold_angles)

def plot_custom_configuration(axes_3d, crease_pattern, fold_angles, color_map_name='winter', alpha=0.75, edges=True):
	'''Draws the crease pattern in a particular folded state

	'''
	# Matplotlib utility for mapping face indices to colors
	scalar_color_map = ScalarMappable(norm=matplotlib.cm.colors.Normalize(0, crease_pattern.num_faces), 
									  cmap=plt.get_cmap(color_map_name)) 

	# Grab the dimensions of the `Axes3D` object
	_, size_x = axes_3d.get_xlim3d()
	_, size_y = axes_3d.get_ylim3d()
	if len(fold_angles) != crease_pattern.num_folds:
		raise Error('Invalid number of fold angles')

	# Reset the collections object to effectively clear the screen
	axes_3d.collections = []

	# Compute the face map based on the provided fold angles
	face_map = crease_pattern.compute_folding_map(fold_angles)

	# Add all face polygons to one array (so that depth testing works)
	all_polys = np.zeros((crease_pattern.num_faces, np.max(crease_pattern.num_face_corner_points), 3))

	for i in range(crease_pattern.num_faces):
		# Grab all of the 2D corner points that form this face
		points_2d = crease_pattern.face_corner_points[i][:crease_pattern.num_face_corner_points[i]]
		extra_0 = np.zeros((np.shape(points_2d)[0], 1))
		extra_1 = np.ones((np.shape(points_2d)[0], 1))

		# "Expand" the 2D coordinates into 4D by adding zeros and ones
		points_3d = np.hstack((points_2d, extra_0))
		points_4d = np.hstack((points_3d, extra_1))

		# Grab the 4x4 transformation matrix
		composite = face_map[i]

		# Transform each corner point by the composite matrix
		for j in range(crease_pattern.num_face_corner_points[i]):
			points_4d[j] = np.dot(composite, points_4d[j])

		# Add a new polygon (drop the w-coordinate)
		all_polys[i] = points_4d[:,:3] + [size_x * 0.5, size_y * 0.5, 0.0]

	# Construct the actual polygon collection object and configure its draw state
	poly_collection = Poly3DCollection(all_polys)
	poly_collection.set_facecolor([scalar_color_map.to_rgba(i)[:3] for i in range(crease_pattern.num_faces)])
	poly_collection.set_alpha(alpha)
	if edges:
		poly_collection.set_edgecolor('k')

	axes_3d.add_collection3d(poly_collection)

def plot_crease_pattern(axes_2d, crease_pattern, color_map_name='autumn', annotate_folds=True, annotate_reference_points=True, annotate_faces=True):
	'''Draws the planar graph corresponding to this crease pattern, along with some
	helpful annotations

	'''
	# Matplotlib utility for mapping fold indices to colors
	scalar_color_map = ScalarMappable(norm=matplotlib.cm.colors.Normalize(0, crease_pattern.num_folds), 
									  cmap=plt.get_cmap(color_map_name)) 

	colors = [scalar_color_map.to_rgba(i)[:3] for i in range(crease_pattern.num_folds)]

	line_segments = [[a, b] for a, b in zip(crease_pattern.p1, crease_pattern.p2)]
	line_segment_midpoints = [(a + b) * 0.5 for a, b in zip(crease_pattern.p1, crease_pattern.p2) ]
	line_collection = collections.LineCollection(line_segments, colors=colors, linewidths=1)
	
	axes_2d.add_collection(line_collection)
	axes_2d.scatter(crease_pattern.reference_points[:, 0], crease_pattern.reference_points[:, 1], zorder=2)

	aspect_ratio = 1.0
	xleft, xright = axes_2d.get_xlim()
	ybottom, ytop = axes_2d.get_ylim()
	axes_2d.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * aspect_ratio)

	if annotate_folds:
		for i, (x, y) in enumerate(line_segment_midpoints):
			label = 'fold {}'.format(i)
			axes_2d.annotate(label, (x, y), textcoords="offset points", xytext=(5, 10), ha='center', fontsize='medium', fontweight='bold')
	if annotate_reference_points:
		for i, (x, y) in enumerate(crease_pattern.reference_points):
			label = 'rp {}'.format(i)
			axes_2d.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize='x-small')
	if annotate_faces:
		for i, (x, y) in enumerate(crease_pattern.face_centers):
			label = 'face {}'.format(i)
			axes_2d.annotate(label, (x, y), textcoords="offset points", xytext=(0, 0), ha='center', fontsize='x-small')