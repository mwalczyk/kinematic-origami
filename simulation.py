import math
import matplotlib.pyplot as plt
from matplotlib import collections
import numpy as np
import pylab as pl
from scipy.spatial.transform import Rotation as R

np.random.seed(0)
flip_y = False
filler_index = -1


'''
Model inputs 

'''
# Coordinates of all of the points in the reference (starting)
# configuration
reference_points = np.array([ 
	[-20, -20],
	[  0, -20],
	[ 20, -20],
	[-20,   0],
	[  0,   0],
	[ 20,   0],
	[-20,  20],
	[  0,  20],
	[ 20,  20]
])

# Matlab's y-axis points downwards, so we flip each of the reference
# points here to be consistent with the author
if flip_y:
	for point in reference_points:
		point[1] *= -1.0

# Indices corresponding to each pair of reference points that form
# the start + end vertices of each fold vector
#
# For example, an entry [0, 1] would correspond to the fold vector 
# pointing from reference point 0 towards reference point 1
fold_vector_points = np.array([
	[4, 0],
	[4, 1],
	[4, 2],
	[4, 3],
	[4, 5],
	[4, 6],
	[4, 7],
	[4, 8]
])

# A matrix containing the fold indices associated with each interior fold
# intersection (in CW order)
#
# In the Matlab implementation, these indices are signed (positive or
# negative), where the sign corresponded to:
# 
# + -> fold vector points away from the intersection 
# - -> fold vector points towards the intersection
#
# Since Matlab uses 1-based indexing, this works...however, it leads to 
# issues in Python, which uses 0-based indexing (since 0 can't be negative)
#
# So, to avoid the use of negative indices, we use a second array
# below:
#
# `True`: the fold vector points away from the intersection
# `False`: the fold vector points towards the intersection
# 
# Indices:
# i -> index of the interior fold intersection
# j -> index of the `j`th fold surrounding the `i`th interior fold intersection
intersection_fold_indices = np.array([
	[0, 1, 2, 4, 7, 6, 5, 3]
])

sign_intersection_fold_indices = np.array([
	[True, True, True, True, True, True, True, True]
])

# Indices of folds and reference points defining the boundary of each 
# face (in CCW order)
#
# Sign:
# `True` or positive (+): fold vector points CCW around the face
# `False` or negative (-): fold vector points CW around the face
#
# Use the index `filler_index` to right-fill "jagged" arrays, i.e.
# if one face has more boundary folds than the others 
face_boundary = np.array([
	[3, 0], # 3, -0
	[0, 1], # 0, -1
	[1, 2], # 1, -2 
	[2, 4], # 2, -4
	[4, 7], # 4, -7
	[7, 6], # 7, -6
	[6, 5], # 6, -5  
	[5, 3]  # 5, -3
])

sign_face_boundary = np.array([
	[True, False],
	[True, False],
	[True, False],
	[True, False],
	[True, False],
	[True, False],
	[True, False],
	[True, False]
])

# The index of the face that will remain fixed throughout the simulation
fixed_face = 7

# Highest allowable fold angle for each fold
fold_angle_upper_bound = np.array([ 
	math.pi, 
	math.pi,
	math.pi,
	math.pi,
	math.pi,
	math.pi,
	math.pi,
	math.pi
])

# Lowest allowable fold angle for each fold
found_angle_lower_bound = np.array([
	-math.pi,
	-math.pi,
	-math.pi,
	-math.pi,
	-math.pi,
	-math.pi,
	-math.pi,
	-math.pi
])

# Initial value for each fold angle
fold_angle_initial_value = np.array([
	0.0,
	0.0,
	0.0,
	0.0,
	0.0,
	0.0,
	0.0,
	0.0
])




'''
Calculated data

'''

# The total number of reference points (i.e. vertices)
num_reference_ponts = np.shape(reference_points)[0]

# The total number of folds (i.e. creases) 
num_folds = np.shape(fold_vector_points)[0] 

# The total number of interior fold intersections (i.e. interior vertices)
num_fold_intersections = np.shape(intersection_fold_indices)[0]

# The total number of faces
num_faces = np.shape(face_boundary)[0]

assert len(fold_angle_upper_bound) == num_folds 
assert len(found_angle_lower_bound) == num_folds 
assert len(fold_angle_initial_value) == num_folds 

# TODO: for each interior fold intersection, calculate the number of incident folds
# at this intersection - in the original Matlab code, this would correspond to the 
# number of non-zero entries in each row of the `intersection_fold_indices` matrix
# (the index `0` is used to right-fill rows so that they are all the same length)
#
# However, things are a bit complicated, since Python uses 0-based indexing - we 
# can't simply use `np.nonzero(...)`
#
# Additionally, entries in the `intersection_fold_indices` matrix are differentiated
# by their sign (positive or negative), and since zero can't be negative, we again 
# run into issues here...
# 
# For now, we just hard-code the array below
for i in range(num_fold_intersections):
	pass

# The number of folds associated with each interior fold intersection
#
# TODO
num_intersection_folds = np.array([
	8 
], dtype=np.int8)

# The starting points of each fold vector
p1 = np.array([reference_points[i] for i, _ in fold_vector_points])
assert p1.shape == (num_folds, 2)

# The ending points of each fold vector
p2 = np.array([reference_points[j] for _, j in fold_vector_points])
assert p2.shape == (num_folds, 2)

fold_vector = p2 - p1
assert fold_vector.shape == (num_folds, 2)

fold_ref_angle_wrt_e1 = np.zeros((num_folds, 1))

for i, v in enumerate(fold_vector):
	# Extract the xy-coordinates of this vector and calculate its length (L2 norm)
	x, y = v
	norm_of_v = math.sqrt(x*x + y*y)

	# Formula (2.13): calculate the angle that this vector makes with e1, i.e. the
	# x-axis
	# 
	# Note that we can avoid doing a cross-product (as in the original formula) and
	# simply check the sign of the vector's y-coordinate
	# 
	# The angles returned by this formula are always positive in the range 0..two pi
	if y >= 0.0: 
		fold_ref_angle_wrt_e1[i] = math.acos(x / norm_of_v)
	else:
		fold_ref_angle_wrt_e1[i] = 2.0 * math.pi - math.acos(x / norm_of_v)


# The direction vectors along each interior fold 
#
# Indices:
# i -> index of the interior fold intersection 
# j -> index of the `j`th fold surrounding the `i`th interior fold intersection
# k -> 0 or 1, the x or y-coordinate
fold_direction_i = np.zeros((num_fold_intersections, np.max(num_intersection_folds), 2))

# The angle that each interior fold makes with e1
fold_ref_angle_wrt_e1_i = np.zeros((num_fold_intersections, np.max(num_intersection_folds)))

for i in range(num_fold_intersections):

	for j in range(num_intersection_folds[i]):

		k = intersection_fold_indices[i][j]
		v = fold_vector[k]

		# We might need to flip the vector along this fold
		if not sign_intersection_fold_indices[i][j]:
			v *= -1.0

		# Store this vector
		fold_direction_i[i][j] = v

		x, y = v
		norm_of_v = math.sqrt(x*x + y*y)

		if y >= 0.0: 
			fold_ref_angle_wrt_e1_i[i][j] = math.acos(x / norm_of_v)
		else:
			fold_ref_angle_wrt_e1_i[i][j] = 2.0 * math.pi - math.acos(x / norm_of_v)

for fold_index, theta in zip(intersection_fold_indices[0], fold_ref_angle_wrt_e1_i[0]):
	print("Fold {}: {} degrees w.r.t. +x-axis".format(fold_index, math.degrees(theta)))


# The face corner angles surrounding each interior fold intersection
face_corner_angles = np.zeros((num_fold_intersections, max(num_intersection_folds)))

for i in range(num_fold_intersections):

	for j in range(num_intersection_folds[i]):
		# If this is the last fold that emanates from the `i`th interior fold intersection, 
		# 
		# 
		# Otherwise, the face corner angle is simply the difference between the next
		# fold angle and the current fold angle
		if j == num_intersection_folds[i] - 1:
			face_corner_angles[i][j] = 2.0 * math.pi + (fold_ref_angle_wrt_e1_i[i][0] - fold_ref_angle_wrt_e1_i[i][j])
		else:
			face_corner_angles[i][j] = fold_ref_angle_wrt_e1_i[i][j + 1] - fold_ref_angle_wrt_e1_i[i][j]

# The corner points of each face (facet)
#
# Indices:
# i -> index of the face
# j -> index of the `j`th corner point bounding the `i`th face
# k -> 0 or 1, the x or y-coordinate of the corner point
#
# The maximum number of boundary folds for a given face is dictated by the 
# size of the 2nd dimension of the `face_boundary` array, which will be 
# right-filled to accomodate any faces that have less than `max_boundary_folds_per_face`
# boundary folds
max_boundary_folds_per_face = np.shape(face_boundary)[1]

face_corner_points = np.zeros((num_faces, 2 * max_boundary_folds_per_face, 2))

# The number of corner points per face, 1D array
num_face_corner_points = np.zeros(num_faces, dtype=np.int8)

for i in range(num_faces):

	count = 0

	for j in range(len(face_boundary[i])):
		# The index of the `j`th fold that bounds the `i`th face
		k = face_boundary[i][j]

		# Have we reached the end of this face?
		if k == filler_index:
			break

		if sign_face_boundary[i][j]:
			# Connect to fold in "positive" direction - the fold already goes CCW
			face_corner_points[i][count] = p1[k]
			count += 1
			face_corner_points[i][count] = p2[k]
			count += 1
		else:
			# Connect to fold in "negative" direction
			face_corner_points[i][count] = p2[k] # This used to be `p2[abs(k)]`, but we aren't using negative indices anymore
			count += 1
			face_corner_points[i][count] = p1[k]
			count += 1
    
	num_face_corner_points[i] = count

for face_index in range(num_faces):
	print('Face {} corner points:'.format(face_index))
	for point_index in range(num_face_corner_points[face_index]):
		print('\tPoint {}: {}'.format(point_index, face_corner_points[face_index][point_index]))

print(face_corner_points)

face_centers = np.zeros((num_faces, 2))
for face_index in range(num_faces):
	center = [0.0, 0.0]

	for point_index in range(num_face_corner_points[face_index]):
		center += face_corner_points[face_index][point_index]

	center /= num_face_corner_points[i]

	face_centers[face_index] = center

# The indices of the folds crossed by each path connecting the fixed face to every other face
#
# Indices:
# i -> index of the "target" face
# j -> index of the `j`th fold crossed en-route to face `i` from face `fixed_face` 
# 
# For example, an entry at index 0 of the form `[3, 2, 1]` would indicate that in order to 
# get from the `fixed_face` (say, the 3rd face) to the 0th face, we would need to cross the 
# folds at indices 3, 2, and 1 (in that order)

# TODO: find a Hamiltonian cycle - page 233 of "Geometric Folding Algorithms" 
fold_paths = np.array([
	[3, filler_index, filler_index, filler_index, filler_index, filler_index, filler_index, filler_index],
	[3, 0, filler_index, filler_index, filler_index, filler_index, filler_index, filler_index],
	[3, 0, 1, filler_index, filler_index, filler_index, filler_index, filler_index],
	[3, 0, 1, 2, filler_index, filler_index, filler_index, filler_index],
	[3, 0, 1, 2, 4, filler_index, filler_index, filler_index],
	[3, 0, 1, 2, 4, 7, filler_index, filler_index],
	[3, 0, 1, 2, 4, 7, 6, filler_index],
	[3, 0, 1, 2, 4, 7, 6, 5]
])

assert np.shape(fold_paths) == (num_faces, num_folds)





# Reference:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
# 
# Use `m.apply(v)` to apply a rotation matrix to a vector
# ...

def r1(phi):
	'''Computes a 3x3 rotation matrix corresponding to a rotation of `phi`
	radians around the positive x-axis (e1)

	'''
	return R.from_euler('x', phi)

def r3(phi):
	'''Computes a 3x3 rotation matrix corresponding to a rotation of `phi`
	radians around the positive z-axis (e3)
	
	'''
	return R.from_euler('z', phi)

def fold_transform(alpha, phi):
	'''Computes the transform associated with a rotation of `phi` 
	radians around a fold that makes an angle `alpha` w.r.t. the 
	positive x-axis (e1)
	
	See equation `2.23` in the text

	'''
	# When traversing the fold map, always use `p1` and compose rotations
	# 
	# If we cross a fold in the "positive" direction, simply apply
	# the rotation matrix as seen below
	# 
	# If we cross a fold in the "negative" direction, add `pi` to `phi`
	# before constructing the rotation matrix

	return r3(alpha) * r1(phi) * r3(alpha).inv()

print('Testing R1 (rotation around x-axis) with pi/8...')
r1_test = r1(math.pi * 0.125)
print(r1_test.as_matrix())

print('Testing R3 (rotation around z-axis) with pi/8...')
r3_test = r3(math.pi * 0.125)
print(r3_test.as_matrix())

alpha = math.pi / 4
phi = math.pi / 8
print(f'Testing composite transform with: `alpha` = {alpha}, `phi` = {phi}')
comp_test = fold_transform(alpha, phi)
print(comp_test.as_matrix())

# Should be (roughly)
#
# [[  0.9619, 0.0380,  0.2705 ]]
# [[  0.0380, 0.9619, -0.2705 ]]
# [[ -0.2705, 0.2705,  0.9238 ]]










'''
Plotting

'''
colors = np.random.rand(len(fold_vector_points), 3)

line_segments = [[a, b] for a, b in zip(p1, p2)]
line_segment_midpoints = [(a + b) * 0.5 for a, b in zip(p1, p2) ]

line_collection = collections.LineCollection(line_segments, colors=colors, linewidths=1)
fig, ax = plt.subplots()
ax.add_collection(line_collection)
ax.scatter(reference_points[:, 0], reference_points[:, 1])

for i, (x, y) in enumerate(line_segment_midpoints):
	label = 'F{}'.format(i)
	ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 10), ha='center', fontweight='bold')

for i, (x, y) in enumerate(reference_points):
	label = 'v{}'.format(i)
	ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize='x-small')

for i, (x, y) in enumerate(face_centers):
	label = 'face {}'.format(i)
	ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 0), ha='center', fontsize='x-small')

ax.autoscale()
ax.margins(0.1)

plt.show()













'''
Simulation inputs 

'''
# Weight for residuals from rotation constraints
weight_rotation_constraint = 1

# Weight for residuals from fold angle bound constraints
weight_fold_angle_bounds = 1e-5

# Tolerance for norm of residual vector: https://en.wikipedia.org/wiki/Residual_(numerical_analysis)
tolerance_residual = 1e-8

# Tolerance for correction of fold angles
tolerance_fold_angle = 1e-8

# Maximum number of iterations allowed for each increment
max_iterations = 15

# Maximum correction allowed for individual fold angles in each iteration
max_fold_angle_correction = math.radians(5.0)

# Number of increments
num_increments = 50

# Initialize matrix that contains each fold angle change for every increment 
fold_angle_change_per_increment = np.zeros((np.shape(fold_vector_points)[0], num_increments))

# Reference angle for filling fold angle change per increment matrix
reference_fold_angle = math.radians(90.0)

# Fold angle change per increment matrix components
for i in range(num_increments):
	# Set the `i`th column of the matrix to the specified array (note that in the
	# author's code, the array is a column vector...does this matter?)
	fold_angle_change_per_increment[:, i] = (reference_fold_angle / num_increments) * np.array([-1, 2, -1, 2, 2, -1, 2, -1])

#print(fold_angle_change_per_increment)

# Input to determine one of two methods for calculating initial fold angle increment. One uses the previous 
# Jacobian (corrected; Select 1) and the other one does not (not corrected; Select 2) 
increment_correction = 1

# Finite difference step in fold angles for calculation of derivatives
fin_diff_step = math.radians(2.0)