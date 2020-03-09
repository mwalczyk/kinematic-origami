import matplotlib
import matplotlib.pyplot as plt
from matplotlib import collections
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from matplotlib.widgets import Slider

import math
import numpy as np
import pylab as pl
from scipy.spatial.transform import Rotation as R

# Hide toolbars (with "save" button, etc.)
matplotlib.rcParams['toolbar'] = 'None'

# Set the Matplotlib theme
matplotlib.style.use('ggplot')

np.random.seed(0)

filler_index = -1


#####################################################################################
#
# Provided data
#
#####################################################################################
# Coordinates of all of the points in the reference (starting)
# configuration: each of these should lie in span(e1, e2), i.e.
# on the xy-plane
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

# Indices corresponding to each pair of reference points that form
# the start + end vertices of each fold vector
#
# For example, an entry [0, 1] would correspond to the fold vector 
# pointing *from* point 0 *towards* point 1
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
fold_angle_lower_bound = np.array([
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






#####################################################################################
#
# Calculated data
#
#####################################################################################

# The total number of reference points (i.e. vertices)
num_reference_ponts = np.shape(reference_points)[0]

# The total number of folds (i.e. creases) 
num_folds = np.shape(fold_vector_points)[0] 

# The total number of interior fold intersections (i.e. interior vertices)
num_fold_intersections = np.shape(intersection_fold_indices)[0]

# The total number of faces
num_faces = np.shape(face_boundary)[0]

assert len(fold_angle_upper_bound) == num_folds 
assert len(fold_angle_lower_bound) == num_folds 
assert len(fold_angle_initial_value) == num_folds 

# The number of folds emanating from each interior fold intersection
num_intersection_folds = np.zeros(num_fold_intersections, dtype=np.int8)

for i in range(num_fold_intersections):
	count = np.count_nonzero(intersection_fold_indices[i] != filler_index)
	num_intersection_folds[i] = count

# The starting points of each fold vector
p1 = np.array([reference_points[i] for i, _ in fold_vector_points])
assert p1.shape == (num_folds, 2)

# The ending points of each fold vector
p2 = np.array([reference_points[j] for _, j in fold_vector_points])
assert p2.shape == (num_folds, 2)

# A direction vector along each fold (not necessarily oriented inwards or outwards)
fold_vector = p2 - p1
assert fold_vector.shape == (num_folds, 2)

# The angle that each fold vector makes w.r.t. the positive x-axis
fold_ref_angle_wrt_e1 = np.zeros((num_folds, 1))

def complete_acos(x, y):
	'''Given the 2D coordinates `x` and `y` corresponding to a *normalized*
	direction vector, calculate the angle that <xy> makes with the positive
	x-axis, [0..360]

	This is in contrast to `math.acos(...)` which always returns angles in 
	the range [0..180]

	'''
	if y >= 0.0: 
		return math.acos(x)
	
	return 2.0 * math.pi - math.acos(x)

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
# k -> index of the coordinate (x or y): thus, `k` should always be 0 or 1
# 
# When indexing into the second dimension of this array, `j`, one should always use a 
# loop like:
# 
# 	for j in range(num_intersection_folds[i])
#		...
#
# This is because different interior fold intersections might have different numbers
# of surrounding folds, so we want to avoid referencing any "extra" / "filler" folds
# that implicitly "pad" the `i`th interior fold intersection
fold_direction_i = np.zeros((num_fold_intersections, np.max(num_intersection_folds), 2))

# The angle that each interior fold makes with e1
fold_ref_angle_wrt_e1_i = np.zeros((num_fold_intersections, np.max(num_intersection_folds)))

for i in range(num_fold_intersections):

	# See notes above...
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

for i in range(num_fold_intersections):

	print(f'Interior fold intersection #{i} angles w.r.t. +x-axis:')

	for j in range(num_intersection_folds[i]):
		# Retrieve the *global* fold index of the fold that corresponds to the `j`th fold
		# emanating from the `i`th interior fold intersection
		global_fold_index = intersection_fold_indices[i][j]
		start_index, end_index = fold_vector_points[global_fold_index]
		theta = fold_ref_angle_wrt_e1_i[i][j]

		print(f'\tFold #{j} (corresponding to fold index {global_fold_index}, with reference point indices [{start_index}-{end_index}]) forms angle {math.degrees(theta)} w.r.t. +x-axis')

# The face corner angles surrounding each interior fold intersection
#
# TODO: bug in the code below...
# face_corner_angles = np.zeros((num_fold_intersections, max(num_intersection_folds)))

# for i in range(num_fold_intersections):

# 	for j in range(num_intersection_folds[i]):
# 		# If this is the last fold that emanates from the `i`th interior fold intersection, 
# 		# 
# 		# 
# 		# Otherwise, the face corner angle is simply the difference between the next
# 		# fold angle and the current fold angle
# 		if j == num_intersection_folds[i] - 1:
# 			face_corner_angles[i][j] = 2.0 * math.pi + (fold_ref_angle_wrt_e1_i[i][0] - fold_ref_angle_wrt_e1_i[i][j])
# 		else:
# 			face_corner_angles[i][j] = fold_ref_angle_wrt_e1_i[i][j + 1] - fold_ref_angle_wrt_e1_i[i][j]

face_corner_angles = np.array([
	# Corner angles surrounding interior fold intersection #0
	[math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4]

	# Corner angles surrounding interior fold intersection #1
	# ...

	# Corner angles surrounding interior fold intersection #2
	# ...
])

for i in range(num_fold_intersections):

	print(f'Interior fold intersection #{i} corner angles:')

	for j in range(num_intersection_folds[i]):
		# Retrieve the *global* fold index of the fold that corresponds to the `j`th fold
		# emanating from the `i`th interior fold intersection
		global_fold_index = intersection_fold_indices[i][j]
		theta = face_corner_angles[i][j]

		print(f'\tCorner angle #{j} (corresponding to fold index {global_fold_index}) has angle: {math.degrees(theta)}')

# The corner points of each polygonal face, which may or may not be triangular
#
# Indices:
# i -> index of the face
# j -> index of the `j`th corner point bounding the `i`th face
# k -> index of the coordinate (x or y): thus, `k` should always be 0 or 1
#
# The maximum number of boundary folds for a given face is dictated by the 
# size of the 2nd dimension of the `face_boundary` array, which will be 
# right-filled to accomodate any faces that have less than `max_boundary_folds_per_face`
# boundary folds
max_boundary_folds_per_face = np.shape(face_boundary)[1]

face_corner_points = np.zeros((num_faces, 2 * max_boundary_folds_per_face, 2))

# The number of corner points per face, 1D array
num_face_corner_points = np.zeros(num_faces, dtype=np.int8)

# Figure out which reference points form this polygonal face, in CCW order
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

# Figure out the center point of each face (for labeling purposes) - note that this is
# not required for simulation
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
# 
# Note that these are *global* fold indices (not indices in reference to any particular interior  
# fold intersection)
#
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





#####################################################################################
#
# Rotations, constraints, and other helper functions
#
#####################################################################################
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

def to_4x4(mat_3x3):
	'''Converts a 3x3 matrix into a 4x4 matrix with a `1` in the last
	row / column

	'''
	last_col = np.zeros((3, 1))
	last_row = np.zeros((1, 4))
	last_row[0, -1] = 1
		
	mat_4x4 = np.hstack((mat_3x3, last_col))
	mat_4x4 = np.vstack((mat_4x4, last_row))

	return mat_4x4

def translation(vector):
	'''Computes a 4x4 translation matrix along `vector`

	'''
	return np.array([
		[1, 0, 0, vector[0]],
		[0, 1, 0, vector[1]],
		[0, 0, 1, vector[2]],
		[0, 0, 0, 1],
	])

def get_fold_transform(alpha, phi, b):
	'''Computes the transform associated with a rotation of `phi` 
	radians around a fold that makes an angle `alpha` w.r.t. the 
	positive x-axis (e1)
	
	See equation `2.23` in the text

	Note that in the text, the "full" fold transformation is actually
	a 4x4 matrix, where the last column represents a translation
	
	X is a point in the reference configuration, S_0
	x is a point in the current configuration, S_t

	The mapping from X -> x involves a series of "fold transformations,"
	based on the folds crossed by a path connecting X to x

	Each fold transformation is represented by a 4x4 matrix, which itself
	is the composition of several transformations:

	1. Translate X by -b
	2. Rotate X by the inverse of R3(α): this aligns the fold vector 
	   to e1 (the positive x-axis)
	3. Rotate X by R1(φ): this performs the "actual" rotation induced
	   by the crease
	4. Rotate X by R3(α) to undo the rotation done in step (2)
	5. Translate X by b to undo the translation done in step (1)

	In the description above, `b` is taken to be the starting reference
	point of the fold in question, i.e. an element from `p1`
	
	Note that in the paper, this is all done with a single 4x4 matrix,
	but since SciPy's rotations are represented as 3x3 matrices, we 
	break it up into a number of sub-steps

	Also note that care should be taken when accumulating fold transforms
	about the way in which the desired fold path crosses each of its
	constituent folds: a fold can be crossed in either the "positive" 
	or "negative" direction

	If we cross a fold in the "positive" direction, we simply apply
	the rotations as described above
	
	If we cross a fold in the "negative" direction, we add `pi` to φ
	before constructing the rotation matrix

	'''	

	# Note that: r3(-alpha) = r3(alpha).inv()
	rotation = (r3(alpha) * r1(phi) * r3(-alpha)).as_matrix()[0]
	rotation = to_4x4(rotation)

	# Calculate the two transformation matrices associated with `b`
	t_inv = translation(-b)
	t = translation(b)

	# Calculate the fold transformation matrix associated with this fold,
	# which itself is the composite of 2 translation matrices and 3 
	# rotation matrices
	fold_transformation = np.matmul(t, np.matmul(rotation, t_inv))

	return fold_transformation

def get_kinematic_constraint(corner_angles, fold_angles):
	'''Constructs the kinematic constraint matrix - in a "valid" configuration,
	this matrix should be close (or equal) to the identity matrix
	
	Args:
		corner_angles: a 1D array containing all of the face corner angles
			around this interior fold intersection

		fold_angles: a 1D array containing all of the desired fold angles
			at each of the folds emanating from this interior fold intersection

	'''
	constraint_matrix = R.identity()

	for theta, alpha in zip(fold_angles, corner_angles):
		constraint_matrix = constraint_matrix * (r1(theta) * r3(alpha))

	return constraint_matrix

def build_face_map(fold_angles):
	face_map = np.zeros((num_faces, 4, 4))

	for face_index in range(num_faces):
		# Create a 4x4 identity matrix
		composite = np.eye(4, 4)

		# Traverse the fold path and accumulate transformation matrices
		for fold_index in fold_paths[face_index]:

			# There are no more "actual" folds along this path, so terminate
			if fold_index == filler_index:
				break

			# TODO: check if fold is crossed in either the positive or negative 
			# direction and adjust accordingly 
			# ...

			alpha = fold_ref_angle_wrt_e1[fold_index]
			phi = fold_angles[fold_index]

			# `b` is the starting reference point along this fold - note that
			# we convert `b` to a 3-element vector with z implicitly set to 0
			# before continuing
			b = p1[fold_index]
			b = np.append(b, 0.0)

			fold_transformation = get_fold_transform(alpha, phi, b)

			# Accumulate transformations
			composite = np.matmul(composite, fold_transformation)

		face_map[face_index] = composite

	return face_map

def run_tests():
	print('Testing R1 (rotation around x-axis) with pi/8...')
	r1_test = r1(math.pi * 0.125)
	print(r1_test.as_matrix())
	# Should be:
	#
	# [[ 1.          0.          0.        ]
	#  [ 0.          0.92387953 -0.38268343]
	#  [ 0.          0.38268343  0.92387953]]

	print('Testing R3 (rotation around z-axis) with pi/8...')
	r3_test = r3(math.pi * 0.125)
	print(r3_test.as_matrix())
	# Should be:
	# 
	# [[ 0.92387953 -0.38268343  0.        ]
	#  [ 0.38268343  0.92387953  0.        ]
	#  [ 0.          0.          1.        ]]

	print(to_4x4(r3_test.as_matrix()))

	alpha = math.pi / 4
	phi = math.pi / 8
	print(f'Testing composite transform with: `alpha` = {alpha}, `phi` = {phi}')
	comp_test = get_fold_transform(alpha, phi)
	print(comp_test)
	# Should be:
	#
	# [[ 0.96193977  0.03806023  0.27059805]
	#  [ 0.03806023  0.96193977 -0.27059805]
	#  [-0.27059805  0.27059805  0.92387953]]


# Create the plot and set the size
fig3d = plt.figure()
#fig3d.scene.renderer.use_depth_peeling = 1
fig3d.canvas.set_window_title('Folded Configuration')
fig3d.tight_layout(pad=2)
fig3d.tight_layout(rect=[0,0,.5,0.5]) 
ax3d = Axes3D(fig3d)
size = 50
ax3d.set_xlim3d(0, size)
ax3d.set_ylim3d(0, size)
ax3d.set_zlim3d(0, size)
ax3d.tick_params(axis='both', which='major', labelsize=6)
ax3d.tick_params(axis='both', which='minor', labelsize=6)

# ax3d.set_xticks([]) 
# ax3d.set_yticks([]) 
# ax3d.set_zticks([])
def plot_reference_configuration():
	for face_index in range(num_faces):
		# Grab all of the 2D corner points that form this face
		points_2d = face_corner_points[face_index][:num_face_corner_points[face_index]]
		z_coords = np.zeros((np.shape(points_2d)[0], 1))

		# "Expand" the 2D coordinates into 3D by adding zeros 
		points_3d = np.hstack((points_2d, z_coords))

		# Draw polygon and center 
		poly = Poly3DCollection([points_3d + [size * 0.5, size * 0.5, 0]])
		poly.set_edgecolor('k')

		ax3d.add_collection3d(poly)

def plot_custom_configuration(fold_angles):
	# Matplotlib utility for mapping face indices to colors
	scalar_color_map = ScalarMappable(norm=matplotlib.cm.colors.Normalize(0, num_faces), 
									  cmap=plt.get_cmap('plasma')) # or 'autumn', 'winter'

	if len(fold_angles) != num_folds:
		raise Error("Invalid number of fold angles")

	# Reset the collections object to effectively clear the screen...is 
	# there a better way to do this?
	ax3d.collections = []

	face_map = build_face_map(fold_angles)

	# Add all face polygons to one array (so that depth testing works)
	all_polys = np.zeros((num_faces, np.max(num_face_corner_points), 3))

	for face_index in range(num_faces):

		# Grab all of the 2D corner points that form this face
		points_2d = face_corner_points[face_index][:num_face_corner_points[face_index]]
		extra_0 = np.zeros((np.shape(points_2d)[0], 1))
		extra_1 = np.ones((np.shape(points_2d)[0], 1))

		# "Expand" the 2D coordinates into 4D by adding zeros and ones
		points_3d = np.hstack((points_2d, extra_0))
		points_4d = np.hstack((points_3d, extra_1))

		# Grab the 4x4 transformation matrix
		composite = face_map[face_index]

		for point_index in range(num_face_corner_points[face_index]):
			# Transform this corner point by the composite matrix
			points_4d[point_index] = np.dot(composite, points_4d[point_index])

		# Add a new polygon (drop the w-coordinate)
		all_polys[face_index] = points_4d[:,:3] + [size * 0.5, size * 0.5, 0.0]# + [size * 0.5 - face_centers[fixed_face][0], size * 0.5 - face_centers[fixed_face][1], 0]

	poly = Poly3DCollection(all_polys)
	poly.set_facecolor([scalar_color_map.to_rgba(face_index)[:3] for face_index in range(num_faces)])
	poly.set_edgecolor('k')
	poly.set_alpha(0.75)
	ax3d.add_collection3d(poly)

def test_plot_custom_configuration():
	# Should be 8 values (8 unique folds):
	#
	# M
	# V
	# M
	# V
	# V
	# M
	# V
	# M
	#
	# Again:
	# 
	# M fold angles are negative (-)
	# V fold angles are positive (+)
	# 
	# For this crease pattern:
	# 
	# 	All M folds should achieve a fold angle of -90 degrees
	# 	All V folds should achieve a fold angle of 180 degrees
	#
	# Final configuration:
	custom_fold_angles = np.array([
		 -math.pi * 0.5, 
		  math.pi, 
		 -math.pi * 0.5,
		  math.pi,			# Horizontal fold (index #3)
		  math.pi,			# Horizontal fold (index #4)
		 -math.pi * 0.5, 
		  math.pi, 
		 -math.pi * 0.5
	])

	plot_custom_configuration(custom_fold_angles)

def test_constraint_matrix():
	# Sanity check: our reference configuration S_0 should always work!
	test_corner_angles = np.array([
		math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4
	])

	test_fold_angles = np.array([
		0, 0, 0, 0, 0, 0, 0, 0
	])

	test_constraint = get_kinematic_constraint(test_corner_angles, test_fold_angles).as_matrix()
	print(test_constraint)
	print(f'Close to identity matrix: { np.allclose(test_constraint, np.eye(3)) }')

	# Test cube corner from the text, `Example 2.9`
	test_corner_angles = np.array([
		math.pi / 2, 
		math.pi / 2, 
		math.pi / 4, 
		math.pi / 4, 
		math.pi / 2
	])

	test_fold_angles = np.array([
		 math.pi / 2,
		 math.pi / 2,
		 math.pi / 2,
		-math.pi,
		 math.pi
	])

	test_constraint = get_kinematic_constraint(test_corner_angles, test_fold_angles).as_matrix()
	print(test_constraint)
	print(f'Close to identity matrix: { np.allclose(test_constraint, np.eye(3)) }')


	test_corner_angles = np.array([
		math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4
	])

	test_fold_angles = np.array([
		-math.pi * 0.5, 
		 math.pi, 
		-math.pi * 0.5,
		 math.pi,			# Horizontal fold (index #3)
		 math.pi,			# Horizontal fold (index #4)
		-math.pi * 0.5, 
		 math.pi, 
		-math.pi * 0.5
	])

	test_fold_angles *= 0.75

	test_constraint = get_kinematic_constraint(test_corner_angles, test_fold_angles).as_matrix()
	print(test_constraint)
	print(f'Close to identity matrix: { np.allclose(test_constraint, np.eye(3)) }')


# test_constraint_matrix()

def plot_crease_pattern():
	colors = np.random.rand(len(fold_vector_points), 3)

	line_segments = [[a, b] for a, b in zip(p1, p2)]
	line_segment_midpoints = [(a + b) * 0.5 for a, b in zip(p1, p2) ]

	line_collection = collections.LineCollection(line_segments, colors=colors, linewidths=1)
	fig, ax = plt.subplots()
	fig.canvas.set_window_title('Crease Pattern')
	plt.subplots_adjust(bottom=0.25)
	ax.tick_params(axis='both', which='major', labelsize=6)
	ax.tick_params(axis='both', which='minor', labelsize=6)
	ax.add_collection(line_collection)
	ax.scatter(reference_points[:, 0], reference_points[:, 1], zorder=2)

	aspect_ratio = 1.0
	xleft, xright = ax.get_xlim()
	ybottom, ytop = ax.get_ylim()
	ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * aspect_ratio)

	for i, (x, y) in enumerate(line_segment_midpoints):
		label = 'F{}'.format(i)
		ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 10), ha='center', fontsize='medium', fontweight='bold')

	for i, (x, y) in enumerate(reference_points):
		label = 'v{}'.format(i)
		ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize='x-small')

	for i, (x, y) in enumerate(face_centers):
		label = 'face {}'.format(i)
		ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 0), ha='center', fontsize='x-small')

plot_crease_pattern()












#####################################################################################
#
# Simulation parameters
#
#####################################################################################
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

# The expected fold angle change per increment, in order to reach the desired 
# configuration
fold_angle_change_per_increment = np.zeros((num_increments, num_folds))

# Reference angle for filling fold angle change per increment matrix
reference_fold_angle = math.radians(90.0)

# Fold angle change per increment: these are essentially the "guess increments" from the text
#
# We assume that each fold moves linearly from its initial value to its final (expected) value,
# which, of course, is not the case, in reality
# 
# However, these guess increments will be modified through the procedure outlined in the solver loop
for i in range(num_increments):
	fold_angle_change_per_increment[i] = (reference_fold_angle / num_increments) * np.array([-1, 2, -1, 2, 2, -1, 2, -1])

# Input to determine one of two methods for calculating initial fold angle increment. One uses the previous 
# Jacobian (corrected; Select 1) and the other one does not (not corrected; Select 2) 
increment_correction = 1

# Finite difference step in fold angles for calculation of derivatives
fin_diff_step = math.radians(2.0)

# At each increment, two constraints must be satisfied:
#
# 1. Kinematic constraints
# 2. Lower and upper bounds of the fold angles
#
# In English, (1) states that the rotation constraint matrix corresponding to each interior fold
# intersection should be equal to the 4x4 identity matrix (this constraint applies to ALL of the
# interior fold intersections, simultaneously)
# 
# Meanwhile, (2) states that each of the fold angles should not exceed (or drop below) its
# upper / lower bound, respectively
history_fold_angles = np.zeros((num_increments + 1, num_folds))

simulate = True 

d_residual_d_fold_angles = np.zeros((3 * num_fold_intersections + 2 * num_folds, num_folds))

if simulate:
	print('Starting solver...')

	for increment in range(num_increments + 1):
		print(f'Increment: {increment}')

		if increment == 0:
			# The residual vector derivatives don't yet exist, so start with the initial values
			history_fold_angles[increment] = fold_angle_initial_value
		else:
			# Calculate the projection of the guess increments for the fold angles
			# onto the null space of the residual vector derivatives from the previous
			# configuration
			delta = np.dot(np.eye(num_folds) - np.matmul(np.linalg.pinv(d_residual_d_fold_angles), d_residual_d_fold_angles), fold_angle_change_per_increment[increment - 1])

			# Set the new fold angles to the previous fold angles plus the projection of the guess 
			# increments vector on the null space of the derivatives of the residual vector
			# 
			# Resource: `https://ocw.mit.edu/courses/mathematics/18-06sc-linear-algebra-fall-2011/least-squares-determinants-and-eigenvalues/projections-onto-subspaces/MIT18_06SCF11_Ses2.2sum.pdf`
			history_fold_angles[increment] = history_fold_angles[increment - 1] + delta

		for iteration in range(max_iterations):

			print(f'\tIteration: {iteration}')

			# The dimension of the residual vector is determined by:
			# 
			# 3 * num_fold_intersections -> kinematic constraints
			# 2 * num_folds -> upper and lower bounds of each fold angle
			residual = np.zeros((3 * num_fold_intersections + 2 * num_folds))

			# Each column `c` of the Jacobian matrix corresponds to the partial deriative of the 
			# residual vector w.r.t. the `c`th fold angle
			d_residual_d_fold_angles = np.zeros((3 * num_fold_intersections + 2 * num_folds, num_folds))


			# Grab all of the fold angles associated with the folds that emanate from
			# each of the interior fold intersections - this is needed in order to 
			# calculate each of the kinematic constraint matrices below
			intersection_fold_angles = np.zeros((num_fold_intersections, max(num_intersection_folds)))
			assert np.shape(intersection_fold_angles) == (1, 8)

			for i in range(num_fold_intersections):
				
				# In our case, `i` only evaluates to: 0

				for j in range(num_intersection_folds[i]):

					# In our case, `j` evaluates to: 0, 1, 2, 3, 4, 5, 6, 7, 8 
					# 8 folds total that emanate from the `0`th (and only) interior fold intersection

					intersection_fold_angles[i][j] = history_fold_angles[increment][intersection_fold_indices[i][j]]

			# Matrix-type constraints
			for i in range(num_fold_intersections):

				# Grab the corner angles and fold angles that are relevant to the `i`th 
				# interior fold intersection
				i_corner_angles = face_corner_angles[i][:num_intersection_folds[i]]
				i_fold_angles = intersection_fold_angles[i][:num_intersection_folds[i]]

				# Calculate the constraint matrix
				constraint_matrix = get_kinematic_constraint(i_corner_angles, i_fold_angles).as_matrix()
			
				# Because the constraint matrix is orthogonal (it is a composition of rotations), only 3 of 
				# its 9 components are independent (i.e., can be chosen freely):
				#
				# 		[a b c]
				#  		[d e f]
				#  		[g h i]
				#
				# The constraints imposed by the elements in the upper triangle of the matrix are equivalent
				# to those imposed by the elements in the lower triangle of the matrix
				#
				# By examining the resulting rotation matrix, we also see that `c` will always be zero
				# 
				# So, we choose to examine `b`, `f` and `g`
				val_12 = 0.5 * weight_rotation_constraint * math.pow(constraint_matrix[1][2], 2.0)
				val_20 = 0.5 * weight_rotation_constraint * math.pow(constraint_matrix[2][0], 2.0)
				val_01 = 0.5 * weight_rotation_constraint * math.pow(constraint_matrix[0][1], 2.0)

				residual[i * 3 + 0] = val_12
				residual[i * 3 + 1] = val_20
				residual[i * 3 + 2] = val_01

				# Calculate the first `3N_I + 2N_F` components of the derivative of the residual vector
				# w.r.t. each of the fold angles using a central finite difference

				for j in range(num_intersection_folds[i]):

					# In our case, `i` only evaluates to: 0
					# Only 1 interior fold intersection
					#
					# In our case, `j` evaluates to: 0, 1, 2, 3, 4, 5, 6, 7, 8 
					# 8 folds total that emanate from the `0`th (and only) interior fold intersection
					#
					# So, `intersection_fold_indices[i][j]` returns the index of the fold (in the *global* array
					# that contains ALL of the folds) of the `j`th fold surrounding the `i`th interior fold
					# intersection...

					forward_step = np.zeros_like(i_fold_angles)
					forward_step[j] = fin_diff_step
					forward = get_kinematic_constraint(i_corner_angles, i_fold_angles + forward_step).as_matrix()

					backward_step = np.zeros_like(i_fold_angles)
					backward_step[j] = -fin_diff_step
					backward = get_kinematic_constraint(i_corner_angles, i_fold_angles + backward_step).as_matrix()

					# Compute central difference
					#
					# Reference: `http://www.math.unl.edu/~s-bbockel1/833-notes/node23.html`
					d_constraint_matrix = (forward - backward) / (2.0 * fin_diff_step)

					# Update components of the Jacobian related to the fold index of the `j`th fold
					# surrounding the `i`th interior fold intersection 
					d_residual_d_fold_angles[i * 3 + 0][intersection_fold_indices[i][j]] = weight_rotation_constraint * d_constraint_matrix[1][2] * constraint_matrix[1][2]
					d_residual_d_fold_angles[i * 3 + 1][intersection_fold_indices[i][j]] = weight_rotation_constraint * d_constraint_matrix[2][0] * constraint_matrix[2][0]
					d_residual_d_fold_angles[i * 3 + 2][intersection_fold_indices[i][j]] = weight_rotation_constraint * d_constraint_matrix[0][1] * constraint_matrix[0][1]

			# Lower / upper bound-type constraints
			for i in range(num_folds):
				# Put into residual vector, starting at index `3 * num_fold_intersections` where
				# `num_fold_intersections` is the TOTAL number of interior fold intersections across
				# the entire crease pattern
				insertion_start = 3 * num_fold_intersections

				# `2.76`
				val_lower_bound = 0.5 * weight_fold_angle_bounds * max(0.0, -history_fold_angles[increment][i] + fold_angle_lower_bound[i])

				# `2.77`
				val_upper_bound = 0.5 * weight_fold_angle_bounds * max(0.0,  history_fold_angles[increment][i] - fold_angle_upper_bound[i])

				residual[insertion_start + 2 * i + 0] = val_lower_bound
				residual[insertion_start + 2 * i + 1] = val_upper_bound

				# Equation `2.76` effectively simplies to a function of the form: 
				# 
				#		`c * max(0, -x + y)^2`
				#
				# According to the equation above, if `x` is greater than or equal to `y`,
				# `max(...)` will return zero, in which case the derivative is also zero
				# 
				# Otherwise, we can use U-substitution to find that the partial derivative 
				# of the function `f(x, y)` w.r.t. `x` is:
				#
				# 		`c * (2x - 2y)`
				#
				# which, in our case, simplifies to the second branch below
				if history_fold_angles[increment][i] >= fold_angle_lower_bound[i]:
					d_residual_d_fold_angles[insertion_start + 2 * i + 0][i] = 0.0
				else:
					d_residual_d_fold_angles[insertion_start + 2 * i + 0][i] = -weight_fold_angle_bounds * (-history_fold_angles[increment][i] + fold_angle_lower_bound[i])
			
				# Equation `2.77` follows
				if history_fold_angles[increment][i] <= fold_angle_upper_bound[i]:
					d_residual_d_fold_angles[insertion_start + 2 * i + 1][i] = 0.0
				else:
					d_residual_d_fold_angles[insertion_start + 2 * i + 1][i] = weight_fold_angle_bounds * (history_fold_angles[increment][i] - fold_angle_upper_bound[i])

			# Numpy `norm(...)` defaults to the L2 norm
			norm_residual = np.linalg.norm(residual) / np.shape(residual)[0]
			
			if norm_residual < tolerance_residual:
				print('The L2 norm of the residual vector is less than the tolerance: fold angle corrections are not necessary - continuing...')
				break
			else:
				# Calculate the fold angle corrections from the Jacobian matrix and the residual vector
				fold_angle_corrections = np.dot(-np.linalg.pinv(d_residual_d_fold_angles), residual)
				assert np.shape(fold_angle_corrections) == (num_folds,)

				norm_fold_angle_corrections = np.linalg.norm(fold_angle_corrections) / np.shape(fold_angle_corrections)[0]

				# Don't allow super large corrections
				if max(fold_angle_corrections) > max_fold_angle_correction:
					print('Rescaling `fold_angle_corrections: too large...')
					fold_angle_corrections = fold_angle_corrections * max_fold_angle_correction / max(fold_angle_corrections)

				# Apply the fold angle corrections to the current fold angles
				history_fold_angles[increment] = history_fold_angles[increment] + fold_angle_corrections

				if norm_fold_angle_corrections < tolerance_fold_angle:
					print('The L2 norm of the fold angle corrections vector is less than the tolerance: exiting solver')
					break



# Display the results!
ax_ui = plt.axes([0.25, 0.125, 0.5, 0.03])
slider_init = num_increments
slider_increment = Slider(ax_ui, 'Increment', 0, num_increments, valinit=slider_init)
slider_increment.label.set_size(6)
slider_increment.valtext.set_size(6)

def update(val):
	'''A small callback function to redraw the UI whenever the slider changes

	'''
	plot_custom_configuration(history_fold_angles[int(slider_increment.val)])
	fig3d.canvas.draw_idle()

slider_increment.on_changed(update)
update(slider_init)

# Some bullshit to close all Matplotlib windows when you hit "ENTER" 
plt.show(block=False)
input("Hit Enter To Close")
plt.close()