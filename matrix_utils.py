import math
import numpy as np

def r1(phi):
	'''Computes a 4x4 rotation matrix corresponding to a rotation of `phi`
	radians around the positive x-axis (e1)

	'''
	c = math.cos(phi)
	s = math.sin(phi)

	return np.array([
		[1, 0,  0, 0],
		[0, c, -s, 0],
		[0, s,  c, 0],
		[0, 0,  0, 1],
	])

def r3(phi):
	'''Computes a 4x4 rotation matrix corresponding to a rotation of `phi`
	radians around the positive z-axis (e3)
	
	'''
	c = math.cos(phi)
	s = math.sin(phi)

	return np.array([
		[c, -s,  0, 0],
		[s,  c,  0, 0],
		[0,  0,  1, 0],
		[0,  0,  0, 1],
	])

def translation(v):
	'''Computes a 4x4 translation matrix along vector `v`

	'''
	x, y, z = v

	return np.array([
		[1, 0, 0, x],
		[0, 1, 0, y],
		[0, 0, 1, z],
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

	rotation = np.matmul(r3(alpha), np.matmul(r1(phi), r3(-alpha)))

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
	constraint_matrix = np.eye(4, 4)

	for theta, alpha in zip(fold_angles, corner_angles):
		constraint_matrix = np.matmul(constraint_matrix, np.matmul(r1(theta), r3(alpha)))

	return constraint_matrix

if __name__ == "__main__":
	print('Testing R1 (rotation around x-axis) with pi/8...')
	r1_test = r1(math.pi * 0.125)
	print(r1_test)
	# Should be:
	#
	# [[ 1.          0.          0.        ]
	#  [ 0.          0.92387953 -0.38268343]
	#  [ 0.          0.38268343  0.92387953]]

	print('Testing R3 (rotation around z-axis) with pi/8...')
	r3_test = r3(math.pi * 0.125)
	print(r3_test)
	# Should be:
	# 
	# [[ 0.92387953 -0.38268343  0.        ]
	#  [ 0.38268343  0.92387953  0.        ]
	#  [ 0.          0.          1.        ]]

	print(r3_test)

	# Sanity check: our reference configuration S_0 should always work!
	test_corner_angles = np.array([
		math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4
	])

	test_fold_angles = np.array([
		0, 0, 0, 0, 0, 0, 0, 0
	])

	test_constraint = get_kinematic_constraint(test_corner_angles, test_fold_angles)
	print(test_constraint)
	print(f'Close to identity matrix: { np.allclose(test_constraint, np.eye(4)) }')

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

	test_constraint = get_kinematic_constraint(test_corner_angles, test_fold_angles)
	print(test_constraint)
	print(f'Close to identity matrix: { np.allclose(test_constraint, np.eye(4)) }')


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

	test_constraint = get_kinematic_constraint(test_corner_angles, test_fold_angles)
	print(test_constraint)
	print(f'Close to identity matrix: { np.allclose(test_constraint, np.eye(4)) }')