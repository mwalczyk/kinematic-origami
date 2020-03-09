import math 
import numpy as np

import matrix_utils as mu

class Solver:

	def __init__(self):
		# Weight for residuals from rotation constraints
		self.weight_rotation_constraint = 1

		# Weight for residuals from fold angle bound constraints
		self.weight_fold_angle_bounds = 1e-5

		# Tolerance for norm of residual vector: https://en.wikipedia.org/wiki/Residual_(numerical_analysis)
		self.tolerance_residual = 1e-8

		# Tolerance for correction of fold angles
		self.tolerance_fold_angle = 1e-8

		# Maximum number of iterations allowed for each increment
		self.max_iterations = 15

		# Maximum correction allowed for individual fold angles in each iteration
		self.max_fold_angle_correction = math.radians(5.0)

		# Number of increments
		self.num_increments = 50

		# Finite difference step in fold angles for calculation of derivatives
		self.fin_diff_step = math.radians(2.0)

	def run(self, crease_pattern, verbose=True):
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
		history_fold_angles = np.zeros((self.num_increments + 1, crease_pattern.num_folds))

		d_residual_d_fold_angles = np.zeros((3 * crease_pattern.num_fold_intersections + 2 * crease_pattern.num_folds, crease_pattern.num_folds))

		# Fold angle change per increment: these are essentially the "guess increments" from the text
		#
		# We assume that each fold moves linearly from its initial value to its final (expected) value,
		# which, of course, is not the case, in reality
		# 
		# However, these guess increments will be modified through the procedure outlined in the solver loop
		fold_angle_change_per_increment = np.tile(crease_pattern.fold_angle_target / float(self.num_increments), (self.num_increments, 1))

		for increment in range(self.num_increments + 1):
			print(f'Increment: {increment}')

			if increment == 0:
				# The residual vector derivatives don't yet exist, so start with the initial values
				history_fold_angles[increment] = crease_pattern.fold_angle_initial_value
			else:
				# Calculate the projection of the guess increments for the fold angles
				# onto the null space of the residual vector derivatives from the previous
				# configuration
				delta = np.dot(np.eye(crease_pattern.num_folds) - np.matmul(np.linalg.pinv(d_residual_d_fold_angles), d_residual_d_fold_angles), fold_angle_change_per_increment[increment - 1])

				# Set the new fold angles to the previous fold angles plus the projection of the guess 
				# increments vector on the null space of the derivatives of the residual vector
				# 
				# Resource: `https://ocw.mit.edu/courses/mathematics/18-06sc-linear-algebra-fall-2011/least-squares-determinants-and-eigenvalues/projections-onto-subspaces/MIT18_06SCF11_Ses2.2sum.pdf`
				history_fold_angles[increment] = history_fold_angles[increment - 1] + delta

			for iteration in range(self.max_iterations):

				print(f'\tIteration: {iteration}')

				# The dimension of the residual vector is determined by:
				# 
				# 3 * num_fold_intersections -> kinematic constraints
				# 2 * num_folds -> upper and lower bounds of each fold angle
				residual = np.zeros((3 * crease_pattern.num_fold_intersections + 2 * crease_pattern.num_folds))

				# Each column `c` of the Jacobian matrix corresponds to the partial deriative of the 
				# residual vector w.r.t. the `c`th fold angle
				d_residual_d_fold_angles = np.zeros((3 * crease_pattern.num_fold_intersections + 2 * crease_pattern.num_folds, crease_pattern.num_folds))


				# Grab all of the fold angles associated with the folds that emanate from
				# each of the interior fold intersections - this is needed in order to 
				# calculate each of the kinematic constraint matrices below
				intersection_fold_angles = np.zeros((crease_pattern.num_fold_intersections, max(crease_pattern.num_intersection_folds)))
				assert np.shape(intersection_fold_angles) == (1, 8)

				for i in range(crease_pattern.num_fold_intersections):
					
					# In our case, `i` only evaluates to: 0

					for j in range(crease_pattern.num_intersection_folds[i]):

						# In our case, `j` evaluates to: 0, 1, 2, 3, 4, 5, 6, 7, 8 
						# 8 folds total that emanate from the `0`th (and only) interior fold intersection

						intersection_fold_angles[i][j] = history_fold_angles[increment][crease_pattern.intersection_fold_indices[i][j]]

				# Matrix-type constraints
				for i in range(crease_pattern.num_fold_intersections):

					# Grab the corner angles and fold angles that are relevant to the `i`th 
					# interior fold intersection
					i_corner_angles = crease_pattern.face_corner_angles[i][:crease_pattern.num_intersection_folds[i]]
					i_fold_angles = intersection_fold_angles[i][:crease_pattern.num_intersection_folds[i]]

					# Calculate the constraint matrix
					constraint_matrix = mu.get_kinematic_constraint(i_corner_angles, i_fold_angles)
				
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
					val_12 = 0.5 * self.weight_rotation_constraint * math.pow(constraint_matrix[1][2], 2.0)
					val_20 = 0.5 * self.weight_rotation_constraint * math.pow(constraint_matrix[2][0], 2.0)
					val_01 = 0.5 * self.weight_rotation_constraint * math.pow(constraint_matrix[0][1], 2.0)

					residual[i * 3 + 0] = val_12
					residual[i * 3 + 1] = val_20
					residual[i * 3 + 2] = val_01

					# Calculate the first `3N_I + 2N_F` components of the derivative of the residual vector
					# w.r.t. each of the fold angles using a central finite difference

					for j in range(crease_pattern.num_intersection_folds[i]):

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
						forward_step[j] = self.fin_diff_step
						forward = mu.get_kinematic_constraint(i_corner_angles, i_fold_angles + forward_step)

						backward_step = np.zeros_like(i_fold_angles)
						backward_step[j] = -self.fin_diff_step
						backward = mu.get_kinematic_constraint(i_corner_angles, i_fold_angles + backward_step)

						# Compute central difference
						#
						# Reference: `http://www.math.unl.edu/~s-bbockel1/833-notes/node23.html`
						d_constraint_matrix = (forward - backward) / (2.0 * self.fin_diff_step)

						# Update components of the Jacobian related to the fold index of the `j`th fold
						# surrounding the `i`th interior fold intersection 
						d_residual_d_fold_angles[i * 3 + 0][crease_pattern.intersection_fold_indices[i][j]] = self.weight_rotation_constraint * d_constraint_matrix[1][2] * constraint_matrix[1][2]
						d_residual_d_fold_angles[i * 3 + 1][crease_pattern.intersection_fold_indices[i][j]] = self.weight_rotation_constraint * d_constraint_matrix[2][0] * constraint_matrix[2][0]
						d_residual_d_fold_angles[i * 3 + 2][crease_pattern.intersection_fold_indices[i][j]] = self.weight_rotation_constraint * d_constraint_matrix[0][1] * constraint_matrix[0][1]

				# Lower / upper bound-type constraints
				for i in range(crease_pattern.num_folds):
					# Put into residual vector, starting at index `3 * crease_pattern.num_fold_intersections` where
					# `crease_pattern.num_fold_intersections` is the TOTAL number of interior fold intersections across
					# the entire crease pattern
					insertion_start = 3 * crease_pattern.num_fold_intersections

					# `2.76`
					val_lower_bound = 0.5 * self.weight_fold_angle_bounds * max(0.0, -history_fold_angles[increment][i] + crease_pattern.fold_angle_lower_bound[i])

					# `2.77`
					val_upper_bound = 0.5 * self.weight_fold_angle_bounds * max(0.0,  history_fold_angles[increment][i] - crease_pattern.fold_angle_upper_bound[i])

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
					if history_fold_angles[increment][i] >= crease_pattern.fold_angle_lower_bound[i]:
						d_residual_d_fold_angles[insertion_start + 2 * i + 0][i] = 0.0
					else:
						d_residual_d_fold_angles[insertion_start + 2 * i + 0][i] = -self.weight_fold_angle_bounds * (-history_fold_angles[increment][i] + crease_pattern.fold_angle_lower_bound[i])
				
					# Equation `2.77` follows
					if history_fold_angles[increment][i] <= crease_pattern.fold_angle_upper_bound[i]:
						d_residual_d_fold_angles[insertion_start + 2 * i + 1][i] = 0.0
					else:
						d_residual_d_fold_angles[insertion_start + 2 * i + 1][i] = self.weight_fold_angle_bounds * (history_fold_angles[increment][i] - crease_pattern.fold_angle_upper_bound[i])

				# Numpy `norm(...)` defaults to the L2 norm
				norm_residual = np.linalg.norm(residual) / np.shape(residual)[0]
				
				if norm_residual < self.tolerance_residual:
					print('The L2 norm of the residual vector is less than the tolerance: fold angle corrections are not necessary - continuing...')
					break
				else:
					# Calculate the fold angle corrections from the Jacobian matrix and the residual vector
					fold_angle_corrections = np.dot(-np.linalg.pinv(d_residual_d_fold_angles), residual)
					assert np.shape(fold_angle_corrections) == (crease_pattern.num_folds,)

					norm_fold_angle_corrections = np.linalg.norm(fold_angle_corrections) / np.shape(fold_angle_corrections)[0]

					# Don't allow super large corrections
					if max(fold_angle_corrections) > self.max_fold_angle_correction:
						print('Rescaling `fold_angle_corrections: too large...')
						fold_angle_corrections = fold_angle_corrections * self.max_fold_angle_correction / max(fold_angle_corrections)

					# Apply the fold angle corrections to the current fold angles
					history_fold_angles[increment] = history_fold_angles[increment] + fold_angle_corrections

					if norm_fold_angle_corrections < self.tolerance_fold_angle:
						print('The L2 norm of the fold angle corrections vector is less than the tolerance: exiting solver')
						break

		return history_fold_angles