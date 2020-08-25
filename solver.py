import logging
import math 
import numpy as np
import sys 

import matrix_utils as mu

class Solver:

	def __init__(self, **kwargs):
		prop_defaults = {
			# Weight for residuals from rotation constraints
			'weight_rotation_constraint': 1.0,

			# Weight for residuals from fold angle bound constraints
			'weight_fold_angle_bounds': 1e-5,

			# Tolerance for norm of residual vector: https://en.wikipedia.org/wiki/Residual_(numerical_analysis)
			'tolerance_residual': 1e-8,

			# Tolerance for correction of fold angles
			'tolerance_fold_angle': 1e-8,

			# Maximum number of iterations allowed for each increment
			'max_iterations': 15,

			# Maximum correction allowed for individual fold angles in each iteration
			'max_fold_angle_correction': math.radians(5.0),

			# Number of increments
			'num_increments': 50,

			# Finite difference step in fold angles for calculation of derivatives
			'fin_diff_step': math.radians(2.0),

			# Whether or not to use the projection onto the null space of the residual vector derivatives from 
			# the previous configuration to initialize the current iteration
			'use_projection': True
		}

		for (prop, default) in prop_defaults.items():
			setattr(self, prop, kwargs.get(prop, default))

	def calculate_residual(self, crease_pattern, history_fold_angles, increment):
		'''Given the crease pattern, the fold angle history, and the current increment, this 
		function calculates the residual vector and the derivative of the residual vector
		w.r.t. the current fold angles (Jacobian matrix).

		'''
		# The dimension of the residual vector is determined by:
		# 
		# 3 * N_I -> 3 rotation constraints per interior fold intersection
		# 2 * N_F -> 2 (upper and lower) bounds per fold angle
		residual = np.zeros((3 * crease_pattern.num_fold_intersections + 2 * crease_pattern.num_folds))

		# Each column `c` of the Jacobian matrix corresponds to the partial deriatives of the 
		# elements of the residual vector w.r.t. the `c`th fold angle
		#
		# In general, this matrix will be quite sparse:
		#
		# 	For the matrix-type constraints, each interior fold intersection "contributes" three 
		#	rows to the Jacobian (corresponding to entries R23, R31, and R12 of that fold intersection's
		#	constraint matrix). For example, one such row contains all of the partial derivatives of 
		#   R23 (for that fold intersection) w.r.t. each of the fold angles of the CP. However,
		#   only a select number of these folds actually emanate from this fold intersection and contribute
		#   non-zero entries to the Jacobian matrix. All fold angles that correspond to creases that are
		#   NOT adjacent to this fold intersection will be 0, since those folds weren't part of the
		#   calculation for that fold intersection's constraint matrix.
		#
		#	For the lower / upper bound-type constraints, each fold angle "contributes" two rows
		#	to the Jacobian (corresponding to the upper and lower bound constraints). For example, one
		#   such row contains all of the partial derivatives of the lower bound constraint on the first
		#   fold angle w.r.t. all of the fold angles. Obviously, only the first fold angle is involved 
		#   in this calculation, so all other entries across this row will be 0 except for the first.
		d_residual_d_fold_angles = np.zeros((3 * crease_pattern.num_fold_intersections + 2 * crease_pattern.num_folds, crease_pattern.num_folds))

		# Grab all of the fold angles associated with the folds that emanate from
		# each of the interior fold intersections - this is needed in order to 
		# calculate each of the kinematic constraint matrices below
		intersection_fold_angles = np.zeros((crease_pattern.num_fold_intersections, max(crease_pattern.num_intersection_folds)))
		for i in range(crease_pattern.num_fold_intersections):
			for j in range(crease_pattern.num_intersection_folds[i]):
				# Grab the fold angle corresponding to the j-th fold emanating from 
				# the i-th interior fold intersection
				intersection_fold_angles[i][j] = history_fold_angles[increment][crease_pattern.intersection_fold_indices[i][j]]

		# Matrix-type constraints
		for i in range(crease_pattern.num_fold_intersections):

			# Grab the corner angles and fold angles that are relevant to the i-th 
			# interior fold intersection
			i_corner_angles = crease_pattern.face_corner_angles[i][:crease_pattern.num_intersection_folds[i]]
			i_fold_angles = intersection_fold_angles[i][:crease_pattern.num_intersection_folds[i]]

			# Calculate the constraint matrix
			constraint_matrix = mu.get_rotation_constraint_matrix(i_corner_angles, i_fold_angles)
		
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
			#
			# Equation `2.75`
			val_12 = 0.5 * self.weight_rotation_constraint * math.pow(constraint_matrix[1][2], 2.0)
			val_20 = 0.5 * self.weight_rotation_constraint * math.pow(constraint_matrix[2][0], 2.0)
			val_01 = 0.5 * self.weight_rotation_constraint * math.pow(constraint_matrix[0][1], 2.0)

			residual[i * 3 + 0] = val_12
			residual[i * 3 + 1] = val_20
			residual[i * 3 + 2] = val_01

			# Calculate the first `3 * N_I + 2 * N_F` components of the derivative of the residual vector
			# w.r.t. each of the fold angles using a central finite difference
			for j in range(crease_pattern.num_intersection_folds[i]):
				# Calculate forwards 
				forward_step = np.zeros_like(i_fold_angles)
				forward_step[j] = self.fin_diff_step
				forward = mu.get_rotation_constraint_matrix(i_corner_angles, i_fold_angles + forward_step)

				# Calculate backwards
				backward_step = np.zeros_like(i_fold_angles)
				backward_step[j] = -self.fin_diff_step
				backward = mu.get_rotation_constraint_matrix(i_corner_angles, i_fold_angles + backward_step)

				# Compute central difference
				d_constraint_matrix = (forward - backward) / (2.0 * self.fin_diff_step)

				# Update components of the Jacobian related to the fold index of the j-th fold
				# surrounding the i-th interior fold intersection 
				#
				# See the notes above on the indices of `crease_pattern.intersection_fold_indices`
				d_residual_d_fold_angles[i * 3 + 0][crease_pattern.intersection_fold_indices[i][j]] = self.weight_rotation_constraint * d_constraint_matrix[1][2] * constraint_matrix[1][2]
				d_residual_d_fold_angles[i * 3 + 1][crease_pattern.intersection_fold_indices[i][j]] = self.weight_rotation_constraint * d_constraint_matrix[2][0] * constraint_matrix[2][0]
				d_residual_d_fold_angles[i * 3 + 2][crease_pattern.intersection_fold_indices[i][j]] = self.weight_rotation_constraint * d_constraint_matrix[0][1] * constraint_matrix[0][1]

		if increment == 1000:
			np.set_printoptions(threshold=sys.maxsize)
			for k in range(3 * crease_pattern.num_fold_intersections + 2 * crease_pattern.num_folds):
				print('row', k, ':', d_residual_d_fold_angles[k])

		# Lower / upper bound-type constraints
		for i in range(crease_pattern.num_folds):
			# Put into residual vector, starting at index `3 * crease_pattern.num_fold_intersections` where
			# `crease_pattern.num_fold_intersections` is the TOTAL number of interior fold intersections across
			# the entire crease pattern
			insertion_start = 3 * crease_pattern.num_fold_intersections

			# Equation `2.76`
			val_lower_bound = 0.5 * self.weight_fold_angle_bounds * max(0.0, -history_fold_angles[increment][i] + crease_pattern.fold_angle_lower_bound[i])

			# Equation `2.77`
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

		return (residual, d_residual_d_fold_angles)

	def run(self, crease_pattern, verbose=False):
		'''Runs the kinematic solver on the provided crease pattern, satisfying two 
		constraints at each increment:
		
		1. Kinematic constraints: the rotation constraint matrix corresponding to each interior fold
			intersection should be equal to the 4x4 identity matrix 
		2. Lower and upper bounds: each of the fold angles should not exceed (or drop below) its 
			upper / lower bound, respectively
		
		'''
		# Set up logging
		logger = logging.getLogger('solver')
		logger.setLevel(logging.DEBUG)
		handler = logging.StreamHandler()
		handler.setLevel(logging.INFO)
		if verbose:
			handler.setLevel(logging.DEBUG)
		logger.addHandler(handler)

		# Keep track of all of the fold angles at each increment so that the model can be animated later
		history_fold_angles = np.zeros((self.num_increments + 1, crease_pattern.num_folds))

		# The derivative of the residual vector w.r.t. the fold angles 
		d_residual_d_fold_angles = np.zeros((3 * crease_pattern.num_fold_intersections + 2 * crease_pattern.num_folds, crease_pattern.num_folds))

		# Fold angle change per increment: these are essentially the "guess increments" from the text
		#
		# We assume that each fold moves linearly from its initial value to its final (expected) value,
		# which, of course, is not the case in reality
		# 
		# However, these guess increments will be modified through the procedure outlined in the solver loop
		fold_angle_change_per_increment = np.tile(crease_pattern.fold_angle_target / self.num_increments, (self.num_increments, 1))
		
		# Keep track of the number of correction iterations that the solver makes per increment
		iterations_per_increment = np.zeros((self.num_increments + 1,), dtype=np.uint8)

		# Start the solver
		for increment in range(self.num_increments + 1):
			
			logger.debug(f'Increment: {increment}')

			if increment == 0:
				# The residual vector derivatives don't yet exist, so start with the initial values: this
				# only happens on the first increment of the solver
				history_fold_angles[increment] = crease_pattern.fold_angle_initial_value

			else:

				if self.use_projection and crease_pattern.has_interior_fold_intersections:
					# Calculate the projection of the guess increments for the fold angles
					# onto the null space of the residual vector derivatives from the previous
					# configuration - this helps reduce the number of subsequent iterations 
					# required by utilizing information from the previous configuration 
					#
					# Resource: `https://ocw.mit.edu/courses/mathematics/18-06sc-linear-algebra-fall-2011/least-squares-determinants-and-eigenvalues/projections-onto-subspaces/MIT18_06SCF11_Ses2.2sum.pdf`
					#
					# Equation `2.78`
					projected_fold_angle_increment = np.dot(np.eye(crease_pattern.num_folds) - 
													 np.matmul(np.linalg.pinv(d_residual_d_fold_angles), d_residual_d_fold_angles), fold_angle_change_per_increment[increment - 1])

					history_fold_angles[increment] = history_fold_angles[increment - 1] + projected_fold_angle_increment 
				
				else:
					history_fold_angles[increment] = history_fold_angles[increment - 1] + fold_angle_change_per_increment[increment - 1]

			if crease_pattern.has_interior_fold_intersections:
				
				for iteration in range(self.max_iterations):
					logger.debug(f'\tIteration: {iteration}')

					# First, calculate the residual vector
					residual, d_residual_d_fold_angles = self.calculate_residual(crease_pattern, history_fold_angles, increment)
					
					# If the norm of the residual vector is sufficiently small, exit 
					norm_residual = np.linalg.norm(residual) / residual.shape[0] 
					if norm_residual < self.tolerance_residual:
						logger.debug('\tThe L2 norm of the residual vector is less than the tolerance: fold angle corrections are not necessary - continuing...')
						iterations_per_increment[increment] = iteration + 1
						break

					else:
						# Calculate the fold angle corrections from the first-order expansion of the residual vector:
						# this is the generalized Newton's method, where we seek to iteratively refine the current 
						# fold angles so as to minimize the norm of the residual vector
						fold_angle_corrections = np.dot(-np.linalg.pinv(d_residual_d_fold_angles), residual)
						assert np.shape(fold_angle_corrections) == (crease_pattern.num_folds,)

						norm_fold_angle_corrections = np.linalg.norm(fold_angle_corrections) / fold_angle_corrections.shape[0]

						# Don't allow super large corrections: this part was not described in the book but was in the 
						# author's code and seems necessary, esp. for increments towards the end of the simulation
						if max(fold_angle_corrections) > self.max_fold_angle_correction:
							logger.debug('\tRescaling `fold_angle_corrections`: too large...')
							fold_angle_corrections = fold_angle_corrections * self.max_fold_angle_correction / max(fold_angle_corrections)

						# Apply the fold angle corrections to the current fold angles
						history_fold_angles[increment] = history_fold_angles[increment] + fold_angle_corrections

						if norm_fold_angle_corrections < self.tolerance_fold_angle:
							logger.debug('\tThe L2 norm of the fold angle corrections vector is less than the tolerance: exiting solver')
							iterations_per_increment[increment] = iteration + 1
							break

		logger.debug(f'Iterations per increment: {iterations_per_increment}')

		return history_fold_angles