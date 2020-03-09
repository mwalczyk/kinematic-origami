import json
import math
import numpy as np

class CreasePattern():

	def __init__(self, path_to_json):
		with open(path_to_json) as f:
			data = json.load(f)

		self.filler_index = -1 

		# A 2D array containing all of the coordinates the points in the 
		# reference (starting) configuration: each of these should lie in 
		# span(e1, e2), i.e. on the xy-plane
		self.reference_points = np.array(data['reference_points'])

		# A 2D array containing the indices corresponding to each pair of 
		# reference points that form the start & end vertices of each fold 
		# vector
		#
		# For example, an entry [0, 1] would correspond to the fold vector 
		# pointing *from* point 0 *towards* point 1
		self.fold_vector_points = np.array(data['fold_vector_points'])


		# A 2D array containing the fold indices associated with each interior fold
		# intersection: these are expected to be in counter-clockwise (CCW) order
		#
		# In the Matlab implementation, these indices are signed (positive or
		# negative), where the sign corresponded to:
		# 
		# (+): the corresponding fold vector points away from the intersection 
		# (-): the corresponding fold vector points towards the intersection
		#
		# Since Matlab uses 1-based indexing, this works - however, it leads to 
		# issues in Python, which uses 0-based indexing (since 0 can't be negative)
		#
		# Indices:
		#
		# i -> index of the interior fold intersection
		# j -> index of the j-th fold surrounding the i-th interior fold intersection
		self.intersection_fold_indices = np.array(data['intersection_fold_indices'])

		# To avoid the use of negative indices (mentioned above), we use a second array
		# below:
		#
		# `True`: the j-th fold vector points away from the i-th interior fold intersection
		# `False`: the j-th fold vector points towards the i-th interior fold intersection
		self.sign_intersection_fold_indices = np.array(data['sign_intersection_fold_indices'])

		# A 2D array containing the indices of all of the folds (and reference points) that  
		# form the boundary of each face (again, in CCW order)
		#
		# Because there may be faces with differing numbers of boundary folds (i.e. triangular
		# vs. quadrilateral faces), we introduce the "filler" index to right-fill "jagged" arrays
		# so that each sub-array has the same number of entries
		self.face_boundary = np.array(data['face_boundary'])

		# A 2D array specifying the "sign" of each fold in the `face_boundary` array:
		#
		# `True`: the fold vector points CCW around the face
		# `False`: the fold vector points CW around the face
		self.sign_face_boundary = np.array(data['sign_face_boundary'])

		# The index of the face that will remain fixed throughout the simulation
		self.fixed_face = np.array(data['fixed_face'])

		# A 1D array that specifies an upper bound on the range of values that each fold
		# angle can take on
		self.fold_angle_upper_bound = np.array(data['fold_angle_upper_bound'])

		# A 1D array that specifies a lower bound on the range of values that each fold
		# angle can take on
		self.fold_angle_lower_bound = np.array(data['fold_angle_lower_bound'])

		# A 1D array that specifies each of the fold angles in the reference configuration
		self.fold_angle_initial_value = np.array(data['fold_angle_initial_value'])

		self.compute_properties()

	def compute_properties(self):
		'''Based on the initial pattern data provided by the user, compute a number of other
		properties that are needed throughout the simulation

		'''
		# The total number of reference points (i.e. vertices)
		self.num_reference_ponts = self.reference_points.shape[0]

		# The total number of folds (i.e. creases) across the ENTIRE pattern
		self.num_folds = self.fold_vector_points.shape[0]

		# The total number of interior fold intersections (i.e. interior vertices)
		self.num_fold_intersections = self.intersection_fold_indices.shape[0]

		# The total number of faces
		self.num_faces = self.face_boundary.shape[0]

		# Quick sanity check...
		assert self.fold_angle_upper_bound.shape == (self.num_folds,) 
		assert self.fold_angle_lower_bound.shape == (self.num_folds,) 
		assert self.fold_angle_initial_value.shape == (self.num_folds,) 




		# A 2D array containing the number of folds emanating from each interior fold intersection
		self.num_intersection_folds = np.zeros(self.num_fold_intersections, dtype=np.int8)

		for i in range(self.num_fold_intersections):
			count = np.count_nonzero(self.intersection_fold_indices[i] != self.filler_index)
			self.num_intersection_folds[i] = count

		# A 2D array containing the coordinates of the starting point of each fold vector
		self.p1 = np.array([self.reference_points[i] for i, _ in self.fold_vector_points])

		# A 2D array containing the coordinates of the ending point of each fold vector
		self.p2 = np.array([self.reference_points[j] for _, j in self.fold_vector_points])

		# A 2D array containing the direction vector along each fold 
		self.fold_vector = self.p2 - self.p1

		# Additional sanity checks...
		assert self.p1.shape == (self.num_folds, 2)
		assert self.p2.shape == (self.num_folds, 2)
		assert self.fold_vector.shape == (self.num_folds, 2)




		# A 1D array containing the angle that each fold vector makes w.r.t. the positive x-axis
		self.fold_ref_angle_wrt_e1 = np.zeros(self.num_folds)

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

		for i, v in enumerate(self.fold_vector):
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
				self.fold_ref_angle_wrt_e1[i] = math.acos(x / norm_of_v)
			else:
				self.fold_ref_angle_wrt_e1[i] = 2.0 * math.pi - math.acos(x / norm_of_v)

		# A 3D array containing the direction vectors along each of the folds that emanate 
		# from each of the interior fold intersections: note that the direction vectors 
		# should always be oriented *outwards* from the interior fold intersection
		#
		# Indices:
		#
		# i -> index of the interior fold intersection 
		# j -> index of the j-th fold surrounding the i-th interior fold intersection
		# k -> index of the coordinate (x or y): thus, k should always be 0 or 1
		# 
		# When indexing into the second dimension of this array, j, one should always use a 
		# loop like:
		# 
		# 	for j in range(num_intersection_folds[i])
		#		...
		#
		# This is because different interior fold intersections might have different numbers
		# of surrounding folds, so we want to avoid referencing any "extra" / "filler" folds
		# that implicitly "pad" the i-th interior fold intersection
		self.fold_direction_i = np.zeros((self.num_fold_intersections, np.max(self.num_intersection_folds), 2))

		# The angle that each interior fold makes with e1
		self.fold_ref_angle_wrt_e1_i = np.zeros((self.num_fold_intersections, np.max(self.num_intersection_folds)))

		for i in range(self.num_fold_intersections):

			# See notes above...
			for j in range(self.num_intersection_folds[i]):

				k = self.intersection_fold_indices[i][j]
				v = self.fold_vector[k]

				# We might need to flip the vector along this fold
				if not self.sign_intersection_fold_indices[i][j]:
					v *= -1.0

				# Store this vector
				self.fold_direction_i[i][j] = v

				# Calculate the angle that it makes with e1
				x, y = v
				norm_of_v = math.sqrt(x*x + y*y)

				if y >= 0.0: 
					self.fold_ref_angle_wrt_e1_i[i][j] = math.acos(x / norm_of_v)
				else:
					self.fold_ref_angle_wrt_e1_i[i][j] = 2.0 * math.pi - math.acos(x / norm_of_v)

		for i in range(self.num_fold_intersections):

			print(f'Interior fold intersection #{i} angles w.r.t. +x-axis:')

			for j in range(self.num_intersection_folds[i]):
				# Retrieve the *global* fold index of the fold that corresponds to the `j`th fold
				# emanating from the `i`th interior fold intersection
				global_fold_index = self.intersection_fold_indices[i][j]
				start_index, end_index = self.fold_vector_points[global_fold_index]
				theta = self.fold_ref_angle_wrt_e1_i[i][j]

				print(f'\tFold #{j} (corresponding to fold index {global_fold_index}, with reference point indices [{start_index}-{end_index}]) forms angle {math.degrees(theta)} w.r.t. +x-axis')

		# A 2D array containing the face corner angles surrounding each interior fold intersection (in CCW order)
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

		self.face_corner_angles = np.array([
			# Corner angles surrounding interior fold intersection #0
			[math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4]

			# Corner angles surrounding interior fold intersection #1
			# ...

			# Corner angles surrounding interior fold intersection #2
			# ...
		])

		for i in range(self.num_fold_intersections):

			print(f'Interior fold intersection #{i} corner angles:')

			for j in range(self.num_intersection_folds[i]):
				# Retrieve the *global* fold index of the fold that corresponds to the `j`th fold
				# emanating from the `i`th interior fold intersection
				global_fold_index = self.intersection_fold_indices[i][j]
				theta = self.face_corner_angles[i][j]

				print(f'\tCorner angle #{j} (corresponding to fold index {global_fold_index}) has angle: {math.degrees(theta)}')

		# A 3D array containing all of the corner points of each polygonal face, which may or 
		# may not be triangular
		#
		# Indices:
		#
		# i -> index of the face
		# j -> index of the j-th corner point bounding the i-th face
		# k -> index of the coordinate (x or y): thus, k should always be 0 or 1
		#
		# The maximum number of boundary folds for a given face is dictated by the 
		# size of the 2nd dimension of the `face_boundary` array, which will be 
		# right-filled to accomodate any faces that have less than `max_boundary_folds_per_face`
		# boundary folds
		max_boundary_folds_per_face = self.face_boundary.shape[1]
		self.face_corner_points = np.zeros((self.num_faces, 2 * max_boundary_folds_per_face, 2))

		# A 1D array containing the number of corner points per face
		self.num_face_corner_points = np.zeros(self.num_faces, dtype=np.int8)

		# Figure out which reference points form this polygonal face, in CCW order
		for i in range(self.num_faces):

			count = 0

			for j in range(len(self.face_boundary[i])):
				
				# The index of the `j`th fold that bounds the `i`th face
				k = self.face_boundary[i][j]

				# Have we reached the end of this face?
				if k == self.filler_index:
					break

				if self.sign_face_boundary[i][j]:
					# Connect to fold in "positive" direction - the fold already goes CCW
					self.face_corner_points[i][count] = self.p1[k]
					count += 1
					self.face_corner_points[i][count] = self.p2[k]
					count += 1
				else:
					# Connect to fold in "negative" direction
					self.face_corner_points[i][count] = self.p2[k] # This used to be `p2[abs(k)]`, but we aren't using negative indices anymore
					count += 1
					self.face_corner_points[i][count] = self.p1[k]
					count += 1
		    
			self.num_face_corner_points[i] = count

		for i in range(self.num_faces):

			print('Face {} corner points:'.format(i))

			for j in range(self.num_face_corner_points[i]):

				print('\tPoint {}: {}'.format(j, self.face_corner_points[i][j]))

		# A 2D array containing the center point of each face (for labeling purposes) - note that this is
		# not required for simulation
		self.face_centers = np.zeros((self.num_faces, 2))

		for i in range(self.num_faces):

			center = [0.0, 0.0]

			# TODO: use `np.sum(...)` or something
			for j in range(self.num_face_corner_points[i]):
				center += self.face_corner_points[i][j]

			center /= self.num_face_corner_points[i]

			self.face_centers[i] = center

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
		self.fold_paths = np.array([
			[3, self.filler_index, self.filler_index, self.filler_index, self.filler_index, self.filler_index, self.filler_index, self.filler_index],
			[3, 0, self.filler_index, self.filler_index, self.filler_index, self.filler_index, self.filler_index, self.filler_index],
			[3, 0, 1, self.filler_index, self.filler_index, self.filler_index, self.filler_index, self.filler_index],
			[3, 0, 1, 2, self.filler_index, self.filler_index, self.filler_index, self.filler_index],
			[3, 0, 1, 2, 4, self.filler_index, self.filler_index, self.filler_index],
			[3, 0, 1, 2, 4, 7, self.filler_index, self.filler_index],
			[3, 0, 1, 2, 4, 7, 6, self.filler_index],
			[3, 0, 1, 2, 4, 7, 6, 5]
		])
		assert self.fold_paths.shape == (self.num_faces, self.num_folds)


if __name__== "__main__":

	cp = CreasePattern('simple_pattern.json')

