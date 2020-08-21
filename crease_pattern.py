import json
import math
import numpy as np
from pprint import pprint
from queue import Queue 

import matrix_utils as mu

class CreasePattern():

	def __init__(self, path_to_json):
		'''Constructs a new crease pattern

		Args:
			path_to_json (str): the path to the file containing all of the required
				crease pattern data

		'''
		with open(path_to_json) as f:
			data = json.load(f)

		# Used to pad arrays that would otherwise be "jagged" because of differing
		# numbers of folds, faces, etc.
		self.filler_index = -1 

		# A 2D array containing all of the coordinates the points in the 
		# reference (starting) configuration: each of these should lie in 
		# span(e1, e2), i.e. on the xy-plane
		self.reference_points = np.array(data['reference_points'], dtype=np.float)

		# A 2D array containing the indices corresponding to each pair of 
		# reference points that form the start & end vertices of each fold 
		# vector
		#
		# For example, an entry [0, 1] would correspond to the fold vector 
		# pointing *from* point 0 *towards* point 1
		self.fold_vector_points = np.array(data['fold_vector_points'], dtype=np.int8)

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
		self.intersection_fold_indices = np.array(data['intersection_fold_indices'], dtype=np.int8)

		# To avoid the use of negative indices (mentioned above), we use a second array
		# below:
		#
		# `True`: the j-th fold vector points away (emanates) from the i-th interior fold intersection
		# `False`: the j-th fold vector points towards the i-th interior fold intersection
		self.sign_intersection_fold_indices = np.array(data['sign_intersection_fold_indices'], dtype=np.bool)

		# A 2D array containing the indices of all of the folds (and reference points) that  
		# form the boundary of each face (again, in CCW order)
		#
		# Because there may be faces with differing numbers of boundary folds (i.e. triangular
		# vs. quadrilateral faces), we introduce the "filler" index to right-fill "jagged" arrays
		# so that each sub-array has the same number of entries
		self.face_boundary = np.array(data['face_boundary'], dtype=np.int8)

		# A 2D array specifying the "sign" of each fold in the `face_boundary` array:
		#
		# `True`: the fold vector points CCW around the face
		# `False`: the fold vector points CW around the face
		self.sign_face_boundary = np.array(data['sign_face_boundary'], dtype=np.bool)

		# The index of the face that will remain fixed throughout the simulation
		self.fixed_face = data['fixed_face']

		# A 1D array that specifies an upper bound on the range of values that each fold
		# angle can take on
		self.fold_angle_upper_bound = np.array(data['fold_angle_upper_bound'], dtype=np.float)

		# A 1D array that specifies a lower bound on the range of values that each fold
		# angle can take on
		self.fold_angle_lower_bound = np.array(data['fold_angle_lower_bound'], dtype=np.float)

		# A 1D array that specifies each of the fold angles in the reference configuration
		self.fold_angle_initial_value = np.array(data['fold_angle_initial_value'], dtype=np.float)

		# A 1D array that specifies the target fold angles
		self.fold_angle_target = np.array(data['fold_angle_target'], dtype=np.float)

		self.compute_properties()

	def compute_properties(self):
		'''Based on the initial pattern data provided by the user, compute a number of other
		properties that are needed throughout the simulation

		'''
		debug = False
		def print_debug(msg):
			if debug: 
				print(msg)

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

		# A 1D array containing the angle that each fold vector makes w.r.t. e1
		self.fold_ref_angle_wrt_e1 = np.zeros(self.num_folds)

		for i, v in enumerate(self.fold_vector):
			# Extract the xy-coordinates of this vector and calculate its length (L2 norm)
			x, y = v
			norm_of_v = math.sqrt(x*x + y*y)

			# Equation `2.13`
			# 
			# Note: we can avoid doing a cross-product (as in the original formula) and
			# simply check the sign of the vector's y-coordinate
			# 
			# The angles returned by this formula are always positive in the range 0..2π
			if y >= 0.0: 
				self.fold_ref_angle_wrt_e1[i] = math.acos(x / norm_of_v)
			else:
				self.fold_ref_angle_wrt_e1[i] = 2.0 * math.pi - math.acos(x / norm_of_v)

		# A 3D array containing the direction vectors along each of the folds that emanate 
		# from each of the interior fold intersections
		#
		# The direction vectors should always be oriented *outwards* from the interior fold 
		# intersection
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

			print_debug(f'Processing interior fold intersection #{i}')

			# See notes above...
			for j in range(self.num_intersection_folds[i]):
				# The index of the j-th fold surrounding the i-th interior fold intersection
				global_fold_index = self.intersection_fold_indices[i][j]
				outward_facing_fold_vector = self.fold_vector[global_fold_index].copy()

				did_flip = False 

				# We might need to flip the vector along this fold
				if not self.sign_intersection_fold_indices[i][j]:
					outward_facing_fold_vector *= -1.0
					did_flip = True

				# Store this vector
				self.fold_direction_i[i][j] = outward_facing_fold_vector

				# Calculate the angle that it makes with e1
				x, y = outward_facing_fold_vector
				norm_of_v = math.sqrt(x*x + y*y)

				if y >= 0.0: 
					self.fold_ref_angle_wrt_e1_i[i][j] = math.acos(x / norm_of_v)
				else:
					self.fold_ref_angle_wrt_e1_i[i][j] = 2.0 * math.pi - math.acos(x / norm_of_v)

				print_debug(f'\tFold {global_fold_index}: outward facing fold vector = <{x}, {y}>, did flip = {did_flip}, norm = {norm_of_v}, angle = {self.fold_ref_angle_wrt_e1_i[i][j]}')


		for i in range(self.num_fold_intersections):
			print_debug(f'Interior fold intersection #{i} angles w.r.t. +x-axis:')
			for j in range(self.num_intersection_folds[i]):
				# Retrieve the *global* fold index of the fold that corresponds to the j-th fold
				# emanating from the i-th interior fold intersection
				global_fold_index = self.intersection_fold_indices[i][j]
				start_index, end_index = self.fold_vector_points[global_fold_index]
				theta = self.fold_ref_angle_wrt_e1_i[i][j]
				print_debug(f'\tFold #{j} (corresponding to fold index {global_fold_index}, with reference point indices [{start_index}-{end_index}]) forms angle {math.degrees(theta)} w.r.t. +x-axis')

		# A 2D array containing the face corner angles surrounding each interior fold intersection (in CCW order)
		self.face_corner_angles = np.zeros((self.num_fold_intersections, max(self.num_intersection_folds)))

		for i in range(self.num_fold_intersections):

			for j in range(self.num_intersection_folds[i]):
				# If this is the last fold that emanates from the i-th interior fold intersection,
				# add 2π - for example, if the first fold makes a 45-degree angle w.r.t. e1 and the
				# last fold makes a 315-degree angle w.r.t. e1, we want to return 90, not 45 - 315
				# (which would be -270): 360 + (45 - 315) = 90
				# 
				# Otherwise, the face corner angle is simply the difference between the next
				# fold angle and the current fold angle
				if j == self.num_intersection_folds[i] - 1:
					self.face_corner_angles[i][j] = 2.0 * math.pi + (self.fold_ref_angle_wrt_e1_i[i][0] - self.fold_ref_angle_wrt_e1_i[i][j])
				else:
					self.face_corner_angles[i][j] = self.fold_ref_angle_wrt_e1_i[i][j + 1] - self.fold_ref_angle_wrt_e1_i[i][j]

				# Prevent face corner angles from ever being negative or greater than 2π - is there
				# a better way to do this? I don't think this should ever happen...
				# if self.face_corner_angles[i][j] >= (2.0 * math.pi):
				# 	self.face_corner_angles[i][j] -= 2.0 * math.pi
				# elif self.face_corner_angles[i][j] < 0.0:
				# 	self.face_corner_angles[i][j] += 2.0 * math.pi

		for i in range(self.num_fold_intersections):
			print_debug(f'Interior fold intersection #{i} corner angles:')
			for j in range(self.num_intersection_folds[i]):
				global_fold_index = self.intersection_fold_indices[i][j]
				theta = self.face_corner_angles[i][j]
				print_debug(f'\tCorner angle #{j} (corresponding to fold index {global_fold_index}) has angle: {math.degrees(theta)}')

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

				# The index of the j-th fold that bounds the i-th face
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
					self.face_corner_points[i][count] = self.p2[k] 
					count += 1
					self.face_corner_points[i][count] = self.p1[k]
					count += 1
		    
			self.num_face_corner_points[i] = count


		for i in range(self.num_faces):
			print_debug('Face {} corner points:'.format(i))
			for j in range(self.num_face_corner_points[i]):
				print_debug('\tPoint {}: {}'.format(j, self.face_corner_points[i][j]))

		# A 2D array containing the center point of each face (for labeling purposes) - note that this is
		# not required for simulation
		self.face_centers = np.zeros((self.num_faces, 2)) 

		# TODO: is there a better way to do this like:
		# `np.average(self.face_corner_points, axis=1)`

		for i in range(self.num_faces):
			# Running sum...
			center = [0.0, 0.0]
			count = 0

			for j in self.face_boundary[i]:

				if j != self.filler_index:
					# Get the start and end points of this fold, compute its midpoint,
					# and accumulate
					s = self.p1[j]
					e = self.p2[j]
					midpoint = (s + e) * 0.5
					center += midpoint
					count += 1

			center /= float(count)

			self.face_centers[i] = center

		assert self.face_centers.shape == (self.num_faces, 2)

		# A 2D array containing the indices of the folds crossed by each path connecting the fixed face to 
		# every other face
		#
		# Indices:
		# i -> index of the "target" face
		# j -> index of the j-th fold crossed en-route to the i-th from the fixed face 
		# 
		# For example, an entry at index 0 of the form [3, 2, 1] would indicate that in order to 
		# get from the fixed face (say, the 3rd face) to the 0-th face, we would need to cross the 
		# folds at indices 3, 2, and 1 (in that order)
		# 
		# Note that these are *global* fold indices (not indices in reference to any particular interior  
		# fold intersection)
		#
		# NOTE: can this be done via Hamiltonian refinement (see page 233 of "Geometric Folding Algorithms")?
		self.fold_paths = np.full((self.num_faces, self.num_folds), self.filler_index, dtype=np.int8)
		self.sign_fold_paths = np.full((self.num_faces, self.num_folds), True, dtype=np.bool)
		
		# Construct a dictionary that maps each face index to all of its neighboring faces
		#
		# Each entry in the `set()` will be a tuple, where the first entry is the index
		# of the neighboring face and the second entry is the index of the fold that is
		# shared by the two faces
		face_neighbors = { i: set() for i in range(self.num_faces) }

		for i in range(self.num_faces): 

			for fold_index in self.face_boundary[i]:

				# Don't consider boundary folds that are filler indices: again, these exist
				# so that the second dimension of the `face_boundary` array is the same
				# across all faces
				if fold_index == self.filler_index:
					break

				# Is there another face that shares this same fold?
				for j in range(self.num_faces):

					if i != j and fold_index in self.face_boundary[j]:
						face_neighbors[i].add((j, fold_index))

		print_debug('Face neighbors:')
		print_debug(face_neighbors)

		def breadth_first_search(root):
			'''A helper function for performing a breadth first search from the fold
			at index `root` to all other reachable folds

			'''
			frontier = Queue()
			frontier.put(root)

			# Keep track of where each node "came from" (i.e. its parent in the BFS algorithm)
			came_from = {}
			came_from[root] = None

			while not frontier.empty():
				current = frontier.get()

				# Parse out face indices, ignoring the shared fold indicies (for now)
				indices_of_neighboring_faces = [neighbor[0] for neighbor in face_neighbors[current]]

				for next in indices_of_neighboring_faces:
					if next not in came_from:
						frontier.put(next)
						came_from[next] = current

			return came_from

		# Create a data structure that maps each face index to a dictionary
		#
		# The dictionary contains the results of a breadth first search from 
		# each face index, outwards
		#
		# Entries are of the form:
		#
		# i: { a:b, c:d, e:f, ... }
		#
		# Which means that during the breadth first search starting at the i-th
		# face, face "a" came from face "b", face "c" came from face "d", etc.
		#
		# We can then use this information below to work backwards from the 
		# fixed face to the i-th face, constructing a "path" between the two
		#
		# However, what we are *actually* interested in is, the indices of the 
		# folds (and their corresponding signs) crossed along such a path
		# 
		# This is where the `face_neighbors` data structure comes in: it helps
		# us determine the index of the shared fold between any two neighboring
		# faces
		came_from = {i: breadth_first_search(i) for i in range(self.num_faces)} 
		print_debug('Results of BFS:')
		print_debug(came_from)

		for i in range(self.num_faces):

			tree = came_from[i]
			
			# We always start at the fixed face and work backwards towards
			# the i-th face
			parent = self.fixed_face
			path = []
			sign_path = []

			while parent is not None:
				# The index of the face that the current parent face "came from"
				child = tree[parent]

				# This should only happen when we reach the i-th face itself
				# (whose corresponding entry in "came from" is necessarily `None`)
				if child is None:
					break

				# Find the index of the shared fold between `parent` and `child`
				shared_fold = next((neighbor[1] for neighbor in face_neighbors[parent] if neighbor[0] == child), None)
				if shared_fold is None:
					raise Error('No shared fold found between faces - this should never happen')

				# Find the index of the shared fold in the parent's list of 
				# boundary folds: use this index to look up the corresponding
				# sign of the shared fold at the parent face
				index = self.face_boundary[parent].tolist().index(shared_fold)
				sign = not self.sign_face_boundary[parent][index]

				path.append(shared_fold)
				sign_path.append(sign)

				# Recurse
				parent = child	

			# Expand each sub-list so that it matches the required length - pad as necessary
			path = path[:self.num_folds] + [self.filler_index] * (self.num_folds - len(path))
			sign_path = sign_path[:self.num_folds] + [True] * (self.num_folds - len(sign_path))

			self.fold_paths[i] = path
			self.sign_fold_paths[i] = sign_path

			print_debug(f'Face {i}: \n{path}\n{sign_path}\n')

		assert self.fold_paths.shape == (self.num_faces, self.num_folds)

	def compute_folding_map(self, fold_angles):
		'''Computes a folding map

		Args:
			fold_angles (numpy.ndarray): a 1D array containing the fold angles of the desired 
				configuration

		Returns:
			numpy.ndarray: a 3D array containing a 4x4 transformation matrix that maps a point
				in the fixed face to a target point in the i-th face, taking into account all
				of the folds that you would encounter on such a path

		'''
		folding_map = np.zeros((self.num_faces, 4, 4))

		for face_index in range(self.num_faces):

			# Create a 4x4 identity matrix
			composite = np.eye(4, 4)

			# Traverse the fold path and accumulate transformation matrices
			for fold_index, fold_sign in zip(self.fold_paths[face_index], self.sign_fold_paths[face_index]):

				# There are no more "actual" folds along this path, so terminate
				if fold_index == self.filler_index:
					break

				alpha = self.fold_ref_angle_wrt_e1[fold_index]
				phi = fold_angles[fold_index]

				# If the path crosses this fold in the "negative" direction, add π 
				if not fold_sign:
					alpha += math.pi

				# `b` is the starting reference point along this fold - note that
				# we convert `b` to a 3-element vector with z implicitly set to 0
				# before continuing
				b = self.p1[fold_index]
				b = np.append(b, 0.0)

				fold_transformation = mu.get_fold_transform(alpha, phi, b)

				# Accumulate transformations
				composite = np.matmul(composite, fold_transformation)

			folding_map[face_index] = composite

		return folding_map

if __name__ == "__main__":
	cp = CreasePattern('patterns/simple.json')

