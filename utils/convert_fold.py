import json
import math 
import numpy as np
from pprint import pprint

output_data = {
	'reference_points': [],
	'fold_vector_points': [],
	'intersection_fold_indices': [],
	'sign_intersection_fold_indices': [],
	'face_boundary': [],
	'sign_face_boundary': [],
	'fixed_face': 0,
	'fold_angle_upper_bound': [],
	'fold_angle_lower_bound': [],
	'fold_angle_initial_value': [],
	'fold_angle_target': []
}

with open('bird.fold') as f:
	input_data = json.load(f)

scale = 10.0
apply_scale_and_center = True 

# Load reference points, which are simply the same as FOLD vertices
centroid = np.zeros((2,), dtype=np.float)

for vertex_coord in input_data['vertices_coords']:
	output_data['reference_points'].append(np.array(vertex_coord, dtype=np.float))
	
	centroid += np.array(vertex_coord)

if apply_scale_and_center:
	centroid /= len(input_data['vertices_coords'])
	for local_index, vertex_coord in enumerate(output_data['reference_points']):
		output_data['reference_points'][local_index] -= centroid
		output_data['reference_points'][local_index] *= scale
		output_data['reference_points'][local_index] = output_data['reference_points'][local_index].tolist()

reassigned_edge_indices = {}

# Load and parse fold information
for local_index, vertices in enumerate(input_data['edges_vertices']):

	# Grab this edges assignment (M, V, B, or U)
	assignment = input_data['edges_assignment'][local_index]	
	if assignment not in ['M', 'V', 'B', 'U']:
		raise Error('Invalid crease assignment')

	# Ignore boundary folds and unknown folds, which are not needed for simulation
	if assignment != 'B' and assignment != 'U':

		reassigned_edge_indices[local_index] = len(output_data['fold_vector_points'])

		output_data['fold_vector_points'].append(vertices)
		output_data['fold_angle_upper_bound'].append(math.pi)
		output_data['fold_angle_lower_bound'].append(-math.pi)
		output_data['fold_angle_initial_value'].append(0.0)

		# M folds are assigned (+) angles
		# V folds are assigned (-) angles
		if assignment == 'M':
			output_data['fold_angle_target'].append(math.pi)
		elif assignment == 'V':
			output_data['fold_angle_target'].append(-math.pi)
		else:
			output_data['fold_angle_target'].append(0.0)
	#else:
	#	print(f'Encountered B or U fold at index {local_index} - skipping')

pprint(reassigned_edge_indices)

num_folds = len(output_data['fold_vector_points'])
print(f'Found {num_folds} folds with valid assignments')

def is_interior(edges):
	'''A helper function to check whether a vertex is an interior fold intersection
	'''
	for edge in edges:
		if input_data['edges_assignment'][edge] == 'B' or input_data['edges_assignment'][edge] == 'U':
			return False

	return True

def emanates_from(vertex_index, edge_index):
	'''A helper function that says whether or not the edge at the specified edge index
	emanates from (i.e. starts at) the vertex at the specified vertex index

	'''
	return input_data['edges_vertices'][edge_index][0] == vertex_index


for vertex_index, edges in enumerate(input_data['vertices_edges']):

	if is_interior(edges):
		output_data['intersection_fold_indices'].append(edges)
		#print(f'Vertex {vertex_index} is interior')

		signs = [emanates_from(vertex_index, edge_index) for edge_index in edges]
		output_data['sign_intersection_fold_indices'].append(signs)

# Back-fill with filler values
if len(output_data['intersection_fold_indices']) > 0:
	max_list = max(output_data['intersection_fold_indices'], key = lambda i: len(i)) 
	max_intersection_folds = len(max_list) 

	for fold_index in range(len(output_data['intersection_fold_indices'])):
		output_data['intersection_fold_indices'][fold_index] = output_data['intersection_fold_indices'][fold_index][:max_intersection_folds] + [-1] * (max_intersection_folds - len(output_data['intersection_fold_indices'][fold_index]))
		output_data['sign_intersection_fold_indices'][fold_index] = output_data['sign_intersection_fold_indices'][fold_index][:max_intersection_folds] + [True] * (max_intersection_folds - len(output_data['sign_intersection_fold_indices'][fold_index]))
	#pprint(intersection_fold_indices)
	#pprint(sign_intersection_fold_indices)

num_fold_intersections = len(output_data['intersection_fold_indices'])
print(f'Found {num_fold_intersections} fold intersections')




# The FOLD spec states that the edges will always be given in a CCW order, but this
# doesn't necessarily mean that each of the edges in *oriented* in a CCW fashion
for face_index, (vertices, edges) in enumerate(zip(input_data['faces_vertices'], input_data['faces_edges'])):

	output_data['face_boundary'].append(edges)

	signs = [not emanates_from(vertex_index, edge_index) for vertex_index, edge_index in zip(vertices, edges)]
	output_data['sign_face_boundary'].append(signs)

# Get rid of B and U folds in face boundary arrays
for face_index, edges in enumerate(output_data['face_boundary']):

	for local_index, edge_index in enumerate(edges):

		if input_data['edges_assignment'][edge_index] == 'B' or input_data['edges_assignment'][edge_index] == 'U':

			output_data['face_boundary'][face_index].pop(local_index)
			output_data['sign_face_boundary'][face_index].pop(local_index)

max_list = max(output_data['face_boundary'], key = lambda i: len(i)) 
max_boundary_edges = len(max_list) 
print(f'Max boundary edges: {max_boundary_edges}')


# Back-fill with filler values
for face_index in range(len(output_data['face_boundary'])):

	output_data['face_boundary'][face_index] = output_data['face_boundary'][face_index][:max_boundary_edges] + [-1] * (max_boundary_edges - len(output_data['face_boundary'][face_index]))
	output_data['sign_face_boundary'][face_index] = output_data['sign_face_boundary'][face_index][:max_boundary_edges] + [True] * (max_boundary_edges - len(output_data['sign_face_boundary'][face_index]))

pprint(output_data['face_boundary'])
pprint(output_data['sign_face_boundary'])

with open('../kinematic_origami/patterns/converted.json', 'w') as outfile:
    json.dump(output_data, outfile, indent=4)


