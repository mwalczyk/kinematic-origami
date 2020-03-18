import json
import numpy as np 
from pprint import pprint

def generate_waterbomb_tessellation(cell_size=4.0, cells_x=5, cells_y=2, path_to_json='waterbomb_tessellation.json'):
	'''
	'''

	reference_points = []

	for y in range(cells_y + 1):
		for x in range(cells_x + 1):
		
			reference_points.append([x * cell_size, y * cell_size])
	
	fold_vector_points = []
	fold_angle_upper_bound = []
	fold_angle_lower_bound = []
	fold_angle_initial_value = []
	fold_angle_target = []
	fixed_face = 0

	def add_fold(start, end, valley=True):
		fold_vector_points.append([start, end])
		fold_angle_upper_bound.append(math.pi)
		fold_angle_lower_bound.append(-math.pi)
		fold_angle_initial_value.append(0.0)

		if valley:
			fold_angle_target.append(-math.pi)
		else:
			fold_angle_target.append(math.pi)

	for y in range(cells_y):
		
		for x in range(cells_x):
			
			# 1 2 3 4 0
			# 3 4 0 1 2
			# 0 1 2 3 4

			# Always specify edges in CCW order, which makes constructing faces
			# easier in subsequent steps
			cell_type = (x + (3 * y)) % 5

			# V folds (blue): -pi
			# M folds (red): +pi

			print(cell_type)
			if cell_type == 0:
				pass
			elif cell_type == 1 or cell_type == 4:
				s = (y + 0) * (cells_x + 1) + (x + 1)
				e = (y + 1) * (cells_x + 1) + (x + 0)
				fold_vector_points.append([s, e])
				fold_angle_upper_bound.append(math.pi)
				fold_angle_lower_bound.append(-math.pi)
				fold_angle_initial_value.append(0.0)
				fold_angle_target.append(-math.pi)


			elif cell_type == 2 or cell_type == 3: 
				
				s = (y + 0) * (cells_x + 1) + (x + 0)
				e = (y + 1) * (cells_x + 1) + (x + 1)
				
				fold_vector_points.append([s, e])
				fold_angle_upper_bound.append(math.pi)
				fold_angle_lower_bound.append(-math.pi)
				fold_angle_initial_value.append(0.0)
				fold_angle_target.append(-math.pi)

	pprint(fold_vector_points)
			


	pprint(reference_points)


if __name__ == "__main__":
	generate_waterbomb_tessellation()
