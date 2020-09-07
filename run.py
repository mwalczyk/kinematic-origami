import json
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from pprint import pprint 
import time

from crease_pattern import CreasePattern
import matrix_utils as mu
import plot_utils as pu
from solver import Solver 

if __name__ == "__main__":
	# Initialize the crease pattern and solver
	increments = 50
	crease_pattern = CreasePattern('patterns/unfolding.json', 30.0)
	solver = Solver(num_increments=increments)

	# Hide toolbars (with "save" button, etc.) and set the GUI theme
	matplotlib.rcParams['toolbar'] = 'None'
	matplotlib.style.use('ggplot')

	# Create a 3D plot for displaying various folded configurations of the model
	solve = True
	print_fold_angles = False
	export_keyframes = False
	show_grid_in_xyz = False
	show_crease_pattern_plot = False
	size = 60

	fig_3d = plt.figure(figsize=(8,8))
	fig_3d.canvas.set_window_title('Folded Configuration')
	axes_3d = Axes3D(fig_3d)
	axes_3d.set_xlim3d(0, size)
	axes_3d.set_ylim3d(0, size)
	axes_3d.set_zlim3d(0, size * 0.8)
	axes_3d.tick_params(axis='both', which='major', labelsize=6)
	axes_3d.tick_params(axis='both', which='minor', labelsize=6)

	if not show_grid_in_xyz:
		axes_3d.set_axis_off()

	# Create a 2D plot for displaying the crease pattern (as a planar graph)
	fig, axes_2d = plt.subplots()
	fig.canvas.set_window_title('Crease Pattern')
	fig.set_size_inches(8, 8)
	plt.subplots_adjust(bottom=0.25)
	axes_2d.tick_params(axis='both', which='major', labelsize=6)
	axes_2d.tick_params(axis='both', which='minor', labelsize=6)

	if solve:
		# Run the solver
		history_fold_angles = solver.run(crease_pattern)
		if print_fold_angles:
			for fold_angles in history_fold_angles:
				pprint(fold_angles)
		
		if export_keyframes:
			# Export keyframes
			folded_positions = [crease_pattern.compute_folded_positions(fold_angles).tolist() for fold_angles in history_fold_angles]
			keyframes = {
				'num_face_corner_points': crease_pattern.num_face_corner_points.tolist(),
				'folded_positions': folded_positions
			}
			with open("keyframes.json", "w") as write_file:
				json.dump(keyframes, write_file, indent=4)

		# Create a separate 2D axes for the GUI elements
		axes_slider = plt.axes([0.25, 0.125, 0.5, 0.03])
		slider_init = increments
		slider_increment = Slider(axes_slider, 'Increment', 0, increments, valinit=slider_init)
		slider_increment.label.set_size(6)
		slider_increment.valtext.set_size(6)

		def update(val):
			'''A small callback function to redraw the UI whenever the slider changes

			'''
			pu.plot_custom_configuration(axes_3d, crease_pattern, history_fold_angles[int(slider_increment.val)])
			fig_3d.canvas.draw_idle()

		slider_increment.on_changed(update)
		update(slider_init)
	else:
		pu.plot_reference_configuration(axes_3d, crease_pattern)

	# Maybe show the 2D plot
	if show_crease_pattern_plot:
		pu.plot_crease_pattern(axes_2d, crease_pattern)

	# Show the 3D plot
	plt.show(block=False)
	input('Hit ENTER to close all windows...')
	plt.close()
