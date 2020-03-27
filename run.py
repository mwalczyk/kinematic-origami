import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from pprint import pprint 

from crease_pattern import CreasePattern
import matrix_utils as mu
import plot_utils as pu
from solver import Solver 

if __name__ == "__main__":
	# Initialize the crease pattern and solver
	increments = 40
	crease_pattern = CreasePattern('patterns/medium.json')
	solver = Solver(num_increments=increments)

	# Hide toolbars (with "save" button, etc.) and set the GUI theme
	matplotlib.rcParams['toolbar'] = 'None'
	matplotlib.style.use('ggplot')

	# Create a 3D plot for displaying various folded configurations of the model
	solve = True
	print_fold_angles = False
	size = 50
	fig_3d = plt.figure()
	fig_3d.canvas.set_window_title('Folded Configuration')
	fig_3d.tight_layout(pad=2)
	fig_3d.tight_layout(rect=[0, 0, 0.5, 0.5]) 
	axes_3d = Axes3D(fig_3d)
	axes_3d.set_xlim3d(0, size)
	axes_3d.set_ylim3d(0, size)
	axes_3d.set_zlim3d(0, size)
	axes_3d.tick_params(axis='both', which='major', labelsize=6)
	axes_3d.tick_params(axis='both', which='minor', labelsize=6)

	# Create a 2D plot for displaying the crease pattern (as a planar graph)
	fig, axes_2d = plt.subplots()
	fig.canvas.set_window_title('Crease Pattern')
	plt.subplots_adjust(bottom=0.25)
	axes_2d.tick_params(axis='both', which='major', labelsize=6)
	axes_2d.tick_params(axis='both', which='minor', labelsize=6)

	if solve:
		# Run the solver
		history_fold_angles = solver.run(crease_pattern)

		if print_fold_angles:
			for fold_angles in history_fold_angles:
				pprint(fold_angles)

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

	pu.plot_crease_pattern(axes_2d, crease_pattern)

	# Show the plot
	plt.show(block=False)
	input('Hit ENTER to close all windows...')
	plt.close()
