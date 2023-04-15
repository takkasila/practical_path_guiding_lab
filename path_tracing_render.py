import drjit as dr
import mitsuba as mi
if __name__ == '__main__':
	mi.set_variant('cuda_ad_rgb')

from src.path_tracing_integrator_py import PathTracingIntegrator

import matplotlib.pyplot as plt

from src.file_name_manager import FileNameManager
FileNameManager.setSceneName('')
import pathlib

from src.common import PerformanceData, printTitle, printBoldUnderLine

from random import randint
import time

import progressbar

"""
	Use as a benchmark renderer
"""

if __name__ == '__main__':

	# Load scene
	# scene = mi.load_file('scenes/cornell-box/scene.xml')
	# scene = mi.load_file('scenes/torus/scene.xml')
	# scene = mi.load_file('scenes/veach-bidir/scene.xml')
	# scene = mi.load_file('scenes/veach-ajar/scene.xml')
	scene = mi.load_file('scenes/kitchen/scene.xml')

	sceneName = 'kitchen'	# for file saving

	# Load ground truth
	groundTruthFileName = 'scenes/kitchen/TungstenRender.exr'
	groundTruthImg = mi.TensorXf( mi.Bitmap( groundTruthFileName ) )
	# film_size = scene.sensors()[0].film().size()
	# imgShape = film_size[0] * film_size[1]
	# groundTruthImg = dr.zeros( mi.Float, shape= imgShape * 3 )
	groundTruth = dr.unravel( mi.Color3f, groundTruthImg.array )

	pathTracingIntegrator: PathTracingIntegrator = scene.integrator()

	# Define desired SPP and chunk size
	target_spp = 40

	isTimeBudget = True		#	if this is true then it overrides target_spp
	timeBudget = 1000 	# in secs

	chunkSize = 4

	# Record variance data
	variance_groundTruth_record = PerformanceData()
	cumm_time = 0
	used_spp = 0

	image_acc = None

	#  Set to 0 if want reproductibility across simulation
	initiali_seed = randint(0, 1000000)
	printBoldUnderLine('Initial seed:', initiali_seed)
	
	# normal-integrator folder path for saving image
	folderPath = f'{FileNameManager.DEBUG_FOLDER_PATH}/normal-integrator/{sceneName}/'
	pathlib.Path( folderPath ).mkdir( parents= True, exist_ok= True )

	# Render progress bar
	render_progressbar = progressbar.ProgressBar(maxval= 100, widgets=[progressbar.Bar('=', 'Render progress [', ']'), ' ', progressbar.Percentage()])
	render_progressbar.start()

	if not isTimeBudget:

		# 	Calculate SPP per batch
		numChunks = target_spp // chunkSize
		spp_arr = [chunkSize] * numChunks

		remainingSPP = target_spp % chunkSize
		if remainingSPP != 0:
			spp_arr.append( remainingSPP )

		for i, spp in enumerate( spp_arr ):

			start_render_time = time.perf_counter()

			# Render
			image = mi.render( scene, seed= initiali_seed + i, spp= spp )
			
			if image_acc is None:
				image_acc = image
			else:
				image_acc += image
			
			dr.eval( image_acc )
		
			used_spp += spp

			mse = pathTracingIntegrator.computeMSE( used_spp, groundTruth )
			variance = pathTracingIntegrator.computeVariance( used_spp, groundTruth )
			
			end_render_time = time.perf_counter()
			elapse_render_time = end_render_time - start_render_time
			curr_time = elapse_render_time + cumm_time

			cumm_time += elapse_render_time

			# Record variance
			variance_groundTruth_record.append( time= curr_time, spp= used_spp, iteration= 0, variance= variance, mse= mse )

			render_progressbar.update( round( 100 * used_spp / target_spp ) )

		# Save image
		image = image_acc / (i+1)
		mi.util.write_bitmap( f'{folderPath}{sceneName}-{used_spp}.png', image )
		mi.util.write_bitmap( f'{folderPath}{sceneName}-{used_spp}.exr', image )
			
			
	else:

		pass_count = 0

		start_render_time = time.perf_counter()

		while cumm_time < timeBudget:

			# Render
			image = mi.render( scene, seed= initiali_seed + pass_count, spp= chunkSize )
			
			pass_count += 1

			if image_acc is None:
				image_acc = image
			else:
				image_acc += image
			
			dr.eval( image_acc )

			used_spp += chunkSize

			mse = pathTracingIntegrator.computeMSE( used_spp, groundTruth )
			variance = pathTracingIntegrator.computeVariance( used_spp, groundTruth )

			now_time = time.perf_counter()

			cumm_time = now_time - start_render_time

			# Record variance
			variance_groundTruth_record.append( time= cumm_time, spp= used_spp, iteration= 0, variance= variance, mse= mse )

			render_progressbar.update( round( 100 * min( 1, cumm_time / timeBudget) ) )

		# Save image
		image = image_acc / pass_count
		mi.util.write_bitmap( f'{folderPath}{sceneName}-{used_spp}.png', image )
		mi.util.write_bitmap( f'{folderPath}{sceneName}-{used_spp}.exr', image )


	render_progressbar.finish()

	# Save plot data
	variance_groundTruth_record.saveToFile( f'{folderPath}{sceneName}_variance_groundTruth_path_tracing_py.csv' )

	# Render image
	plt.axis("off")
	plt.imshow( image ** (1.0 / 2.2) ); # approximate sRGB tonemapping
	plt.show()

