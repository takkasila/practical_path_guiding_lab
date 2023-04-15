from __future__ import annotations

import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from src.path_guiding_integrator import PathGuidingIntegrator

from src.file_name_manager import FileNameManager

import matplotlib.pyplot as plt

from src.common import PerformanceData, printTitle, printBoldUnderLine

from random import randint
import time

import math

import progressbar

import pandas as pd


def doFullSimulation( start_iteration: int, end_iteration: int, max_tree_iteration: int, iter_spp: int, batch_spp: int, sim_iter: int ) -> None:

	# Render image buffer
	image = None

	# Record performance data
	isRecordPerformanceInIteration = True

	variance_inIter_record = PerformanceData()
	variance_groundTruth_inIter_record = PerformanceData()
	mse_groundTruth_inter_record = PerformanceData()

	variance_endIter_record = PerformanceData()
	variance_groundTruth_endIter_record = PerformanceData()
	mse_groundTruth_endIter_record = PerformanceData()

	variance = 0
	variance_groundTruth = 0
	mse_groundTruth = 0

	# Theoretical Cumulative SPP 
	theo_cumm_iter_spp = 0		# 0 4 12 28 60 ...

	# True Cumulative spp across iteration
	cumm_spp = 0

	# Number of SPP used to generate the current image.
	iter_spp_count = 0

	iteration_count = 0

	# 	Performance timer
	# 	Only record time relating to main computation, no file saving
	elapse_iter_time = 0 

	# Theoretical prev iter time
	theo_iter_time = 0
	theo_cumm_iter_time = 0

	# 	Set seed
	initiali_seed = randint(0, 1000000)
	printBoldUnderLine('Initial seed:', initiali_seed)

	# Iteratively rendering every iteration with same fixed amount of SPP
	for iteration_count in range( start_iteration, end_iteration + 1 ):
		
		# Reset data
		image = None
		iter_spp_count = 0
		pathGuidingIntegrator.resetVarianceCounter()
		pathGuidingIntegrator.setIteration( iteration_count, isFinalIter= True )
		theo_iter_time_trigger_flag = True


		# Theoretical prev spp
		theo_prev_iter_spp = 0
		if iteration_count > 0:
			theo_prev_iter_spp = 2 ** ( iteration_count + 1 )	# 4 8 16 ...
		
		theo_cumm_iter_spp += theo_prev_iter_spp
			
		# Load SDTree of the current iteration
		if 0 < iteration_count and iteration_count <= max_tree_iteration:
			quadTreeFileName = FileNameManager.generateTreeDataFileName( iteration_count - 1 )
			pathGuidingIntegrator.loadSDTreeFromFile( quadTreeFileName )
		
		# Start timer
		start_iter_time = time.perf_counter()

		printTitle(f'Iteration {iteration_count}')
		
		# 
		# 	Render pass
		# 
		
		spp_per_pass = batch_spp
		
		iter_pass = math.ceil(iter_spp / spp_per_pass)

		iter_spp_count = 0
		
		# Render progress bar
		render_progressbar = progressbar.ProgressBar(maxval= 100, widgets=[progressbar.Bar('=', 'Render progress [', ']'), ' ', progressbar.Percentage()])
		render_progressbar.start()

		# Render multiple passes
		for pass_i in range( iter_pass ):

			# Compute SPP of the current pass
			avaible_spp = iter_spp - iter_spp_count
			curr_pass_spp = spp_per_pass

			if curr_pass_spp > avaible_spp:
				curr_pass_spp = avaible_spp
			
			# Render a pass
			image_one_pass = mi.render( scene= scene, spp= curr_pass_spp, seed= initiali_seed + cumm_spp )
			
			# Accumulate render result
			weighted_image_one_pass = image_one_pass * float( curr_pass_spp / iter_spp )
			if image is None:
				image = weighted_image_one_pass
			else:
				image += weighted_image_one_pass

			# Evaluate the computation so that it won't keep putting 
			# into cache and then exceed the memory limit
			dr.eval( image )

			iter_spp_count += curr_pass_spp
			cumm_spp  += curr_pass_spp

			# 	Record theoretical iteration time i.e. time took to render 4 samples, 8 samples, 16 samples, ...
			if (iter_spp_count >= (2 ** ( iteration_count + 2 ))) and theo_iter_time_trigger_flag:
				theo_iter_time_trigger_flag = False
				theo_iter_time =  time.perf_counter() - start_iter_time


			if isRecordPerformanceInIteration:
				# 
				# 	Compute Variance and MSE within iteration
				# 
				
				# 		Variance wrt. self
				variance = pathGuidingIntegrator.computeVariance( iter_spp_count )

				# 		Variance wrt. ground truth
				variance_groundTruth = pathGuidingIntegrator.computeVariance( iter_spp_count, groundTruth )

				# 		Mean Square Error wrt. ground truth
				mse_groundTruth = pathGuidingIntegrator.computeMSE( iter_spp_count, groundTruth )

				# Record Variance and MSE
				curr_iter_time = time.perf_counter()
				elapse_iter_time = ( curr_iter_time - start_iter_time ) + theo_cumm_iter_time

				variance_inIter_record.append( time= elapse_iter_time, spp= iter_spp_count, cumm_spp= theo_cumm_iter_spp + iter_spp_count, iteration= iteration_count, variance= variance )
				variance_groundTruth_inIter_record.append( time= elapse_iter_time, spp= iter_spp_count, cumm_spp= theo_cumm_iter_spp + iter_spp_count, iteration= iteration_count, variance= variance_groundTruth )
				mse_groundTruth_inter_record.append( time= elapse_iter_time, spp= iter_spp_count, cumm_spp= theo_cumm_iter_spp + iter_spp_count, iteration= iteration_count, mse= mse_groundTruth )

			
			render_progressbar.update( round( 100 * iter_spp_count / iter_spp ) )

		render_progressbar.finish()

		# 
		# 	Compute Estimated Variance of the final image
		# 

		if not isRecordPerformanceInIteration:
			
			# 		Variance wrt. self
			variance = pathGuidingIntegrator.computeVariance( iter_spp_count )

			# 		Variance wrt. ground truth
			variance_groundTruth = pathGuidingIntegrator.computeVariance( iter_spp_count, groundTruth )

			# 		Mean Square Error wrt. ground truth
			mse_groundTruth = pathGuidingIntegrator.computeMSE( iter_spp_count, groundTruth )

		
		# Record end iteration variance
		variance_endIter_record.append( time= elapse_iter_time, spp= iter_spp_count, cumm_spp= theo_cumm_iter_spp + iter_spp_count, iteration= iteration_count, variance= variance )
		variance_groundTruth_endIter_record.append( time= elapse_iter_time, spp= iter_spp_count, cumm_spp= theo_cumm_iter_spp + iter_spp_count, iteration= iteration_count, variance= variance_groundTruth )
		mse_groundTruth_endIter_record.append( time= elapse_iter_time, spp= iter_spp_count, cumm_spp= theo_cumm_iter_spp + iter_spp_count, iteration= iteration_count, mse= mse_groundTruth )


		printBoldUnderLine('Variance:', variance)
		printBoldUnderLine('Variance wrt. Ground Truth:', variance_groundTruth)
		printBoldUnderLine('Mean Square Error wrt. Ground Truth:', mse_groundTruth)


		# Update theoretical iter time
		theo_cumm_iter_time += theo_iter_time

		# Save image
		imageFileName = FileNameManager.generateImageFileName( iteration_count, iter_spp_count )
		mi.util.write_bitmap( imageFileName + '.png', image)
		mi.util.write_bitmap( imageFileName + '.exr', image)


	# Save performance record: variance, MSE to file
	if isRecordPerformanceInIteration:
		variance_inIter_record.saveToFile( FileNameManager.PERFORMANCE_FOLDER_PATH + f'variance_inIter_high_spp_sim-{sim_iter}.csv' )
		variance_groundTruth_inIter_record.saveToFile( FileNameManager.PERFORMANCE_FOLDER_PATH + f'variance_groundTruth_inIter_high_spp_sim-{sim_iter}.csv' )
		mse_groundTruth_inter_record.saveToFile( FileNameManager.PERFORMANCE_FOLDER_PATH + f'mse_groundTruth_inIter_high_spp_sim-{sim_iter}.csv' )

	variance_endIter_record.saveToFile( FileNameManager.PERFORMANCE_FOLDER_PATH + f'variance_endIter_high_spp_sim-{sim_iter}.csv' )
	variance_groundTruth_endIter_record.saveToFile( FileNameManager.PERFORMANCE_FOLDER_PATH + f'variance_groundTruth_endIter_high_spp_sim-{sim_iter}.csv' )
	mse_groundTruth_endIter_record.saveToFile( FileNameManager.PERFORMANCE_FOLDER_PATH + f'mse_groundTruth_endIter_high_spp_sim-{sim_iter}.csv' )



if __name__ == '__main__':

	# Load scene
	# sceneFileName = 'scenes/cornell-box/scene.xml'
	# sceneFileName = 'scenes/cornell-box/cornell-box-empty.xml'
	# sceneFileName = 'scenes/teapot/teapot_compact.xml'
	# sceneFileName = 'scenes/veach-mis/scene.xml'
	# sceneFileName = 'scenes/veach-bidir/scene.xml'
	# sceneFileName = 'scenes/veach-ajar/scene.xml'
	sceneFileName = 'scenes/torus/scene.xml'

	scene = mi.load_file( sceneFileName )
	sceneName = 'torus'	# for file saving

	# Load ground truth image
	groundTruthFileName = 'scenes/torus/TungstenRender.exr'
	groundTruthImage = mi.TensorXf( mi.Bitmap( groundTruthFileName ) )
	# Reshape into 1D array
	groundTruth = dr.unravel( mi.Color3f, groundTruthImage.array )

	# Set FileNameManager scene name and create debug folders if not exist yet
	FileNameManager.setSceneName( sceneName )
	FileNameManager.createDebugFolder()

	# Read scene properties 
	pathGuidingIntegrator: PathGuidingIntegrator = scene.integrator()
	max_depth = pathGuidingIntegrator.max_depth
	sensors = scene.sensors()
	film_size = sensors[0].film().size()
	spp = sensors[0].sampler().sample_count()
	bbox: mi.ScalarBoundingBox3f = scene.bbox()

	# Setup Path Guiding Integrator
	epsilon = 1e-4
	pathGuidingIntegrator.setup( 
		numRays=(film_size[0] * film_size[1]), 
		# Extend the scene bounding box by epsilon to handle floating point error when scattering data into
		bbox_min= bbox.min - epsilon, bbox_max= bbox.max + epsilon,
		sdTreeMaxDepth = 20,
		quadTreeMaxDepth = 20,
		isStoreNEERadiance = False,
		bsdfSamplingFraction = 0.5
	)

	# 	Iter : SPP : cumulative SPP
	# 	1	 2	  3		4	 5	   6	 7		8		9		10		11
	# 	4	 8	  16	32	 64	   128	 256	512		1024	2048	4096
	# 	4	 12	  28	60	 124   252	 508	1020	2044	4092	8188

	max_tree_iteration = 9		# this can be use to determine when to stop loading tree
	start_iteration = 0
	end_iteration = 9

	iter_spp = 1024

	# Use for batch rendering in the final iteration where
	# there is no restriction for 1spp per pass
	batch_spp = 1

	# Number of total simulation. Use this to accumulate data across all simulation
	total_simulation = 2

	# Run all simulation
	for sim_iter in range( total_simulation ):
		printTitle(f'--- Simulation Iteration: {sim_iter} ---')
		doFullSimulation( start_iteration, end_iteration, max_tree_iteration, iter_spp, batch_spp, sim_iter )


	# Average performance result across every simulation

	printTitle('Average data across simulation')
	
	variance_inIter_DF_list = []
	variance_groundTruth_inIter_DF_list = []
	mse_groundTruth_inter_DF_list = []
	variance_endIter_DF_list = []
	variance_groundTruth_endIter_DF_list = []
	mse_groundTruth_endIter_DF_list = []

	for sim_iter in range( total_simulation ):

		variance_inIter_DF = pd.read_csv( FileNameManager.PERFORMANCE_FOLDER_PATH + f'variance_inIter_high_spp_sim-{sim_iter}.csv' )
		variance_groundTruth_inIter_DF = pd.read_csv( FileNameManager.PERFORMANCE_FOLDER_PATH + f'variance_groundTruth_inIter_high_spp_sim-{sim_iter}.csv' )
		mse_groundTruth_inter_DF = pd.read_csv( FileNameManager.PERFORMANCE_FOLDER_PATH + f'mse_groundTruth_inIter_high_spp_sim-{sim_iter}.csv' )
		variance_endIter_DF = pd.read_csv( FileNameManager.PERFORMANCE_FOLDER_PATH + f'variance_endIter_high_spp_sim-{sim_iter}.csv' )
		variance_groundTruth_endIter_DF = pd.read_csv( FileNameManager.PERFORMANCE_FOLDER_PATH + f'variance_groundTruth_endIter_high_spp_sim-{sim_iter}.csv' )
		mse_groundTruth_endIter_DF = pd.read_csv( FileNameManager.PERFORMANCE_FOLDER_PATH + f'mse_groundTruth_endIter_high_spp_sim-{sim_iter}.csv' )

		variance_inIter_DF_list.append( variance_inIter_DF )
		variance_groundTruth_inIter_DF_list.append( variance_groundTruth_inIter_DF )
		mse_groundTruth_inter_DF_list.append( mse_groundTruth_inter_DF )
		variance_endIter_DF_list.append( variance_endIter_DF )
		variance_groundTruth_endIter_DF_list.append( variance_groundTruth_endIter_DF )
		mse_groundTruth_endIter_DF_list.append( mse_groundTruth_endIter_DF )
	
	
	variance_inIter_avg_DF = pd.concat( variance_inIter_DF_list ).groupby(level= 0).mean()
	variance_groundTruth_inIter_avg_DF =  pd.concat( variance_groundTruth_inIter_DF_list ).groupby(level= 0).mean()
	mse_groundTruth_inter_avg_DF =  pd.concat( mse_groundTruth_inter_DF_list ).groupby(level= 0).mean()
	variance_endIter_avg_DF =  pd.concat( variance_endIter_DF_list ).groupby(level= 0).mean()
	variance_groundTruth_endIter_avg_DF =  pd.concat( variance_groundTruth_endIter_DF_list ).groupby(level= 0).mean()
	mse_groundTruth_endIter_avg_DF =  pd.concat( mse_groundTruth_endIter_DF_list ).groupby(level= 0).mean()



	# Save average performance result
	variance_inIter_avg_DF.to_csv( FileNameManager.PERFORMANCE_FOLDER_PATH + f'variance_inIter_high_spp_avg-{total_simulation}-sim.csv' )
	variance_groundTruth_inIter_avg_DF.to_csv( FileNameManager.PERFORMANCE_FOLDER_PATH + f'variance_groundTruth_inIter_high_spp_avg-{total_simulation}-sim.csv' )
	mse_groundTruth_inter_avg_DF.to_csv( FileNameManager.PERFORMANCE_FOLDER_PATH + f'mse_groundTruth_inIter_high_spp_avg-{total_simulation}-sim.csv' )
	variance_endIter_avg_DF.to_csv( FileNameManager.PERFORMANCE_FOLDER_PATH + f'variance_endIter_high_spp_avg-{total_simulation}-sim.csv' )
	variance_groundTruth_endIter_avg_DF.to_csv( FileNameManager.PERFORMANCE_FOLDER_PATH + f'variance_groundTruth_endIter_high_spp_avg-{total_simulation}-sim.csv' )
	mse_groundTruth_endIter_avg_DF.to_csv( FileNameManager.PERFORMANCE_FOLDER_PATH + f'mse_groundTruth_endIter_high_spp_avg-{total_simulation}-sim.csv' )