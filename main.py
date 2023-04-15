import drjit as dr
# dr.set_log_level(dr.LogLevel.Info)
import mitsuba as mi
# print( 'Mitsuba from:', mi.__file__ )
mi.set_variant('cuda_ad_rgb')

from src.path_guiding_integrator import PathGuidingIntegrator

from src.file_name_manager import FileNameManager

import matplotlib.pyplot as plt

from src.common import PerformanceData, printTitle, printBoldUnderLine
from typing import List


from random import randint
import time

import math

import progressbar


if __name__ == '__main__':

	# Load scene
	# sceneFileName = 'scenes/cornell-box/scene.xml'
	sceneFileName = 'scenes/torus/scene.xml'
	# sceneFileName = 'scenes/veach-bidir/scene.xml'
	# sceneFileName = 'scenes/veach-ajar/scene.xml'
	# sceneFileName = 'scenes/kitchen/scene.xml'

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

	epsilon = 1e-4
	pathGuidingIntegrator.setup( 
		numRays=(film_size[0] * film_size[1]), 
		# Extend the scene bounding box by epsilon to handle floating point error when scattering data into
		bbox_min= bbox.min - epsilon, bbox_max= bbox.max + epsilon,
		sdTreeMaxDepth = 20,
		quadTreeMaxDepth = 20,
		isStoreNEERadiance = True,
		bsdfSamplingFraction = 0.5
	)

	# Set to 0 if want reproductibility across simulation
	initiali_seed = randint(0, 1000000)
	printBoldUnderLine('Initial seed:', initiali_seed)

	# Render image buffer
	image = None
	curr_iter_image = None
	prev_iter_image = None

	# Record performance data
	isRecordPerformanceInIteration = True

	variance_inIter_record = PerformanceData()
	variance_groundTruth_inIter_record = PerformanceData()
	mse_groundTruth_inter_record = PerformanceData()

	variance_endIter_record = PerformanceData()
	variance_groundTruth_endIter_record = PerformanceData()
	mse_groundTruth_endIter_record = PerformanceData()

	variance_estimated_final_record = PerformanceData()

	variance = 0
	variance_groundTruth = 0
	mse_groundTruth = 0

	# 	Iter : SPP : cumulative SPP
	# 	1	 2	  3		4	 5	   6	 7		8		9		10		11
	# 	4	 8	  16	32	 64	   128	 256	512		1024	2048	4096
	# 	4	 12	  28	60	 124   252	 508	1020	2044	4092	8188

	# Total budget for both training and rendering
	# budget_spp = 28
	budget_spp = 252
	# budget_spp = 2044
	# budget_spp = 19444
	# budget_spp = 10000

	# Compute all possible cumm_spp of every iteration
	def computePossibleCummSPP( budget_spp: int ) -> List[int]:
		cumm_spp = 0
		iter_count = 0
		cummSPPs = []
		while cumm_spp < budget_spp:
			
			iter_spp = 2 ** (iter_count + 2)
			cumm_spp += iter_spp

			cummSPPs.append( cumm_spp )
			iter_count += 1
		
		return cummSPPs
	
	possibleCummSPPs = computePossibleCummSPP( budget_spp )

	# Use for batch rendering in the final iteration where
	# there is no restriction for 1spp per pass
	batch_spp = 4
	# batch_spp = 1

	#  Stable Variance SPP threashold deciding how much budget need to spend
	# 	in order to have a reliable variance
	stable_variance_spp_threshold = 256
	# stable_variance_spp_threshold = 100000

	# Cumulative SPP from start to current iteration
	cumm_spp = 0
	cumm_spp_prev = cumm_spp

	# Number of SPP used to generate the current image.
	# This could be more than just one iteration budget
	# because we could accumulate across upto two iterations.
	image_spp = 0

	remainingSPP = budget_spp
	isFinalIter = False
	isTrainSDTree = True
	isClearData = True

	iteration_count = 0

	variance_prev = 0
	variance_current = 0

	# 	Performance timer
	# 	Only record time relating to main computation, no file saving
	cumm_time = 0
	elapse_time = 0 

	# Iteratively rendering and training with certain criteria
	# Continue as long as still have budget
	while remainingSPP > 0:
		
		start_iter_time = time.perf_counter()

		if isClearData:
			pathGuidingIntegrator.resetVarianceCounter()
			# image = None
			image_spp = 0
		
		curr_iter_image = None

		# Compute SPP for this iteration
		if not isFinalIter:
			iter_spp = 2 ** ( iteration_count + 2 )	# 4 8 16 ...
			
			# If we used up exactly the remaining SPP then it is final.
			if iter_spp == remainingSPP:
				isFinalIter = True
			
		else:
			iter_spp = remainingSPP

			curr_iter_image_acc = None
		

		# Update integrator's iteration count
		pathGuidingIntegrator.setIteration( iteration_count, isFinalIter )

		printTitle(f'Iteration {iteration_count}')
		print( f'SPP: {iter_spp}, cumm_SPP: {cumm_spp}, remaining budget: {budget_spp - cumm_spp}, isFinalIter: {isFinalIter}' )
		
		# 
		# 	Render pass
		# 
		
		spp_per_pass = 1

		# 	In the final iteration, we doesn't record surface interactions in each pass,
		# 	so we can render multiple SPP per pass for better performance
		if isFinalIter:
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
			if curr_iter_image is None:
				curr_iter_image = weighted_image_one_pass
			else:
				curr_iter_image += weighted_image_one_pass
			
			# For image blending in final iteration
			if isFinalIter:
				if curr_iter_image_acc is None:
					curr_iter_image_acc = image_one_pass
				else:
					curr_iter_image_acc += image_one_pass
				
				dr.schedule( curr_iter_image_acc )


			# Evaluate the computation so that it won't keep putting 
			# into cache and then exceed the memory limit
			dr.eval( curr_iter_image )

			image_spp += curr_pass_spp
			iter_spp_count += curr_pass_spp
			cumm_spp += curr_pass_spp

			if isRecordPerformanceInIteration:
				# 
				# 	Compute Variance and MSE within iteration
				# 
				
				# 		Variance wrt. self
				variance = pathGuidingIntegrator.computeVariance( image_spp )

				# 		Variance wrt. ground truth
				variance_groundTruth = pathGuidingIntegrator.computeVariance( image_spp, groundTruth )

				# 		Mean Square Error wrt. ground truth
				mse_groundTruth = pathGuidingIntegrator.computeMSE( image_spp, groundTruth )

				# Record Variance and MSE
				curr_iter_time = time.perf_counter()
				elapse_time = ( curr_iter_time - start_iter_time ) + cumm_time

				variance_inIter_record.append( time= elapse_time, spp= image_spp, cumm_spp= cumm_spp, iteration= iteration_count, variance= variance )
				variance_groundTruth_inIter_record.append( time= elapse_time, spp= image_spp, cumm_spp= cumm_spp, iteration= iteration_count, variance= variance_groundTruth )
				mse_groundTruth_inter_record.append( time= elapse_time, spp= image_spp, cumm_spp= cumm_spp, iteration= iteration_count, mse= mse_groundTruth )
			
			if isFinalIter and cumm_spp in possibleCummSPPs:
				# If at the final iter, it could happen that we use batch rendering but still would like render result in
				# 	cumm_spp that equal to iteration spp.

				current_iter_spp_count = cumm_spp - cumm_spp_prev
				curr_iter_image_non_weighted = curr_iter_image_acc / (pass_i + 1)
				image = (curr_iter_image_non_weighted * current_iter_spp_count + prev_iter_image * (image_spp - current_iter_spp_count) ) / image_spp

				# Save image
				imageFileName = FileNameManager.generateImageFileName( iteration_count, image_spp )
				mi.util.write_bitmap( imageFileName + f'_cumm_spp-{cumm_spp}' + '.png', image)
				mi.util.write_bitmap( imageFileName + f'_cumm_spp-{cumm_spp}' + '.exr', image)
			
			render_progressbar.update( round( 100 * iter_spp_count / iter_spp ) )

		render_progressbar.finish()

		# When arriving at the final iteration and did not train the SDTree in the previous iteration,
		# it means than we continue rendering the previous iteration image with the rest of budget.
		# Thus, we need to combine both of them weight on samples spent
		if isFinalIter and not isTrainSDTree:
			image = (curr_iter_image * iter_spp + prev_iter_image * (image_spp - iter_spp) ) / image_spp

		else:
			image = curr_iter_image

		# 
		# 	Compute Estimated Variance of the final image
		# 

		if not isRecordPerformanceInIteration:
			
			# 		Variance wrt. self
			variance = pathGuidingIntegrator.computeVariance( image_spp )

			# 		Variance wrt. ground truth
			variance_groundTruth = pathGuidingIntegrator.computeVariance( image_spp, groundTruth )

			# 		Mean Square Error wrt. ground truth
			mse_groundTruth = pathGuidingIntegrator.computeMSE( image_spp, groundTruth )

			curr_iter_time = time.perf_counter()
			elapse_time = ( curr_iter_time - start_iter_time ) + cumm_time

		
		# Record end iteration variance
		variance_endIter_record.append( time= elapse_time, spp= image_spp, cumm_spp= cumm_spp, iteration= iteration_count, variance= variance )
		variance_groundTruth_endIter_record.append( time= elapse_time, spp= image_spp, cumm_spp= cumm_spp, iteration= iteration_count, variance= variance_groundTruth )
		mse_groundTruth_endIter_record.append( time= elapse_time, spp= image_spp, cumm_spp= cumm_spp, iteration= iteration_count, mse= mse_groundTruth )


		printBoldUnderLine('Variance:', variance)
		printBoldUnderLine('Variance wrt. Ground Truth:', variance_groundTruth)
		printBoldUnderLine('Mean Square Error wrt. Ground Truth:', mse_groundTruth)


		# Compute Estimated Variance
		budget_upto_prev = budget_spp - cumm_spp_prev
		variance_current = ( variance * image_spp ) / budget_upto_prev

		printBoldUnderLine('Estimated Variance Final Image:', variance_current)
		variance_estimated_final_record.append( time= elapse_time, spp= image_spp, cumm_spp= cumm_spp, iteration= iteration_count, variance= variance_current )
		
		
		# 
		# 	Compute next iteration conditions
		# 
		nextIterSPP = 2 ** ( iteration_count + 3 )

		remainingSPP = budget_spp - cumm_spp

		if nextIterSPP < remainingSPP:

			# Allow guarantee training for certain period
			# After than period, if the current variance is higer than previous,
			# then stop training
			stopTrainingCondition = (cumm_spp > stable_variance_spp_threshold) & (variance_current > variance_prev)

			if cumm_spp >= 1000:
				stopTrainingCondition = True


			if stopTrainingCondition:
				isFinalIter = True
				isTrainSDTree = False
				isClearData = False

		
		elif nextIterSPP == remainingSPP:
			# If we are going to use up exactly the remaining budget,
			# then then it is a final iteration. But we still need to check whether continue training or not

			isFinalIter = True

			stopTrainingCondition = (cumm_spp > stable_variance_spp_threshold) & (variance_current > variance_prev)

			if cumm_spp >= 1000:
				stopTrainingCondition = True


			if stopTrainingCondition:
				isTrainSDTree = False
				isClearData = False

		else:
			# Use the remainging budget for rendering
		
			# Final iteration
			isFinalIter = True
			isTrainSDTree = False
			isClearData = False
		


		# Refine and Prepare SDTree for the next iteration
		if isTrainSDTree:
			pathGuidingIntegrator.refineAndPrepareSDTreeForNextIteration()

		# When we stop training for the first time
		elif not isTrainSDTree and prev_iter_image is None:
			printBoldUnderLine('-- Stop training SDTree --')

		# Save previous iteration image for two iteration blending
		prev_iter_image = image
		

		# Only record time relating to main computation, no file saving
		end_iter_time = time.perf_counter()
		cumm_time += end_iter_time - start_iter_time


		# Save image
		imageFileName = FileNameManager.generateImageFileName( iteration_count, image_spp )
		mi.util.write_bitmap( imageFileName + f'_cumm_spp-{cumm_spp}' + '.png', image)
		mi.util.write_bitmap( imageFileName + f'_cumm_spp-{cumm_spp}' + '.exr', image)

		# Save SDTree data of this iteration 
		# Always save the tree even if it's in the final iteration (which has no modification compared to the previous iteration)
		# for convenience
		quadTreeFileName = FileNameManager.generateTreeDataFileName( iteration_count )
		pathGuidingIntegrator.saveSDTreeToFile( quadTreeFileName )

		# Save KDTree bounding box of this iteration 
		quadTreeOBJFileName = FileNameManager.generateOBJFileName( iteration_count )
		pathGuidingIntegrator.saveSDTreeOBJ( quadTreeOBJFileName )

		# Update loop variable
		variance_prev = variance_current
		iteration_count += 1
		cumm_spp_prev = cumm_spp
	

	# Save performance record: variance, MSE to file
	if isRecordPerformanceInIteration:
		variance_inIter_record.saveToFile( FileNameManager.PERFORMANCE_FOLDER_PATH + 'variance_inIter.csv' )
		variance_groundTruth_inIter_record.saveToFile( FileNameManager.PERFORMANCE_FOLDER_PATH + 'variance_groundTruth_inIter.csv' )
		mse_groundTruth_inter_record.saveToFile( FileNameManager.PERFORMANCE_FOLDER_PATH + 'mse_groundTruth_inIter.csv' )

	variance_endIter_record.saveToFile( FileNameManager.PERFORMANCE_FOLDER_PATH + 'variance_endIter.csv' )
	variance_groundTruth_endIter_record.saveToFile( FileNameManager.PERFORMANCE_FOLDER_PATH + 'variance_groundTruth_endIter.csv' )
	mse_groundTruth_endIter_record.saveToFile( FileNameManager.PERFORMANCE_FOLDER_PATH + 'mse_groundTruth_endIter.csv' )

	variance_estimated_final_record.saveToFile( FileNameManager.PERFORMANCE_FOLDER_PATH + 'variance_estimated_final.csv' )

	# Show image
	plt.axis('off')
	plt.imshow(image ** (1.0 / 2.2))
	# plt.savefig( sceneName, bbox_inches='tight', dpi=300, transparent=True, pad_inches=0 )
	plt.show()