import pandas as pd
from matplotlib import pyplot as plt

from typing import List, Tuple

from src.file_name_manager import FileNameManager


def saveFig( fig_title: str ):
	figFileName = fig_title.splitlines()
	figFileName = FileNameManager.PLOT_FOLDER_PATH + figFileName[0] + ', ' + figFileName[1] + '.png'
	plt.savefig( fname= figFileName, dpi= 300 )


def plotOneSimulationPerformance( sceneName: str ):

	variance_groundTruth_InIter_DF = pd.read_csv(f'{FileNameManager.PERFORMANCE_FOLDER_PATH}variance_groundTruth_inIter.csv')

	variance_inIter_DF = pd.read_csv(f'{FileNameManager.PERFORMANCE_FOLDER_PATH}variance_inIter.csv')
	
	mse_groundTruth_inIter_DF = pd.read_csv(f'{FileNameManager.PERFORMANCE_FOLDER_PATH}mse_groundTruth_inIter.csv')

	variance_groundTruth_endIter_DF = pd.read_csv(f'{FileNameManager.PERFORMANCE_FOLDER_PATH}variance_groundTruth_endIter.csv')

	variance_endIter_DF = pd.read_csv(f'{FileNameManager.PERFORMANCE_FOLDER_PATH}variance_endIter.csv')

	mse_groundTruth_endIter_DF = pd.read_csv(f'{FileNameManager.PERFORMANCE_FOLDER_PATH}mse_groundTruth_endIter.csv')

	variance_estimated_final_DF = pd.read_csv(f'{FileNameManager.PERFORMANCE_FOLDER_PATH}variance_estimated_final.csv')


	# 
	# 	In-between Iteration plot
	# 

	# Variance wrt. Ground Truth vs. Cumulative SPP
	plt.figure()
	variance_groundTruth_InIter_DF.set_index( 'cumm_spp', inplace= True )
	variance_groundTruth_InIter_DF_group = variance_groundTruth_InIter_DF.groupby( 'iteration' )
	variance_groundTruth_InIter_DF_group['variance'].plot( loglog= True, legend = True )
	spp_legend = list( map( lambda x: 2 ** (x+2), variance_groundTruth_InIter_DF_group.groups.keys() ) )
	plt.legend( spp_legend )
	plt.xlabel( 'Cumulative SPP' )
	plt.ylabel( 'Variance' )
	title = f'{sceneName}\nIn-iteration Variance (Ground Truth) vs. Cumulative SPP'
	plt.title( title )
	plt.grid( which='minor', markevery= 1, alpha= 0.2)
	plt.grid( which='major', markevery= 10, alpha= 0.5)
	saveFig( title )

	# Variance vs. Cumulative SPP
	plt.figure()
	variance_inIter_DF.set_index( 'cumm_spp', inplace= True )
	variance_inIter_DF.groupby( 'iteration' )['variance'].plot( loglog= True, legend = True )
	plt.legend( spp_legend )
	plt.xlabel( 'Cumulative SPP' )
	plt.ylabel( 'Variance' )
	title = f'{sceneName}\nIn-iteration Variance vs. Cumulative SPP'
	plt.title( title )
	plt.grid( which='minor', markevery= 1, alpha= 0.2)
	plt.grid( which='major', markevery= 10, alpha= 0.5)
	saveFig( title )

	# Mean Square Error wrt. Ground Truth vs. Cumulative SPP
	plt.figure()
	mse_groundTruth_inIter_DF.set_index( 'cumm_spp', inplace= True )
	mse_groundTruth_inIter_DF.groupby( 'iteration' )['mse'].plot( loglog= True, legend = True )
	plt.legend( spp_legend )
	plt.xlabel( 'Cumulative SPP' )
	plt.ylabel( 'Mean Square Error' )
	title = f'{sceneName}\nIn-iteration Mean Square Error (Ground Truth) vs. Cumulative SPP'
	plt.title( title )
	plt.grid( which='minor', markevery= 1, alpha= 0.2)
	plt.grid( which='major', markevery= 10, alpha= 0.5)
	saveFig( title )
	

	# 
	# 	End Iteration plot
	# 

	# Variance wrt. Ground Truth vs. Time
	plt.figure()
	plt.loglog( variance_groundTruth_endIter_DF.time, variance_groundTruth_endIter_DF.variance )
	plt.scatter( variance_groundTruth_endIter_DF.time, variance_groundTruth_endIter_DF.variance )
	plt.xlabel( 'Time(s)' )
	plt.ylabel( 'Variance' )
	title = f'{sceneName}\nEnd-iteration Variance (Ground Truth) vs. Time'
	plt.title( title )
	plt.grid( which='minor', markevery= 1, alpha= 0.2)
	plt.grid( which='major', markevery= 10, alpha= 0.5)
	saveFig( title )
	
	# Variance vs. Time
	plt.figure()
	plt.loglog( variance_endIter_DF.time, variance_endIter_DF.variance )
	plt.scatter( variance_endIter_DF.time, variance_endIter_DF.variance )
	plt.xlabel( 'Time(s)' )
	plt.ylabel( 'Variance' )
	title = f'{sceneName}\nEnd-iteration Variance vs. Time'
	plt.title( title )
	plt.grid( which='minor', markevery= 1, alpha= 0.2)
	plt.grid( which='major', markevery= 10, alpha= 0.5)
	saveFig( title )
	
	# Mean Square Error wrt. Ground Truth vs. Time
	plt.figure()
	plt.loglog( mse_groundTruth_endIter_DF.time, mse_groundTruth_endIter_DF.mse )
	plt.scatter( mse_groundTruth_endIter_DF.time, mse_groundTruth_endIter_DF.mse )
	plt.xlabel( 'Time(s)' )
	plt.ylabel( 'Mean Square Error' )
	title = f'{sceneName}\nEnd-iteration Mean Square Error (Ground Truth) vs. Time'
	plt.title( title )
	plt.grid( which='minor', markevery= 1, alpha= 0.2)
	plt.grid( which='major', markevery= 10, alpha= 0.5)
	saveFig( title )
	
	# Estimated Variance vs. Time
	plt.figure()
	plt.loglog( variance_estimated_final_DF.time, variance_estimated_final_DF.variance )
	plt.scatter( variance_estimated_final_DF.time, variance_estimated_final_DF.variance )
	plt.xlabel( 'Time(s)' )
	plt.ylabel( 'Estimated Final Image Variance' )
	title = f'{sceneName}\nEnd-iteration Estimated Final Image Variance vs. Time'
	plt.title( title )
	plt.grid( which='minor', markevery= 1, alpha= 0.2)
	plt.grid( which='major', markevery= 10, alpha= 0.5)
	saveFig( title )


def plotConvergencePerformance( inIter_high_spp_DF: pd.DataFrame, plot_key: str, intersection_list: List[int] = None ) -> None:
	"""
		Plot average convergence of Variance and MSE
	"""

	# Set index for sorting
	inIter_high_spp_DF.set_index( 'cumm_spp', inplace= True )

	# Group by iteration
	inIter_high_spp_DF_group = inIter_high_spp_DF.groupby( 'iteration' )

	# Get spp for each iteration
	iteration_spp_list = list( map( lambda x: int(2 ** (x+2)), inIter_high_spp_DF_group.groups.keys() ) )

	# Split into "normal" group and "convergence" group
	normal_DF_list = []
	convergence_DF_list = []

	for iteration, dataFrame in inIter_high_spp_DF_group:

		iteration_spp_end = iteration_spp_list[ int(iteration) ]

		normal_DF_list.append( dataFrame.drop( dataFrame[ dataFrame.spp > iteration_spp_end ].index ) )
		convergence_DF_list.append( dataFrame.drop( dataFrame[ dataFrame.spp <= iteration_spp_end -1 ].index ) )


	normal_DF_group = pd.concat(normal_DF_list).groupby( 'iteration' )
	convergence_DF_group = pd.concat(convergence_DF_list).groupby( 'iteration' )

	# Create a new figure
	plt.figure()

	# Plot normal 
	axes = normal_DF_group[ plot_key ].plot( loglog= True, legend = True )

	# Get colors to reuse in the convergence plot
	plot_colors = []
	plot_lines = axes[0].get_lines()
	for plot_line in plot_lines:
		color = plot_line.get_color()
		plot_colors.append( color )

	# Plot convergence
	for key, dataFrame in convergence_DF_group[ plot_key ]:
		dataFrame.plot( loglog= True, legend = True, style= ['--'], color= plot_colors[int(key)] )
	
	# Set some plot properties
	plt.legend( iteration_spp_list[:-1], title= 'iteration spp' )		# [-1] to discard the last iteration plot
	plt.grid( which='minor', markevery= 1, alpha= 0.2)
	plt.grid( which='major', markevery= 10, alpha= 0.5)

	# Plot cross section lines
	for value in intersection_list:
		axes = plt.axvline(x= value, color = 'gray', linestyle= '-.', alpha = 0.8)
	
		# Annotate the corss section line
		min_y = plt.axis()[2]

		plt.text(x = value + 10, y = min_y * 1.1, s= str(value) )


def plotVarianceAndMSEWithConvergence( sceneName, intersection_list: List[int] = None ) -> None:

	# Plot Variance vs. Samples Per Pixel
	variance_groundTruth_InIter_high_spp_DF = pd.read_csv(f'{FileNameManager.PERFORMANCE_FOLDER_PATH}variance_groundTruth_inIter_high_spp_avg-2-sim.csv')
	plotConvergencePerformance( variance_groundTruth_InIter_high_spp_DF, 'variance', intersection_list )
	plt.xlabel( 'Samples Per Pixel' )
	plt.ylabel( 'Variance' )
	title = f'{sceneName}\nVariance Convergence vs. Samples Per Pixel'
	plt.title( title )
	plt.xlim(0, 1000)
	saveFig( title )
	

	# Plot Mean Square Error vs. Samples Per Pixel
	mse_groundTruth_InIter_high_spp_DF = pd.read_csv(f'{FileNameManager.PERFORMANCE_FOLDER_PATH}mse_groundTruth_inIter_high_spp_avg-2-sim.csv')
	plotConvergencePerformance( mse_groundTruth_InIter_high_spp_DF, 'mse', intersection_list )
	plt.xlabel( 'Samples Per Pixel' )
	plt.ylabel( 'Mean Square Error' )
	title = f'{sceneName}\nMean Square Error (Ground Truth) Convergence vs. Samples Per Pixel'
	plt.title( title )
	saveFig( title )


def plotVarianceConvergenceCrossSection( sceneName, cross_values: List[int] ) -> None:

	variance_groundTruth_InIter_high_spp_DF = pd.read_csv(f'{FileNameManager.PERFORMANCE_FOLDER_PATH}variance_groundTruth_inIter_high_spp_avg-2-sim.csv')
	
	# Set index for sorting
	variance_groundTruth_InIter_high_spp_DF.set_index( 'cumm_spp', inplace= True )

	# Group by iteration
	variance_groundTruth_InIter_high_spp_DF_group = variance_groundTruth_InIter_high_spp_DF.groupby( 'iteration' )

	intersectionMap = {}

	# Iterate over cross values 
	for cross_value in cross_values:

		# For each cross value, find the intersected value within the data
		crossValuesMap = {}
		for iteration, dataFrame in variance_groundTruth_InIter_high_spp_DF_group:

			if cross_value in dataFrame.index:
				crossValuesMap[int(iteration)] = dataFrame.at[ cross_value, 'variance' ]
			
		intersectionMap[cross_value] = crossValuesMap
	

	# Create a new figure
	plt.figure()

	line_colors = []

	# Plot every cross value
	for cross_value, crossValueMap in intersectionMap.items():

		# Plot intersection line
		plt.plot(crossValueMap.keys(), crossValueMap.values(), color = 'gray', linestyle= '-.', alpha = 0.8)
		
		# Plot scatter
		pathCollection = plt.scatter(crossValueMap.keys(), crossValueMap.values())


		# Record scatter plot color
		line_colors.append( pathCollection.get_facecolor()[0] )

	plt.xlabel( 'Iteration' )
	plt.ylabel( 'Variance' )
	title = f'{sceneName}\nFixed-budget Variance'
	plt.title( title )

	# Custom legend
	from matplotlib.lines import Line2D
	custom_lines = []
	for line_color in line_colors:
		custom_lines.append( Line2D( [0], [0], markerfacecolor= line_color, marker='o', linestyle='-.', color='gray' ) )

	plt.legend( custom_lines, cross_values, title = 'spp' )

	plt.grid( which='major', markevery= 1, alpha= 0.5 )
	
	saveFig( title )


def convertToIncrementalSPPDataFrame( inIter_dataFrame: pd.DataFrame ) -> pd.DataFrame:
	
	# Set index for sorting
	inIter_dataFrame.set_index( 'cumm_spp', inplace= True )

	# Group by iteration
	inIter_high_spp_DF_group = inIter_dataFrame.groupby( 'iteration' )

	# Get spp for each iteration
	numIter = inIter_high_spp_DF_group.groups.keys()
	iteration_spp_list = list( map( lambda x: int(2 ** (x+2)), inIter_high_spp_DF_group.groups.keys() ) )
	iteration_spp_list.insert( 0, 0 )

	newDataFrameList = []

	# For each iteration
	for iteration, dataFrame in inIter_high_spp_DF_group:

		prev_iter_spp = iteration_spp_list[ int(iteration) ]

		newDataFrameList.append( dataFrame.drop( dataFrame[ dataFrame.spp <= prev_iter_spp ].index ) )


	inIter_Filter_DF = pd.concat(newDataFrameList)
	inIter_Filter_DF.set_index('spp')

	return inIter_Filter_DF


def getStopTrainingSPPAndTime( inIter_dataFrame: pd.DataFrame ) -> Tuple[ int, int ]:

	# Set index for sorting
	inIter_dataFrame.set_index( 'cumm_spp' )

	# Group by iteration
	inIter_high_spp_DF_group = inIter_dataFrame.groupby( 'iteration' )

	# Get last iteration data frame
	lastDF = inIter_high_spp_DF_group.groups[ len(inIter_high_spp_DF_group.groups) - 1 ]

	index = lastDF[0]

	stoptrain_row_data = inIter_dataFrame.loc[index]

	stoptrain_spp = int( stoptrain_row_data['cumm_spp'] )
	stoptrain_time = stoptrain_row_data['time']

	return stoptrain_spp, stoptrain_time


def plotComparingVariancePatgGuideAndPTwNEE( sceneName ) -> None:

	# 
	# 	Variance vs SPP
	# 

	# Scene Variance using Path Guiding
	pathGuidingVariance_DF = pd.read_csv(f'{FileNameManager.PERFORMANCE_FOLDER_PATH}variance_groundTruth_inIter.csv')
	stoptrain_spp, stoptrain_time = getStopTrainingSPPAndTime( pathGuidingVariance_DF )
	pathGuidingVariance_DF = convertToIncrementalSPPDataFrame( pathGuidingVariance_DF )

	plt.figure()
	plt.loglog( pathGuidingVariance_DF.spp, pathGuidingVariance_DF.variance )

	# Scene Variance using Path Tracing with Next Event Estimation
	FileNameManager.setSceneName( '' )
	ptPythonFileName = f'{FileNameManager.DEBUG_FOLDER_PATH}normal-integrator/{sceneName}/{sceneName}_variance_groundTruth_path_tracing_py.csv'
	ptPython_DF = pd.read_csv( ptPythonFileName )
	FileNameManager.setSceneName( sceneName )

	plt.loglog( ptPython_DF.spp, ptPython_DF.variance )

	# Stop training line
	plt.axvline(x= stoptrain_spp, color = 'gray', linestyle= '-.', alpha = 0.8)
	# Annotate the stop training line
	min_y = plt.axis()[2]
	plt.text(x = stoptrain_spp + 10, y = min_y * 1.1, s= str(stoptrain_spp) )

	# Plot settings
	plt.xlabel( 'Samples Per Pixel' )
	plt.ylabel( 'Variance' )
	plt.title( f'{sceneName}\nVariance vs. SPP', weight= 'bold' )
	plt.grid( which='minor', markevery= 1, alpha= 0.2)
	plt.grid( which='major', markevery= 10, alpha= 0.5)
	plt.legend(['Path Guiding', 'Path Tracing w/ NEE'])
	plt.xlim([1, 10 ** 4])

	figFileName = f'{FileNameManager.PLOT_FOLDER_PATH}{sceneName}_variance_vs_ptnee_compare_spp.png'
	plt.savefig( fname= figFileName, dpi= 300 )

	# 
	# 	Variance vs Time
	# 

	# Scene Variance using Path Guiding
	plt.figure()
	plt.loglog( pathGuidingVariance_DF.time, pathGuidingVariance_DF.variance )

	# Scene Variance using Path Tracing with Next Event Estimation
	plt.loglog( ptPython_DF.time, ptPython_DF.variance )

	# Stop training line
	plt.axvline(x= stoptrain_time, color = 'gray', linestyle= '-.', alpha = 0.8)
	min_y = plt.axis()[2]
	plt.text(x = stoptrain_time + 10, y = min_y * 1.1, s= f'{stoptrain_time:.2f}' )

	# Plot settings
	plt.xlabel( 'Time' )
	plt.ylabel( 'Variance' )
	plt.title( f'{sceneName}\nVariance vs. Time', weight= 'bold' )
	plt.grid( which='minor', markevery= 1, alpha= 0.2)
	plt.grid( which='major', markevery= 10, alpha= 0.5)
	plt.legend(['Path Guiding', 'Path Tracing w/ NEE'])
	plt.xlim([1, 10 ** 3])

	figFileName = f'{FileNameManager.PLOT_FOLDER_PATH}{sceneName}_variance_vs_ptnee_compare_time.png'
	plt.savefig( fname= figFileName, dpi= 300 )



def plotCompareMitsubaPathtracingVSPythonPathtracing( sceneName ) -> None:

	# 
	# 	Variance vs SPP
	# 

	plt.figure()

	# Scene Variance using Path Tracing with Next Event Estimation
	FileNameManager.setSceneName( '' )

	ptFileName = f'{FileNameManager.DEBUG_FOLDER_PATH}normal-integrator/{sceneName}/{sceneName}_variance_groundTruth.csv'
	ptMitsuba_DF = pd.read_csv( ptFileName )

	ptPythonFileName = f'{FileNameManager.DEBUG_FOLDER_PATH}normal-integrator/{sceneName}/{sceneName}_variance_groundTruth_path_tracing_py.csv'
	ptPython_DF = pd.read_csv( ptPythonFileName )

	FileNameManager.setSceneName( sceneName )

	plt.loglog( ptMitsuba_DF.spp, ptMitsuba_DF.variance )
	plt.loglog( ptPython_DF.spp, ptPython_DF.variance )

	# Plot settings
	plt.xlabel( 'Samples Per Pixel' )
	plt.ylabel( 'Variance' )
	plt.title( f'{sceneName}\nVariance vs. SPP', weight= 'bold' )
	plt.grid( which='minor', markevery= 1, alpha= 0.2)
	plt.grid( which='major', markevery= 10, alpha= 0.5)
	plt.legend(['Path Tracing', 'Path Tracing Python'])


	# 
	# 	Variance vs Time
	# 

	# Scene Variance using Path Guiding
	plt.figure()

	# Scene Variance using Path Tracing with Next Event Estimation
	plt.loglog( ptMitsuba_DF.time, ptMitsuba_DF.variance )
	plt.loglog( ptPython_DF.time, ptPython_DF.variance )

	# Plot settings
	plt.xlabel( 'Time' )
	plt.ylabel( 'Variance' )
	plt.title( f'{sceneName}\nVariance vs. Time', weight= 'bold' )
	plt.grid( which='minor', markevery= 1, alpha= 0.2)
	plt.grid( which='major', markevery= 10, alpha= 0.5)
	plt.legend(['Path Tracing', 'Path Tracing Python'])

	

if __name__ == '__main__':

	# sceneName = 'cornell-box'
	sceneName = 'torus'
	# sceneName = 'veach-bidir'
	# sceneName = 'veach-ajar'
	# sceneName = 'kitchen'

	FileNameManager.setSceneName( sceneName )

	plotOneSimulationPerformance( sceneName )
	
	# 	In order to run 'plotVarianceAndMSEWithConvergence', and 'plotVarianceConvergenceCrossSection', 
	# 		you have to run 'repeat_high_spp_renderer.py' to get the average performance first!
	# intersection_list = [100, 500]
	# plotVarianceAndMSEWithConvergence( sceneName, intersection_list )

	# plotVarianceConvergenceCrossSection( sceneName, intersection_list )

	#	Have to run 'mitsuba_render_comparison.py' first
	# plotComparingVariancePatgGuideAndPTwNEE( sceneName )

	# plotCompareMitsubaPathtracingVSPythonPathtracing( sceneName )

	plt.show()