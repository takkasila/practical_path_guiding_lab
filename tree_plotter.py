from __future__ import annotations

import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from src.common import *

from src.quadtree import QuadTreeNode
from src.kdtree import KDTreeNode
from src.file_name_manager import FileNameManager

import numpy as np
import matplotlib.pyplot as plt

class QuadTreePlotter:

	def __init__( self, fileName: str ) -> None:
		"""
			QuadTree data represented as numpy array fashion.
			Mainly use for offline data processing.
			- fileName: file name of the QuadTree data to be loaded.
		"""

		# Load from file
		treeDataNumpy = np.load(fileName)

		# Init the QuaddTreeNode and load data into
		self.quadTreeNode = QuadTreeNode()
		self.quadTreeNode.loadFromFile( treeDataNumpy )

	
	def getMaxDepth( self, rootIndex: int ) -> int:
		"""
			Get maximum depth of a QuadTree
			-
		"""
		leafNodeIndex = self.quadTreeNode.getAllLeafNodeIndex( mi.UInt32( rootIndex ) )
		leafNodeDepth = dr.gather( mi.UInt32, self.quadTreeNode.depth, leafNodeIndex )
		maxDepth = np.max(leafNodeDepth.numpy())

		return maxDepth

	
	def sampleIrradiance( self, rootIndex: mi.UInt32, position: mi.Vector2f ) -> mi.Float:
		"""
			Sample irradiance from a given postion. 'rootIndex' and 'position' MUST have
			the same size.
			- rootIndex: index of tree root (0, 1, 2, ...).
			- position: sampling position.
		"""
		# 	Start searching at root node index
		nodeIndex = dr.gather( mi.UInt32, self.quadTreeNode.rootNodeIndex, rootIndex )

		# Test if data is within the root node bbox
		rootNodeBBox = self.quadTreeNode.getBBox( nodeIndex )
		isInsideRoot = rootNodeBBox.contains( position )
		active = mi.Bool( isInsideRoot )

		# Traverse the tree to find the corresponding leaf node
		loop = mi.Loop(name= 'QuadTreeDataPlotter sample irradiance', state= lambda: (active, nodeIndex) )
		while loop(active):

			# If at leaf node then stop, if not then continue
			isLeafNode = dr.gather(mi.Bool, self.quadTreeNode.isLeaf, nodeIndex, active )
			isNotLeafNode = ~isLeafNode
			active &= isNotLeafNode

			# Else, check which child node it belongs to and store the node index
			child_1_idx = dr.gather(mi.UInt32, self.quadTreeNode.child_1_index, index= nodeIndex, active= active)
			child_2_idx = dr.gather(mi.UInt32, self.quadTreeNode.child_2_index, index= nodeIndex, active= active)
			child_3_idx = dr.gather(mi.UInt32, self.quadTreeNode.child_3_index, index= nodeIndex, active= active)
			child_4_idx = dr.gather(mi.UInt32, self.quadTreeNode.child_4_index, index= nodeIndex, active= active)

			child_1_bbox = self.quadTreeNode.getBBox( child_1_idx )
			child_1_bbox_test = child_1_bbox.contains( position )
			nodeIndex[child_1_bbox_test & active] = child_1_idx

			child_2_bbox = self.quadTreeNode.getBBox( child_2_idx )
			child_2_bbox_test = child_2_bbox.contains( position )
			nodeIndex[child_2_bbox_test & active] = child_2_idx

			child_3_bbox = self.quadTreeNode.getBBox( child_3_idx )
			child_3_bbox_test = child_3_bbox.contains( position )
			nodeIndex[child_3_bbox_test & active] = child_3_idx

			child_4_bbox = self.quadTreeNode.getBBox( child_4_idx )
			child_4_bbox_test = child_4_bbox.contains( position )
			nodeIndex[child_4_bbox_test & active] = child_4_idx

		# Get irradiance from the corresponding leaf node index
		irradiance = dr.gather( mi.Float, self.quadTreeNode.irradiance, nodeIndex, isInsideRoot )
		# Compute area of the node
		nodeBBox = self.quadTreeNode.getBBox( nodeIndex )
		nodeBBoxExtent = (nodeBBox.max - nodeBBox.min)
		nodeArea = nodeBBoxExtent.x * nodeBBoxExtent.y

		# Normalized irradiance by the array size
		irradiance /= nodeArea

		return irradiance
	
	
	def plotQuadTree( self, rootIndex: int, title: str, saveFig: bool = False ) -> None:
		"""
			Plot a QuadTree as heat map.
			- rootIndex: index of tree root (0, 1, 2, ...).
		"""

		# Generate sampling position in a grid fashion
		depth = self.getMaxDepth( 0 )
		depth = min( depth, 10 )
		numCell = pow( 2, depth )

		cellSize = 1 / numCell
		cellCenterOffset =  cellSize / 2
		x = dr.arange( mi.Float, numCell ) / numCell
		y = dr.arange( mi.Float, numCell ) / numCell
		x += cellCenterOffset
		y += cellCenterOffset

		x, y = dr.meshgrid( x, y )

		samplingPosition = mi.Vector2f( x, y )

		N = dr.width( samplingPosition )

		# Sample QuadTree irradiance
		rootIndexArray = dr.full( mi.UInt32, rootIndex, N )
		irradiance = self.sampleIrradiance( rootIndexArray, samplingPosition )
		irradiance_numpy = irradiance.numpy().reshape(numCell, numCell)

		# Plot heatmap
		plt.figure()
		im = plt.imshow( irradiance_numpy, cmap='jet', interpolation='nearest', extent= [0, 1, 0, 1], origin= 'lower')
		plt.xlabel( 'Normalized Φ (Phi)' )
		plt.ylabel( 'Normalized Cos(θ) (Theta)' )
		plt.title( title )
		plt.colorbar( im )

		if saveFig:
			# Title contains new line character so we join into one line
			figFileName = title.splitlines()
			figFileName = FileNameManager.PLOT_FOLDER_PATH + figFileName[0] + ', ' + figFileName[1] + '.png'
			plt.savefig(fname= figFileName, dpi=300)
		

		# samplingPos = mi.Vector2f( 0.25, 0.25 )
		# rootIndexArray = dr.full( mi.UInt32, rootIndex, 1 )
		# irradiance = self.sampleIrradiance( rootIndexArray, samplingPos )
		# print(irradiance)


class KDTreePlotter:

	def __init__(self, fileName: str) -> None:
		
		# Load from file
		treeDataNumpy = np.load(fileName)

		# Init the QuaddTreeNode and load data into
		self.kdTreeNode = KDTreeNode()
		self.kdTreeNode.loadFromFile( treeDataNumpy  )

		# Init QuadTreePlotter
		self.quadTreePlotter = QuadTreePlotter( fileName )


	def getSceneBBox( self ) -> Tuple[ Vec3, Vec3 ]:
		pass


	def findLeafNode( self, position: mi.Vector3f ) -> Tuple[ mi.UInt32, mi.Bool ]:
		"""
			Find the corresponding leaf node index of the given position.
			- position: query position.

			Return:
			- nodeIndex
			- isValid: 
		"""

		# Start traversing from root node
		nodeIndex = dr.zeros( mi.UInt32, dr.width( position ) )

		# 	Test if data is within the root node bbox
		rootNodeBBox = self.kdTreeNode.getBBox( 0 )
		isInsideRoot = rootNodeBBox.contains( position )
		active = mi.Bool( isInsideRoot )

		loop = mi.Loop( name = 'KDTreePlotter.findLeafNode', state= lambda: (active, nodeIndex) )
		while loop( active ):

			# If at leaf node then stop. Otherwise, continue
			isLeafNode = dr.gather( mi.Bool, self.kdTreeNode.isLeaf, nodeIndex, active )
			
			# Else, check which children node it belongs to and store the node index
			isNotLeafNode = ~isLeafNode
			active &= isNotLeafNode 

			child_left_idx = dr.gather( mi.UInt32, self.kdTreeNode.child_left_index, nodeIndex, active )
			child_right_idx = dr.gather( mi.UInt32, self.kdTreeNode.child_right_index, nodeIndex, active )

			child_left_bbox = self.kdTreeNode.getBBox( child_left_idx )
			child_left_bbox_test = child_left_bbox.contains( position )
			nodeIndex[ child_left_bbox_test & active ] = child_left_idx

			child_right_bbox = self.kdTreeNode.getBBox( child_right_idx )
			child_right_bbox_test = child_right_bbox.contains( position )
			nodeIndex[ child_right_bbox_test & active ] = child_right_idx

		
		return nodeIndex, isInsideRoot


	def plotQuadTreeAtPosition( self, position: Vec3, title: str, saveFig: bool = False ) -> None:
		
		# Traverse the KDTree to find the corresponding rootnode
		position_mi = mi.Vector3f( position )
		leafNodeIndex_mi, isValid_mi = self.findLeafNode( position_mi )
		leafNodeIndex = leafNodeIndex_mi.numpy()[0]
		isValid = isValid_mi.numpy()[0]


		# If the query position is valid (withing tree bounding box) then plot the corresponding quadtree
		if isValid:
			# Get the corresponding root node index
			quadTreeRootIndex = dr.gather( mi.UInt32, self.kdTreeNode.quadTreeRootIndex, leafNodeIndex )

			# Plot the QuadTree
			self.quadTreePlotter.plotQuadTree( quadTreeRootIndex.numpy()[0], title, saveFig )


class MultiIterationTreePlotter:
	
	def __init__(self, sceneName: str, numIteration: int) -> None:
		"""
			- sceneName: the scene name of the file. Not the actual file path.
			- numIteration: total number of iteration
		"""

		self.sceneName = sceneName
		self.kdTreePlotters: List[ KDTreePlotter ] = []
		
		# Initialize KDTreePlotter for refinement each iteration
		for i in range( numIteration ):
			fileName = FileNameManager.generateTreeDataFileName( i, withNpzEnding= True )
			kdTreePlotter = KDTreePlotter( fileName )

			self.kdTreePlotters.append( kdTreePlotter )

	
	def plotQuadTreeAtPosition( self, position: Vec3, saveFig: bool = False ) -> None:
		"""
			Plot a QuadTree of every iteration of a given world position.
			- position: world position. Will be use to traverse a KDTree to find a corresponding QuadTree.
		"""
		# Plot the refinement map at every iteration
		for index, kdTreePlotter in enumerate( self.kdTreePlotters ):
			title = self.sceneName + ' Irradiance Map\nworld_pos= [{0}, {1}, {2}], Refinement Iteration= {3}' \
					.format( position[0], position[1], position[2], index )
			kdTreePlotter.plotQuadTreeAtPosition( position, title, saveFig )

		# Show plot
		plt.show()
		

if __name__ == '__main__':

	# sceneName = 'veach-bidir'
	sceneName = 'torus'
	# sceneName = 'cornell-box-empty'
	iteration = 5
	FileNameManager.setSceneName( sceneName )
	multiIterationTreePlotter = MultiIterationTreePlotter( sceneName, iteration )
	# multiIterationTreePlotter.plotQuadTreeAtPosition( [0, 0, 0], saveFig= True )
	# multiIterationTreePlotter.plotQuadTreeAtPosition( [-2.38, 1.9, 0.4617], saveFig= True )	# 	veach-bidir behind egg
	# multiIterationTreePlotter.plotQuadTreeAtPosition( [-1.749760, 0.157591, 3.000237], saveFig= True )	# veach-bidir:	This position is in mid air
	multiIterationTreePlotter.plotQuadTreeAtPosition( [-6.7, 1.5, -1.828], saveFig= True )	# torus, bright spot in the shadow
