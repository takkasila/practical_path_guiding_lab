from __future__ import annotations

import drjit as dr
import mitsuba as mi
if __name__ == '__main__':
	mi.set_variant('cuda_ad_rgb')

from src.common import *

import numpy as np

class QuadTreeNode:

	DRJIT_STRUCT = {
		'rootNodeIndex': mi.UInt32,			# Should be handle independently and carefully
		'bbox': mi.BoundingBox2f,
		'depth' : mi.UInt32,
		'irradiance' : mi.Float,		# radiance / woPodf
		'isLeaf' : mi.Bool,
		'refinementThreshold': mi.Float,
		'child_1_index' : mi.UInt32,
		'child_2_index' : mi.UInt32,
		'child_3_index' : mi.UInt32,
		'child_4_index' : mi.UInt32,
	}

	def __init__(self) -> None:
		self.rootNodeIndex = mi.UInt32()
		self.bbox = mi.BoundingBox2f()
		self.depth = mi.UInt32()
		self.irradiance = mi.Float()
		self.isLeaf = mi.Bool()
		self.refinementThreshold = mi.Float()
		self.child_1_index = mi.UInt32()
		self.child_2_index = mi.UInt32()
		self.child_3_index = mi.UInt32()
		self.child_4_index = mi.UInt32()


	def copyFrom(self, quadTreeNode: QuadTreeNode) -> None:
		"""
			Copy from a QuadTreeNode.
		"""
		self.rootNodeIndex = mi.UInt32( quadTreeNode.rootNodeIndex )
		self.bbox = mi.BoundingBox2f()
		self.bbox.min = mi.Vector2f( quadTreeNode.bbox.min )
		self.bbox.max = mi.Vector2f( quadTreeNode.bbox.max )
		self.depth = mi.UInt32( quadTreeNode.depth )
		self.irradiance = mi.Float( quadTreeNode.irradiance )
		self.isLeaf = mi.Bool( quadTreeNode.isLeaf )
		self.refinementThreshold = mi.Float( quadTreeNode.refinementThreshold )
		self.child_1_index = mi.UInt32( quadTreeNode.child_1_index )
		self.child_2_index = mi.UInt32( quadTreeNode.child_2_index )
		self.child_3_index = mi.UInt32( quadTreeNode.child_3_index )
		self.child_4_index = mi.UInt32( quadTreeNode.child_4_index )

	
	def loadFromFile( self, dataNumpy: np.array ) -> None:
		"""
			Load data from numpy array-like object
		"""
		self.rootNodeIndex = mi.UInt32( dataNumpy['quadtree_rootNodeIndex'] )
		self.bbox = mi.BoundingBox2f( dataNumpy['quadtree_bbox_min'], dataNumpy['quadtree_bbox_max'] )
		self.depth = mi.UInt32( dataNumpy['quadtree_depth'] )
		self.irradiance = mi.Float( dataNumpy['quadtree_irradiance'] )
		self.isLeaf = mi.Bool( dataNumpy['quadtree_isLeaf'] )
		self.refinementThreshold = mi.Float( dataNumpy['quadtree_refinementThreshold'] )
		self.child_1_index = mi.UInt32( dataNumpy['quadtree_child_1_index'] )
		self.child_2_index = mi.UInt32( dataNumpy['quadtree_child_2_index'] )
		self.child_3_index = mi.UInt32( dataNumpy['quadtree_child_3_index'] )
		self.child_4_index = mi.UInt32( dataNumpy['quadtree_child_4_index'] )


	def getWidth(self) -> int:
		return dr.width(self.depth)
	

	def getBBox(self, idx: mi.UInt32) -> mi.BoundingBox2f:
		"""
			Get mitsuba.BoundingBox2f for the given node index
		"""
		bbox_min = dr.gather(mi.Vector2f, self.bbox.min, idx)
		bbox_max = dr.gather(mi.Vector2f, self.bbox.max, idx)
		bbox = mi.BoundingBox2f(bbox_min, bbox_max)
		return bbox


	def addIrradiance(self, idx: mi.UInt32, irradiance: mi.Float, active: mi.Bool) -> None:
		"""
			Add Irradiance and add into the node index.
			Should be use only on the leaf node.
		"""
		dr.scatter_reduce( dr.ReduceOp.Add, self.irradiance, irradiance, idx, active )


	def split(self, idx: mi.UInt32) -> None:
		"""
			- idx: node index to spit. Need to guaranteed that there is no duplicating index.
		"""

		# 	Allocate new memory 
		numRefineNode = dr.width(idx)
		numNewNodes = numRefineNode * 4
		oldSize = self.getWidth()
		newSize = oldSize + numNewNodes
		self.resize( newSize )

		# 	Calculate new children index
		refineNodeIndex = dr.arange(mi.UInt32, numRefineNode)
		child_1_index = refineNodeIndex * 4 + 0 + oldSize
		child_2_index = refineNodeIndex * 4 + 1 + oldSize
		child_3_index = refineNodeIndex * 4 + 2 + oldSize
		child_4_index = refineNodeIndex * 4 + 3 + oldSize

		# 	Update current node's new children index
		dr.scatter( self.child_1_index, child_1_index, idx )
		dr.scatter( self.child_2_index, child_2_index, idx )
		dr.scatter( self.child_3_index, child_3_index, idx )
		dr.scatter( self.child_4_index, child_4_index, idx )

		# 	Update current node's isLeaf
		dr.scatter( self.isLeaf, False, idx )

		# 	Update children's depth
		depth = dr.gather( mi.UInt32, self.depth, idx )
		depth += 1
		dr.scatter( self.depth, depth, child_1_index )
		dr.scatter( self.depth, depth, child_2_index )
		dr.scatter( self.depth, depth, child_3_index )
		dr.scatter( self.depth, depth, child_4_index )
		
		# 	Update children's irradiance
		irradiance = dr.gather( mi.Float, self.irradiance, idx )
		irradiance /= 4		#	Split the irradiance equally
		dr.scatter( self.irradiance, irradiance, child_1_index )
		dr.scatter( self.irradiance, irradiance, child_2_index )
		dr.scatter( self.irradiance, irradiance, child_3_index )
		dr.scatter( self.irradiance, irradiance, child_4_index )

		# 	Update children's refinementThreshold
		refinementThreshold = dr.gather( mi.Float, self.refinementThreshold, idx )
		dr.scatter( self.refinementThreshold, refinementThreshold, child_1_index )
		dr.scatter( self.refinementThreshold, refinementThreshold, child_2_index )
		dr.scatter( self.refinementThreshold, refinementThreshold, child_3_index )
		dr.scatter( self.refinementThreshold, refinementThreshold, child_4_index )

		# 	Update children's bbox
		# 		Bounding Box splitting
		bbox_min = dr.gather(mi.Vector2f, self.bbox.min, idx)
		bbox_max = dr.gather(mi.Vector2f, self.bbox.max, idx)
		bbox_mid = (bbox_min + bbox_max) / 2

		# 		Quadrant 1
		bbox_q1_min = mi.Vector2f( bbox_mid )
		bbox_q1_max = mi.Vector2f( bbox_max )
		bbox_q1 = mi.BoundingBox2f(bbox_q1_min, bbox_q1_max)

		# 		Quadrant 2
		bbox_q2_min = mi.Vector2f( bbox_min )
		bbox_q2_min.y = bbox_mid.y
		bbox_q2_max = mi.Vector2f( bbox_max )
		bbox_q2_max.x = bbox_mid.x
		bbox_q2 = mi.BoundingBox2f(bbox_q2_min, bbox_q2_max)

		# 		Quadrant 3
		bbox_q3_min = mi.Vector2f( bbox_min )
		bbox_q3_max = mi.Vector2f( bbox_mid )
		bbox_q3 = mi.BoundingBox2f(bbox_q3_min, bbox_q3_max)

		# 		Quadrant 4
		bbox_q4_min = mi.Vector2f( bbox_min )
		bbox_q4_min.x = bbox_mid.x
		bbox_q4_max = mi.Vector2f( bbox_max )
		bbox_q4_max.y = bbox_mid.y
		bbox_q4 = mi.BoundingBox2f(bbox_q4_min, bbox_q4_max)

		# 	Save children bounding box
		dr.scatter( self.bbox.min, bbox_q1.min, child_1_index )
		dr.scatter( self.bbox.max, bbox_q1.max, child_1_index )

		dr.scatter( self.bbox.min, bbox_q2.min, child_2_index )
		dr.scatter( self.bbox.max, bbox_q2.max, child_2_index )

		dr.scatter( self.bbox.min, bbox_q3.min, child_3_index )
		dr.scatter( self.bbox.max, bbox_q3.max, child_3_index )

		dr.scatter( self.bbox.min, bbox_q4.min, child_4_index )
		dr.scatter( self.bbox.max, bbox_q4.max, child_4_index )

		# Update the tree new state. Otherwise this could causes error in loop.
		dr.eval(self.irradiance, self.bbox.min, self.bbox.max)

	
	def merge(self, idx: mi.UInt32) -> None:
		"""
			Merging children node of the given parent node into the parent's itself.
			In merging children means we combine children's value into parent's and the delete children node.
			And since the parent value is already the sum of its children value, we can simply just delete
			the children node.
			- idx: Parent node index.
		"""
		# Return if passed in empty array
		if dr.width(idx) == 0:
			return
			
		# Set children index = 0
		dr.scatter( self.child_1_index, 0, idx )
		dr.scatter( self.child_2_index, 0, idx )
		dr.scatter( self.child_3_index, 0, idx )
		dr.scatter( self.child_4_index, 0, idx )

		# Set isLeaf to True
		dr.scatter( self.isLeaf, True, idx )

	
	def resize(self, newSize: int) -> None:
		"""
			Resize the entire data structure except for the rootIndex array.
			If the new size is less than current then cut the tail. Careful, this might corrupt tree structure.
			If the new size is more than current the concat tail with dr.zeros. Except for isLeaf.
		"""
		self.depth = resizeDrJitArray( self.depth, newSize )
		self.irradiance = resizeDrJitArray( self.irradiance, newSize )
		self.isLeaf = resizeDrJitArray( self.isLeaf, newSize, isDefaultZero= False )
		self.refinementThreshold = resizeDrJitArray( self.refinementThreshold, newSize )
		self.child_1_index = resizeDrJitArray( self.child_1_index, newSize )
		self.child_2_index = resizeDrJitArray( self.child_2_index, newSize )
		self.child_3_index = resizeDrJitArray( self.child_3_index, newSize )
		self.child_4_index = resizeDrJitArray( self.child_4_index, newSize )

		bbox_min = resizeDrJitArray(self.bbox.min, newSize)
		bbox_max = resizeDrJitArray(self.bbox.max, newSize)
		self.bbox = mi.BoundingBox2f( bbox_min, bbox_max )

		# Need to evaluate after resizing all the arrays to ensure consistency, otherwise this will causes bug
		dr.eval(
			self.depth,
			self.irradiance,
			self.isLeaf,
			self.refinementThreshold,
			self.child_1_index,
			self.child_2_index,
			self.child_3_index,
			self.child_4_index,
			self.bbox.min,
			self.bbox.max
		)		


	def createRootNode(self, numRootNode: int) -> mi.UInt32:
		"""
			Create new root nodes and return the new root node indicies.
			For example, if the current rootNodeIndex = [0], create new roots with numRootNode = 2
			will result as rootNodeIndex = [0, 401, 402] where 401 and 402 are indicies to the root node in
			the array and this function will return [1, 2] which are the indicies to rootNodeIndex array.
		"""
		# Allocate memory
		# 	Root Index
		oldRootSize = dr.width( self.rootNodeIndex )
		newRootSize = oldRootSize + numRootNode
		self.rootNodeIndex = resizeDrJitArray(self.rootNodeIndex, newRootSize )

		# 	Root properties
		oldSize = self.getWidth()
		newSize = oldSize + numRootNode
		self.resize( newSize )

		# Set value
		# 	Root index
		newRootNodeIndex = dr.arange(mi.UInt32, numRootNode) + oldSize
		newRootIndex = dr.arange(mi.UInt32, numRootNode) + oldRootSize
		dr.scatter( self.rootNodeIndex, newRootNodeIndex, newRootIndex )

		# 	Root properties
		# 		Bounding Box
		bbox_min = dr.zeros( mi.Vector2f, shape= numRootNode )
		bbox_max = dr.ones( mi.Vector2f, shape= numRootNode )
		dr.scatter( self.bbox.min, bbox_min, newRootNodeIndex )
		dr.scatter( self.bbox.max, bbox_max, newRootNodeIndex )

		# 		IsLeaf
		isLeaf = dr.ones( mi.Bool, shape= numRootNode )
		dr.scatter( self.isLeaf, isLeaf, newRootNodeIndex )

		return newRootIndex

	
	def getAllLeafNodeIndex(self, rootIndex: mi.UInt32 = None) -> mi.UInt32:
		"""
			Get all leaf node indicies from the given root node index.
			If no rootIndex is provided then simply return all the leaf node of every tree.
			- rootIndex: index of tree root (0, 1, 2, ...) to start traversing from.
		"""

		if type(rootIndex) == None or dr.width(rootIndex) == 0:
			return dr.compress( self.isLeaf )

		else:
			# Traverse from root to leaf
			nodeIndex = dr.gather( mi.UInt32, self.rootNodeIndex, rootIndex )

			# Array to store leaf node
			allLeafNode = mi.UInt32()
			
			# Iteratively traverse through the depth until no more root node to travel
			active = True
			while active:

				# Leaf node
				# 	Get leaf node index
				isLeaf = dr.gather( mi.Bool, self.isLeaf, nodeIndex )
				leafNode = dr.gather(mi.UInt32, nodeIndex, dr.compress( isLeaf ))

				# 	Save into output array
				oldAllLeafNodeSize = dr.width(allLeafNode)
				leafNodeSize = dr.width(leafNode)
				if leafNodeSize > 0:
					newSize = oldAllLeafNodeSize + leafNodeSize
					allLeafNode = resizeDrJitArray( allLeafNode, newSize )
					scatterLeafNodeIndex = dr.arange( mi.UInt32, leafNodeSize ) + oldAllLeafNodeSize
					dr.scatter( allLeafNode, leafNode, index= scatterLeafNodeIndex )


				# Non-leaf node
				# 	Get non-leaf node index
				isNotLeaf = ~isLeaf
				nonLeafNode = dr.gather( mi.UInt32, nodeIndex, dr.compress( isNotLeaf ) )

				# 	Get children index
				child_1_index = dr.gather(mi.UInt32, self.child_1_index, nonLeafNode)
				child_2_index = dr.gather(mi.UInt32, self.child_2_index, nonLeafNode)
				child_3_index = dr.gather(mi.UInt32, self.child_3_index, nonLeafNode)
				child_4_index = dr.gather(mi.UInt32, self.child_4_index, nonLeafNode)

				# 	Copy children index as nodeIndex for the next iteration
				nodeIndex = concatDrJitArray(
					concatDrJitArray( child_1_index, child_2_index ),
					concatDrJitArray( child_3_index, child_4_index )
				)

				# Update condition. Only continue if there is atleast one node
				active = dr.width(nodeIndex) > 0


			return allLeafNode


class QuadTree:

	def __init__(self, maxDepth: int = 20, isStoreNEERadiance: bool = False) -> None:
		"""
			- maxDepth: Maximum depth of the QuadTree.
		"""
		# Init QuadTreeNode
		self.quadTreeNode: QuadTreeNode = dr.zeros( QuadTreeNode, shape= 1 )
		self.quadTreeNode.rootNodeIndex = mi.UInt32(0)	# start with one root node
		self.quadTreeNode.refinementThreshold = mi.Float( float('inf') )
		self.quadTreeNode.isLeaf = mi.Bool(True)
		self.quadTreeNode.bbox = mi.BoundingBox2f([0, 0], [1, 1])

		self.maxDepth = maxDepth
		self.isStoreNEERadiance = isStoreNEERadiance
		

	def loadFromFile( self, dataNumpy: np.array ) -> None:
		"""
			Load data from numpy array-like object
		"""
		# Load QuadTree
		self.maxDepth = int(dataNumpy['quadtree_maxDepth'])
		self.isStoreNEERadiance = bool(dataNumpy['quadtree_isStoreNEERadiance'])
		
		# Load QuadTreeNode
		self.quadTreeNode = QuadTreeNode()
		self.quadTreeNode.loadFromFile( dataNumpy )



	def createRootNode(self, numRootNode: int) -> mi.UInt32:
		"""
			Create new root nodes and return the new root node indicies.
			For example, if the current rootNodeIndex = [0], create new roots with numRootNode = 2
			will result as rootNodeIndex = [0, 401, 402] where 401 and 402 are indicies to the root node in
			the array and this function will return [1, 2] which are the indicies to rootNodeIndex array.
		"""
		return self.quadTreeNode.createRootNode(numRootNode)


	def addDataPropagate(self, rootIndex: mi.UInt32, surfaceInteractionRecord: SurfaceInteractionRecord) -> None:
		"""
			Traverse the tree and add data along the way until reach the corresponding leaf node.
			Both rootIndex and surfaceInteractionRecord must have the same size!
			- rootIndex: index of tree root (0, 1, 2, ...).
			- surfaceInteractionRecord: contains interaction data on the surface.
		"""

		
		def addIrradiancePropagate( position: mi.Vector2f, irradiance: mi.Float ):

			# 	Start with searching index at root node
			nodeIndex = dr.gather( mi.UInt32, self.quadTreeNode.rootNodeIndex, rootIndex )

			# Test if data is within the root node bbox
			rootNodeBBox = self.quadTreeNode.getBBox( nodeIndex )
			active = rootNodeBBox.contains( position )

			loop = mi.Loop(name= 'QuadTree propagate data to leaf', state= lambda: (active, nodeIndex) )
			while loop(active):

				# Add irradiance into the node
				self.quadTreeNode.addIrradiance( nodeIndex, irradiance, active )

				# If at leaf node then stop, if not then continue
				isLeafNode = dr.gather(mi.Bool, self.quadTreeNode.isLeaf, nodeIndex, active )
				isNotLeafNode = ~isLeafNode		# Bit flip to active on the non-leaf node
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
			
			# May not needed
			dr.eval(self.quadTreeNode.irradiance)


		# 
		# Add path-irradiance
		# 

		# Position
		position = surfaceInteractionRecord.direction
		# Irradiance
		irradiance = dr.select( surfaceInteractionRecord.woPdf > 0, surfaceInteractionRecord.radiance / surfaceInteractionRecord.woPdf, 0 )

		addIrradiancePropagate( position, irradiance )

		if( self.isStoreNEERadiance ):
			# 
			# Add NEE-irradiance
			# 
			position = surfaceInteractionRecord.direction_nee
			# Irradiance
			radiance_nee = mi.luminance( surfaceInteractionRecord.radiance_nee )
			irradiance_nee = dr.select( surfaceInteractionRecord.woPdf > 0, radiance_nee / surfaceInteractionRecord.woPdf, 0 )

			addIrradiancePropagate( position, irradiance_nee )
		


	def validateQuadTreeNodeBBox(self, quadTreeNode: QuadTreeNode) -> bool:
		# 	Start with searching index at root node
		nodeIndex = dr.arange(mi.UInt32, dr.width(quadTreeNode.isLeaf))

		# If at leaf node then stop, if not then continue
		isNotLeafNode = ~quadTreeNode.isLeaf		# Bit flip to active on the non-leaf node
		active = isNotLeafNode 

		# Else, check which child node it belongs to and store the node index
		child_1_idx = dr.gather(mi.UInt32, quadTreeNode.child_1_index, index= nodeIndex, active= active)
		child_2_idx = dr.gather(mi.UInt32, quadTreeNode.child_2_index, index= nodeIndex, active= active)
		child_3_idx = dr.gather(mi.UInt32, quadTreeNode.child_3_index, index= nodeIndex, active= active)
		child_4_idx = dr.gather(mi.UInt32, quadTreeNode.child_4_index, index= nodeIndex, active= active)

		node_bbox = quadTreeNode.getBBox( nodeIndex )

		child_1_bbox = quadTreeNode.getBBox( child_1_idx )
		child_2_bbox = quadTreeNode.getBBox( child_2_idx )
		child_3_bbox = quadTreeNode.getBBox( child_3_idx )
		child_4_bbox = quadTreeNode.getBBox( child_4_idx )

		valid_child_1 = ((child_1_bbox.min.x >= node_bbox.min.x) & (child_1_bbox.max.x <= node_bbox.max.x) & (child_1_bbox.min.y >= node_bbox.min.y) & (child_1_bbox.max.y <= node_bbox.max.y))
		valid_child_2 = ((child_2_bbox.min.x >= node_bbox.min.x) & (child_2_bbox.max.x <= node_bbox.max.x) & (child_2_bbox.min.y >= node_bbox.min.y) & (child_2_bbox.max.y <= node_bbox.max.y))
		valid_child_3 = ((child_3_bbox.min.x >= node_bbox.min.x) & (child_3_bbox.max.x <= node_bbox.max.x) & (child_3_bbox.min.y >= node_bbox.min.y) & (child_3_bbox.max.y <= node_bbox.max.y))
		valid_child_4 = ((child_4_bbox.min.x >= node_bbox.min.x) & (child_4_bbox.max.x <= node_bbox.max.x) & (child_4_bbox.min.y >= node_bbox.min.y) & (child_4_bbox.max.y <= node_bbox.max.y))

		none_bbox_test_fail = active & (~valid_child_1 | ~valid_child_2 | ~valid_child_3 | ~valid_child_4)

		dr.printf_async("failed bbox test n:(%f %f)-(%f %f) c1:(%f %f)-(%f %f) c2:(%f %f)-(%f %f) c3:(%f %f)-(%f %f) c4:(%f %f)-(%f %f)\n",
			node_bbox.min.x, node_bbox.min.y, node_bbox.max.x, node_bbox.max.y,
			child_1_bbox.min.x, child_1_bbox.min.y, child_1_bbox.max.x, child_1_bbox.max.y,
			child_2_bbox.min.x, child_2_bbox.min.y, child_2_bbox.max.x, child_2_bbox.max.y,
			child_3_bbox.min.x, child_3_bbox.min.y, child_3_bbox.max.x, child_3_bbox.max.y,
			child_4_bbox.min.x, child_4_bbox.min.y, child_4_bbox.max.x, child_4_bbox.max.y,
			active=none_bbox_test_fail)

		dr.eval()

		if dr.width( dr.compress( none_bbox_test_fail ) ) > 0:
			return False
		else:
			return True


	def setRefinementThreshold(self, rootIndex: mi.UInt32, total_flux_prev_quadtree: mi.Float) -> None:
		"""
			Calculate and traverse the tree from root to set the refinement threshold.
			Flux in leaf node should contain no more than 1% of the total flux.
			- rootIndex: index of tree root (0, 1, 2, ...).
			- total_flux_prev_quadtree: total flux from previous quadtree.
		"""
		refinementThreshold = total_flux_prev_quadtree / 100
		
		# Start from the root node
		nodeIndex = dr.gather( mi.UInt32, self.quadTreeNode.rootNodeIndex, rootIndex )

		active = dr.width( nodeIndex ) > 0

		while active:

			# Set node refinement thresolhd
			dr.scatter( self.quadTreeNode.refinementThreshold, refinementThreshold, nodeIndex )

			# Check if leaf node, if yes then stop
			isLeafNode = dr.gather( mi.Bool, self.quadTreeNode.isLeaf, nodeIndex )
			isNotLeafNode = ~isLeafNode		# Bit flip to active on the non-leaf node

			# If there is at least one non-leaf node
			active = dr.any( isNotLeafNode )
			if active:

				# Get leaf node index
				isNotLeafNodeIndex = dr.compress( isNotLeafNode )
				nonLeafNodeIndex = dr.gather( mi.UInt32, nodeIndex, isNotLeafNodeIndex )
				nonLeafNodeRefinementThreshold = dr.gather( mi.Float, refinementThreshold, isNotLeafNodeIndex )

				# Get children index
				child_1_idx = dr.gather( mi.UInt32, self.quadTreeNode.child_1_index, nonLeafNodeIndex )
				child_2_idx = dr.gather( mi.UInt32, self.quadTreeNode.child_2_index, nonLeafNodeIndex )
				child_3_idx = dr.gather( mi.UInt32, self.quadTreeNode.child_3_index, nonLeafNodeIndex )
				child_4_idx = dr.gather( mi.UInt32, self.quadTreeNode.child_4_index, nonLeafNodeIndex )

				# Set all children index to be travel in the new iteration
				nodeIndex = concatDrJitArray(
					concatDrJitArray( child_1_idx, child_2_idx ),
					concatDrJitArray( child_3_idx, child_4_idx )
				)

				# Set children refinement threshold reference for the next iteration
				refinementThreshold = concatDrJitArray(
					concatDrJitArray( nonLeafNodeRefinementThreshold, nonLeafNodeRefinementThreshold ),
					concatDrJitArray( nonLeafNodeRefinementThreshold, nonLeafNodeRefinementThreshold )
				)

	
	def refine(self, rootIndex: mi.UInt32) -> None:
		""" Refine the tree until conditions are met.
			1. Merge if sum of children node is less than threshold.
			2. Split if the node's flux is more than the threshold and node's depth is less than maximum depth.
			- rootIndex: index of tree root (0, 1, 2, ...).
		"""
		# 
		# 	Merge condition
		# 
		# Traverse the tree and merge small nodes until reach leaf node
		# Start from the root node
		parentNodeIndex = dr.gather( mi.UInt32, self.quadTreeNode.rootNodeIndex, rootIndex )

		active = dr.width( parentNodeIndex ) > 0
		while active:

			# Check condition
			# 	Is not leaf node
			isLeafNode = dr.gather(mi.Bool, self.quadTreeNode.isLeaf, parentNodeIndex)
			isNotLeafNode = ~isLeafNode
			# 	Is irradiance less than threshold
			irradiance = dr.gather(mi.Float, self.quadTreeNode.irradiance, parentNodeIndex)
			refinementThreshold = dr.gather( mi.Float, self.quadTreeNode.refinementThreshold, parentNodeIndex )
			smallParentCondition = isNotLeafNode & ( irradiance < refinementThreshold )

			# Gather all node that passed the condition
			smallParentNodeIndex = dr.gather(mi.UInt32, parentNodeIndex, dr.compress(smallParentCondition) )

			# Merge children of the small parent node
			self.quadTreeNode.merge( smallParentNodeIndex )

			# Gather valid parent node index for the new iteration
			normalParentCondition = isNotLeafNode & ( irradiance >= refinementThreshold )
			validParentNodeIndex = dr.gather(mi.UInt32, parentNodeIndex, dr.compress(normalParentCondition) )

			# Gather children's index
			child_1_index = dr.gather(mi.UInt32, self.quadTreeNode.child_1_index, validParentNodeIndex)
			child_2_index = dr.gather(mi.UInt32, self.quadTreeNode.child_2_index, validParentNodeIndex)
			child_3_index = dr.gather(mi.UInt32, self.quadTreeNode.child_3_index, validParentNodeIndex)
			child_4_index = dr.gather(mi.UInt32, self.quadTreeNode.child_4_index, validParentNodeIndex)

			# Set children index to be parent of the new iteration
			parentNodeIndex = concatDrJitArray(
				concatDrJitArray(child_1_index, child_2_index),
				concatDrJitArray(child_3_index, child_4_index)
			)

			# Continue if there is atleast one node
			active = dr.width(parentNodeIndex) > 0


		# 
		# 	Split condition
		# 
		active = True
		while active:
			# Get all leaf node
			leafNodeIndex = self.getAllLeafNodeIndex( rootIndex )

			# If leaf node irradiance is more threshold and less than maximum depth then split
			irradiance = dr.gather(mi.Float, self.quadTreeNode.irradiance, leafNodeIndex)
			refinementThreshold = dr.gather(mi.Float, self.quadTreeNode.refinementThreshold, leafNodeIndex)
			depth = dr.gather( mi.UInt32, self.quadTreeNode.depth, leafNodeIndex )
			condition = ( irradiance > refinementThreshold ) & ( depth < self.maxDepth )

			# If there is at least one node to split. This is to avoid dr.gather() with active are all false 
			# which will result return in array size = src size
			active = dr.any( condition )
			if active:

				# Get list of leaf node that need refine
				splitNodeIndex = dr.gather( mi.UInt32, leafNodeIndex, dr.compress( condition ) )

				# Split 
				self.quadTreeNode.split( splitNodeIndex )

	
	def resetTreeIrradiance(self, rootIndex: mi.UInt32) -> None:
		"""
			Traverse the tree from root to leaf and set all irradiance to zero.
			- rootIndex: index of tree root (0, 1, 2, ...).
		"""
		
		# Start from the root node
		nodeIndex = dr.gather( mi.UInt32, self.quadTreeNode.rootNodeIndex, rootIndex )

		active = dr.width( nodeIndex ) > 0

		while active:

			# Set node irradiance to zero
			dr.scatter( self.quadTreeNode.irradiance, 0, nodeIndex )

			# Check if leaf node, if yes then stop
			isLeafNode = dr.gather( mi.Bool, self.quadTreeNode.isLeaf, nodeIndex )
			isNotLeafNode = ~isLeafNode		# Bit flip to active on the non-leaf node

			# If there is at least one non-leaf node
			active = dr.any( isNotLeafNode )
			if active:

				# Get leaf node index
				nonLeafNodeIndex = dr.gather( mi.UInt32, nodeIndex, dr.compress( isNotLeafNode ) )

				# Get children index
				child_1_idx = dr.gather( mi.UInt32, self.quadTreeNode.child_1_index, nonLeafNodeIndex )
				child_2_idx = dr.gather( mi.UInt32, self.quadTreeNode.child_2_index, nonLeafNodeIndex )
				child_3_idx = dr.gather( mi.UInt32, self.quadTreeNode.child_3_index, nonLeafNodeIndex )
				child_4_idx = dr.gather( mi.UInt32, self.quadTreeNode.child_4_index, nonLeafNodeIndex )
				childrenIndex_part1 = concatDrJitArray( child_1_idx, child_2_idx )
				childrenIndex_part2 = concatDrJitArray( child_3_idx, child_4_idx )

				# Set all children index to be travel in the new iteration
				nodeIndex = concatDrJitArray( childrenIndex_part1, childrenIndex_part2 )

	
	def resetAllTreeIrradiance(self) -> None:
		"""
			Reset every tree irradiance from root node to leaf
		"""
		self.resetTreeIrradiance( self.quadTreeNode.rootNodeIndex )


	def getAllLeafNodeIndex(self, rootIndex: mi.UInt32 = None) -> mi.UInt32:
		"""
			Get all leaf node indicies from the given root node index.
			If no rootIndex is provided then simply return all the leaf node of every tree.
			- rootIndex: index of tree root (0, 1, 2, ...) to start traversing from.
		"""
		return self.quadTreeNode.getAllLeafNodeIndex( rootIndex )

	
	def copyTree(self, rootIndex: mi.UInt32) -> QuadTreeNode:
		"""
			Traverse the tree from root node, copy data with calculation of new children index
			and return the new QuadtreeNode_CUDA class.
			- rootIndex: index of tree root (0, 1, 2, ...) to start traversing from.
		"""
		# 
		# Create a new "Empty" QuadTree node to copy into
		# 
		quadTreeNode: QuadTreeNode = dr.zeros( QuadTreeNode, shape= 0 )
		# 	Set new root node index.
		# 		Because we're visiting all the root node first, it is guaranteed that 
		# 		the root node index (quadTreeNode.rootNodeIndex) will be [0, 1, ..., numRoots-1]
		numRoots = dr.width( rootIndex )
		quadTreeNode.rootNodeIndex = dr.arange( mi.UInt32, numRoots )
		
		# 	BoundingBox2f is a special variable that has to declare explicitly
		quadTreeNode.bbox = mi.BoundingBox2f([0, 0], [1, 1])

		# Loop variables
		# 	Current node index. Start traversing the tree from root node
		nodeIndex = dr.gather( mi.UInt32, self.quadTreeNode.rootNodeIndex, rootIndex )

		# 	Current node's parent index. Use inconjuction with nodeIndex and parentChildIndex.
		# 	nodeIndex, parentIndex, and parentChildIndex must all have the same size except when starting.
		# 	Since we're starting from root, there is parent from root so we just initialized an empty array.
		parentIndex = mi.UInt32()

		# 	Current node's parent's child index.
		# 	Each parent node can have four children and this is the value indicating which child_i_index
		# 	the current node came from. An element of this array can have value of [1, 2, 3, 4] indicating
		# 	child_1_index, child_2_index, child_3_index, and child_4_index respectively.
		parentChildIndex = mi.UInt32()


		# Starting variable
		numNodes = dr.width(nodeIndex)
		active = numNodes > 0
		while active:

			# Allocate new memory
			numNodes = dr.width(nodeIndex)
			oldSize = quadTreeNode.getWidth()
			newSize = oldSize + numNodes
			quadTreeNode.resize(newSize)

			# Create new node index
			newNodeIndex = dr.arange( mi.UInt32, numNodes ) + oldSize

			# Copy new node index to parent's children index
			if dr.width( parentIndex ) > 0:

				active_parent_1 = dr.eq( parentChildIndex, 1 )
				active_parent_2 = dr.eq( parentChildIndex, 2 )
				active_parent_3 = dr.eq( parentChildIndex, 3 )
				active_parent_4 = dr.eq( parentChildIndex, 4 )

				dr.scatter( quadTreeNode.child_1_index, newNodeIndex, parentIndex, active= active_parent_1 )
				dr.scatter( quadTreeNode.child_2_index, newNodeIndex, parentIndex, active= active_parent_2 )
				dr.scatter( quadTreeNode.child_3_index, newNodeIndex, parentIndex, active= active_parent_3 )
				dr.scatter( quadTreeNode.child_4_index, newNodeIndex, parentIndex, active= active_parent_4 )


			# Copy node value
			# 	Bounding box
			bbox_min = dr.gather( mi.Vector2f, self.quadTreeNode.bbox.min, nodeIndex )
			dr.scatter( quadTreeNode.bbox.min, bbox_min, newNodeIndex )
			bbox_max = dr.gather( mi.Vector2f, self.quadTreeNode.bbox.max, nodeIndex )
			dr.scatter( quadTreeNode.bbox.max, bbox_max, newNodeIndex )

			# 	Depth
			depth = dr.gather( mi.UInt32, self.quadTreeNode.depth, nodeIndex )
			dr.scatter( quadTreeNode.depth, depth, newNodeIndex )

			# 	Irradiance
			irradiance = dr.gather( mi.Float, self.quadTreeNode.irradiance, nodeIndex )
			dr.scatter( quadTreeNode.irradiance, irradiance, newNodeIndex )

			# 	isLeaf
			isLeaf = dr.gather( mi.Bool, self.quadTreeNode.isLeaf, nodeIndex )
			dr.scatter( quadTreeNode.isLeaf, isLeaf, newNodeIndex )

			# 	RefinementThreshold
			refinementThreshold = dr.gather( mi.Float, self.quadTreeNode.refinementThreshold, nodeIndex )
			dr.scatter( quadTreeNode.refinementThreshold, refinementThreshold, newNodeIndex )


			# Continue if there is atleast one non leaf node
			isNotLeaf = ~isLeaf
			nonLeafNodeIndex = dr.gather( mi.UInt32, nodeIndex, dr.compress( isNotLeaf ) )
			newNonLeafNodeIndex = dr.gather( mi.UInt32, newNodeIndex, dr.compress( isNotLeaf ) )

			if dr.width( nonLeafNodeIndex ) > 0:

				# 	Child_1_index
				child_1_index = dr.gather( mi.UInt32, self.quadTreeNode.child_1_index, nonLeafNodeIndex )

				# 	Child_2_index
				child_2_index = dr.gather( mi.UInt32, self.quadTreeNode.child_2_index, nonLeafNodeIndex )

				# 	Child_3_index
				child_3_index = dr.gather( mi.UInt32, self.quadTreeNode.child_3_index, nonLeafNodeIndex )

				# 	Child_4_index
				child_4_index = dr.gather( mi.UInt32, self.quadTreeNode.child_4_index, nonLeafNodeIndex )

				# Copy current node index as next parent index
				nonLeafNodeSize = dr.width( nonLeafNodeIndex )
				parentIndex = dr.repeat( newNonLeafNodeIndex, 4 )

				# Copy current node's parent's child index [1, 2, 3, 4, 1, 2, 3, 4, ...]
				parentChildIndex = dr.tile( dr.arange( mi.UInt32, 1, 5 ), nonLeafNodeSize )

				# Set new node index
				# 	The child index will be set in an order of node then children, e.g.
				# 		node_1_child_1, node_1_child_2, node_1_child_3, node_1_child_4, 
				# 		node_2_child_1, node_2_child_2, node_2_child_3, node_2_child_4, ... 
				nodeIndex = dr.zeros( mi.UInt32, nonLeafNodeSize * 4 )
				scatterNodeIndex = dr.arange( mi.UInt32, nonLeafNodeSize ) * 4
				dr.scatter( nodeIndex, child_1_index, scatterNodeIndex + 0 )
				dr.scatter( nodeIndex, child_2_index, scatterNodeIndex + 1 )
				dr.scatter( nodeIndex, child_3_index, scatterNodeIndex + 2 )
				dr.scatter( nodeIndex, child_4_index, scatterNodeIndex + 3 )

				# Update loop condition
				active = True
						
			else:
				active = False


		dr.eval(quadTreeNode)
		
		return quadTreeNode

	
	def copyFrom(self, quadTree: QuadTree) -> None:
		"""
			Copy from a QuadTree. Unlike 'copyTree', this does not traverse and modify the tree.
			It copy every attributes identically.
		"""
		# Copy QuadTreeNode
		self.quadTreeNode.copyFrom( quadTree.quadTreeNode )

		# Copy self properties
		self.maxDepth = quadTree.maxDepth
		self.isStoreNEERadiance = quadTree.isStoreNEERadiance

	
	def clearTreeUnusedNode(self) -> None:
		"""
			Remove all the unused node that caused by node deletion/merging operation.
			Traverse the tree and copy only the relevent node into new array.
			And then set self to this new array.
		"""
		treeRootIndex = dr.arange( mi.UInt32, dr.width( self.quadTreeNode.rootNodeIndex ) )
		self.quadTreeNode = self.copyTree( rootIndex= treeRootIndex )


	def appendQuadTreeNode(self, quadTreeNode: QuadTreeNode) -> mi.UInt32:
		"""
			Append a QuadTreeNode at the back of the main self.quadTreeNode. 
			Effectively merging two QuadTree into one. It also modify the input QuadTreeNode
			indicies accordingly.

			For example, if the current tree have two roots (0, 142)
			and the input tree have one root (0) the merge result can be something like (0, 142, 238).
			The number '238' is a result from offseting by the current tree size. And the function
			will return (2) which is a reference index to the position in self.quadTreeNode.rootNodeIndex.

			WARNING: This function modifies the input quadTreeNode.
			
			Return: New offset rootNodeIndex. 

			- quadTreeNode: Input QuadTreeNode_CUDA object to merged with.
		"""
		
		# Allocate memory
		# 	RootNode Index
		oldRootSize = dr.width( self.quadTreeNode.rootNodeIndex )
		inputRootSize = dr.width( quadTreeNode.rootNodeIndex )
		newRootSize = oldRootSize + inputRootSize
		self.quadTreeNode.rootNodeIndex = resizeDrJitArray( self.quadTreeNode.rootNodeIndex, newRootSize )
		
		# 	Other tree's values
		oldSize = self.quadTreeNode.getWidth()
		inputSize = dr.width( quadTreeNode.depth )
		newSize = oldSize + inputSize
		self.quadTreeNode.resize( newSize )

		# Offset Input QuadTreeNode index by the old size
		offset = oldSize
		quadTreeNode.rootNodeIndex += offset
		isNotLeaf = ~quadTreeNode.isLeaf
		quadTreeNode.child_1_index[ isNotLeaf ] += offset
		quadTreeNode.child_2_index[ isNotLeaf ] += offset
		quadTreeNode.child_3_index[ isNotLeaf ] += offset
		quadTreeNode.child_4_index[ isNotLeaf ] += offset

		# Copy Input QuadTreeNode values into self.quadTreeNode
		# 	rootNodeIndex
		scatterRootNodeIndex = dr.arange( mi.UInt32, inputRootSize ) + oldRootSize
		dr.scatter( self.quadTreeNode.rootNodeIndex, quadTreeNode.rootNodeIndex, scatterRootNodeIndex )

		# 	bbox
		scatterValueIndex = dr.arange( mi.UInt32, inputSize ) + oldSize
		dr.scatter( self.quadTreeNode.bbox.min, quadTreeNode.bbox.min, scatterValueIndex )
		dr.scatter( self.quadTreeNode.bbox.max, quadTreeNode.bbox.max, scatterValueIndex )

		# 	depth
		dr.scatter( self.quadTreeNode.depth, quadTreeNode.depth, scatterValueIndex )

		# 	irradiance
		dr.scatter( self.quadTreeNode.irradiance, quadTreeNode.irradiance, scatterValueIndex )
		
		# 	isLeaf
		dr.scatter( self.quadTreeNode.isLeaf, quadTreeNode.isLeaf, scatterValueIndex )
		
		# 	refinementThreshold
		dr.scatter( self.quadTreeNode.refinementThreshold, quadTreeNode.refinementThreshold, scatterValueIndex )
		
		# 	child_1_index
		dr.scatter( self.quadTreeNode.child_1_index, quadTreeNode.child_1_index, scatterValueIndex )
		
		# 	child_2_index
		dr.scatter( self.quadTreeNode.child_2_index, quadTreeNode.child_2_index, scatterValueIndex )
		
		# 	child_3_index
		dr.scatter( self.quadTreeNode.child_3_index, quadTreeNode.child_3_index, scatterValueIndex )
		
		# 	child_4_index
		dr.scatter( self.quadTreeNode.child_4_index, quadTreeNode.child_4_index, scatterValueIndex )

		return scatterRootNodeIndex


	def sampleQuadTree( self, rootIndex: mi.UInt32, sampler: mi.Sampler, active_sample: mi.Bool = True ) -> mi.Vector3f:
		"""
			Sample direction/position from the quadtree.
			- rootIndex: index of tree root (0, 1, 2, ...) to start traversing from.
			- sampler: mitsuba.Sampler
		"""
		# Start from root node
		nodeIndex = dr.gather( mi.UInt32, self.quadTreeNode.rootNodeIndex, rootIndex )

		samplePosition = mi.Vector2f(0, 0)

		active_sample = mi.Bool(active_sample)

		loop = mi.Loop( 'Sample QuadTree', lambda: ( nodeIndex, active_sample, samplePosition, sampler ) )

		while loop(active_sample):

			isLeaf = dr.gather( mi.Bool, self.quadTreeNode.isLeaf, nodeIndex, active_sample )

			# 
			# If leaf, uniform random position in node
			# 
			bbox_min = dr.gather( mi.Vector2f, self.quadTreeNode.bbox.min, nodeIndex, active_sample )
			bbox_max = dr.gather( mi.Vector2f, self.quadTreeNode.bbox.max, nodeIndex, active_sample )

			samplePosition[ active_sample & isLeaf ] = bbox_min + sampler.next_2d( active_sample ) * ( bbox_max - bbox_min )

			# 
			# Else, sample a child by energy
			# 
			isNonLeaf = ~isLeaf
			active_sample &= isNonLeaf

			# Gather children energy
			child_1_index = dr.gather( mi.UInt32, self.quadTreeNode.child_1_index, nodeIndex, active_sample )
			child_2_index = dr.gather( mi.UInt32, self.quadTreeNode.child_2_index, nodeIndex, active_sample )
			child_3_index = dr.gather( mi.UInt32, self.quadTreeNode.child_3_index, nodeIndex, active_sample )
			child_4_index = dr.gather( mi.UInt32, self.quadTreeNode.child_4_index, nodeIndex, active_sample )
			child_1_irradiance = dr.gather( mi.Float, self.quadTreeNode.irradiance, child_1_index, active_sample )
			child_2_irradiance = dr.gather( mi.Float, self.quadTreeNode.irradiance, child_2_index, active_sample )
			child_3_irradiance = dr.gather( mi.Float, self.quadTreeNode.irradiance, child_3_index, active_sample )
			child_4_irradiance = dr.gather( mi.Float, self.quadTreeNode.irradiance, child_4_index, active_sample )

			# Construct a CDF
			child_2_irradiance += child_1_irradiance
			child_3_irradiance += child_2_irradiance
			child_4_irradiance += child_3_irradiance

			# Sampling a range
			sample_irradiance = sampler.next_1d() * child_4_irradiance

			# Check which bin it falls into
			sample_child_1_active = sample_irradiance < child_1_irradiance
			sample_child_2_active = (child_1_irradiance <= sample_irradiance) & (sample_irradiance < child_2_irradiance)
			sample_child_3_active = (child_2_irradiance <= sample_irradiance) & (sample_irradiance < child_3_irradiance)
			sample_child_4_active = child_3_irradiance <= sample_irradiance

			nodeIndex[ active_sample & sample_child_1_active ] = child_1_index
			nodeIndex[ active_sample & sample_child_2_active ] = child_2_index
			nodeIndex[ active_sample & sample_child_3_active ] = child_3_index
			nodeIndex[ active_sample & sample_child_4_active ] = child_4_index

			# Continue the loop
		
		# After finished sampling all the positions in QuadTree, convert them into direction vector
		sampleDirection = canonicalToDir( samplePosition )

		return sampleDirection

	
	def pdfQuadTree( self, rootIndex: mi.UInt32, direction: mi.Vector3f, active: mi.Bool = True ) -> mi.Float:
		"""
			Compute PDF of sampling direction from a QuadTree
			- rootIndex: index of tree root (0, 1, 2, ...) to start traversing from.
			- direction: sampling direction
		"""

		# Start from root node
		nodeIndex = dr.gather( mi.UInt32, self.quadTreeNode.rootNodeIndex, rootIndex )

		pdf = mi.Float(1)

		active_pdf = mi.Bool(active)

		# Searching in canonical position (2D) instead of direction (3D)
		position = dirToCanonical( direction )

		loop = mi.Loop( 'PDF QuadTree', lambda: ( nodeIndex, pdf, active_pdf ) )

		while loop( active_pdf ):
			
			# 
			# If leaf then return with 1/4Pi
			# 
			isLeaf = dr.gather( mi.Bool, self.quadTreeNode.isLeaf, nodeIndex, active_pdf )

			# If irradiance is 0 then return 0

			# This is a PDF of sampling a point over a sphere surface
			pdf[ active_pdf & isLeaf ] *= dr.inv_four_pi

			# 
			# Else, traverse the corresponding child
			# 
			isNonLeaf = ~isLeaf
			active_pdf &= isNonLeaf
			child_1_index = dr.gather( mi.UInt32, self.quadTreeNode.child_1_index, nodeIndex, active_pdf )
			child_2_index = dr.gather( mi.UInt32, self.quadTreeNode.child_2_index, nodeIndex, active_pdf )
			child_3_index = dr.gather( mi.UInt32, self.quadTreeNode.child_3_index, nodeIndex, active_pdf )
			child_4_index = dr.gather( mi.UInt32, self.quadTreeNode.child_4_index, nodeIndex, active_pdf )

			# Bounding box test
			child_1_bbox = self.quadTreeNode.getBBox( child_1_index )
			child_1_bbox_test = child_1_bbox.contains( position )

			child_2_bbox = self.quadTreeNode.getBBox( child_2_index )
			child_2_bbox_test = child_2_bbox.contains( position )

			child_3_bbox = self.quadTreeNode.getBBox( child_3_index )
			child_3_bbox_test = child_3_bbox.contains( position )

			child_4_bbox = self.quadTreeNode.getBBox( child_4_index )
			child_4_bbox_test = child_4_bbox.contains( position )
			
			# Multiply weight by the ratio between child's and parent's irradiance
			nodeIrradiance = dr.gather( mi.Float, self.quadTreeNode.irradiance, nodeIndex, active_pdf )
			child_1_irradiance = dr.gather( mi.Float, self.quadTreeNode.irradiance, child_1_index, active_pdf & child_1_bbox_test )
			child_2_irradiance = dr.gather( mi.Float, self.quadTreeNode.irradiance, child_2_index, active_pdf & child_2_bbox_test )
			child_3_irradiance = dr.gather( mi.Float, self.quadTreeNode.irradiance, child_3_index, active_pdf & child_3_bbox_test )
			child_4_irradiance = dr.gather( mi.Float, self.quadTreeNode.irradiance, child_4_index, active_pdf & child_4_bbox_test )

			# 	Select irradiance of the correct child
			childIrradiance = dr.select( child_1_bbox_test, 
				child_1_irradiance,
				dr.select( child_2_bbox_test,
					child_2_irradiance,
					dr.select( child_3_bbox_test,
						child_3_irradiance,
						dr.select( child_4_bbox_test,
							child_4_irradiance,
							0
						)
					)
				)
			)

			# 	Compute PDF of the current iteration
			# 		I think the reason why there is 4 * child irradiance here is because we
			# 		want to normalize the irradiance and get irradiance per area.
			# 		Which, in this case, parent node always has four times the size of a children node
			# 		so need to divide parent node irradiance by 4
			# 		PDF = child_irradiance / ( node_irradiance / 4 )
			# 		or: PDF = 4 * child_irradiance / node_irradiance
			pdf[ active_pdf ] *= 4 * childIrradiance / nodeIrradiance

			# 	If encounter nan this means that both childIrradiance and nodeIrradiance are 0.
			# 	This should never happens because if parent is 0 then it should never have any children.
			# 	Also, from further investigation, the 3D position of this QuadTree (from KDTree) is in the air, not touching any surfaces.
			# 	TODO further investigate the causes.
			is_pdf_nan = dr.isnan( pdf[ active_pdf ] )
			pdf[ active_pdf & is_pdf_nan ] = 0
			active_pdf &= ~is_pdf_nan	#	stop further traversal

			# Set next interation traversal node
			nodeIndex[ active_pdf & child_1_bbox_test ] = child_1_index
			nodeIndex[ active_pdf & child_2_bbox_test ] = child_2_index
			nodeIndex[ active_pdf & child_3_bbox_test ] = child_3_index
			nodeIndex[ active_pdf & child_4_bbox_test ] = child_4_index


		return pdf


	
# Test tree
if __name__ == '__main__':

	from numpy.random import rand as np_rand
	import time		# Performance timer

	N = 1000000
	
	def generateRandomData():
		surfaceInteractionRecord = dr.zeros( SurfaceInteractionRecord, N)

		surfaceInteractionRecord.position = mi.Vector3f( np_rand(N, 3) )
		surfaceInteractionRecord.direction = mi.Vector2f( np_rand(N, 2) )
		surfaceInteractionRecord.radiance = mi.Float( np_rand(N) )
		surfaceInteractionRecord.product = mi.Float( np_rand(N) )
		surfaceInteractionRecord.woPdf = mi.Float( np_rand(N) )
		surfaceInteractionRecord.bsdfPdf = mi.Float( np_rand(N) )

		isDelta = np_rand(N)
		isDelta = np.array( list( map( lambda x: (x > 0.5), isDelta ) ) )
		surfaceInteractionRecord.isDelta = mi.Bool( isDelta )

		return surfaceInteractionRecord
	

	surfaceInteractionRecord = generateRandomData()

	# Start performance timer
	startTime = time.perf_counter()

	# Create the tree
	myTree = QuadTree()


	# 
	# Split the leaf node once for proper testing of copy and addData
	# 
	printTitle('Test Create Tree and Splitting')
	rootIndex = mi.UInt32(0)
	# 1st split
	leafNodeIndex = myTree.getAllLeafNodeIndex( rootIndex= rootIndex )
	myTree.quadTreeNode.split(leafNodeIndex);
	# 2nd split
	leafNodeIndex = myTree.getAllLeafNodeIndex( rootIndex= rootIndex )
	myTree.quadTreeNode.split(leafNodeIndex);
	# 3rd split
	leafNodeIndex = myTree.getAllLeafNodeIndex( rootIndex= rootIndex )
	myTree.quadTreeNode.split(leafNodeIndex);

	# Print tree structure
	printBoldUnderLine('tree size:', myTree.quadTreeNode.getWidth())
	printBoldUnderLine('Tree structure:')
	print('child_1_index:')
	print( myTree.quadTreeNode.child_1_index.numpy().tolist() )
	print('child_2_index:')
	print( myTree.quadTreeNode.child_2_index.numpy().tolist() )
	print('child_3_index:')
	print( myTree.quadTreeNode.child_3_index.numpy().tolist() )
	print('child_4_index:')
	print( myTree.quadTreeNode.child_4_index.numpy().tolist() )

	validateResult = myTree.validateQuadTreeNodeBBox( myTree.quadTreeNode )
	printBoldUnderLine('Validate myTree bounding box result:', validateResult)


	# 
	# Test copy tree
	# 
	printTitle('Test Copy Tree')

	quadTreeNode = myTree.copyTree( rootIndex= mi.UInt32(0) )

	# Print tree structure
	printBoldUnderLine('tree size:', quadTreeNode.getWidth())
	printBoldUnderLine('Copy tree structure:')
	print('child_1_index:')
	print( quadTreeNode.child_1_index.numpy().tolist() )
	print('child_2_index:')
	print( quadTreeNode.child_2_index.numpy().tolist() )
	print('child_3_index:')
	print( quadTreeNode.child_3_index.numpy().tolist() )
	print('child_4_index:')
	print( quadTreeNode.child_4_index.numpy().tolist() )
	
	irradianceArray = quadTreeNode.irradiance.numpy()
	print('Copy tree irradiance node: ', irradianceArray)
	printBoldUnderLine('Copy tree irradiance:', irradianceArray[0] )
	printBoldUnderLine('Copy tree size:', quadTreeNode.getWidth())

	validateResult = myTree.validateQuadTreeNodeBBox( quadTreeNode )
	printBoldUnderLine('Validate copy tree bounding box result:', validateResult)


	# 
	# Add data
	# 
	printTitle('Test Add Data')
	dataRootIndex = dr.full( mi.UInt32, 0, N )
	myTree.addDataPropagate( dataRootIndex, surfaceInteractionRecord)

	# Check tree irradiance
	print('Tree irradiance node:', myTree.quadTreeNode.irradiance.numpy())

	irradianceArray = myTree.quadTreeNode.irradiance.numpy()
	printBoldUnderLine('Tree irradiance by root:', irradianceArray[0] )

	leafNodeIndex = myTree.getAllLeafNodeIndex( rootIndex )
	leafNodeIrradiance = dr.gather(mi.Float, myTree.quadTreeNode.irradiance, leafNodeIndex)
	leafNodeIrradianceSum = np.sum(leafNodeIrradiance.numpy())
	printBoldUnderLine('Tree irradiance by leaf sum:', leafNodeIrradianceSum )

	irradiance = surfaceInteractionRecord.radiance / surfaceInteractionRecord.woPdf
	trueIrradianceSum = np.sum( irradiance.numpy() )
	printBoldUnderLine('True irradiance sum:', trueIrradianceSum)


	# 
	# Refine tree
	# 
	printTitle('Test Refine Tree')
	myTree.setRefinementThreshold( rootIndex= rootIndex , total_flux_prev_quadtree= mi.Float( trueIrradianceSum.tolist() ) )
	print('Refinement threshold:', myTree.quadTreeNode.refinementThreshold.numpy())
	
	# Refine
	myTree.refine( rootIndex )

	# Check tree irradiance
	printBoldUnderLine('Tree size after refine:', myTree.quadTreeNode.getWidth())
	printBoldUnderLine('Tree irradiance after refine:', myTree.quadTreeNode.irradiance.numpy())
	leafNodeIndex = myTree.getAllLeafNodeIndex( rootIndex )
	leafNodeIrradiance = dr.gather(mi.Float, myTree.quadTreeNode.irradiance, leafNodeIndex)
	leafNodeIrradianceSum = np.sum(leafNodeIrradiance.numpy())
	printBoldUnderLine('Tree irradiance sum after refine:', leafNodeIrradianceSum)

	# Print tree structure
	printBoldUnderLine('tree size:', myTree.quadTreeNode.getWidth())
	print('child_1_index:')
	print( myTree.quadTreeNode.child_1_index.numpy().tolist() )
	print('child_2_index:')
	print( myTree.quadTreeNode.child_2_index.numpy().tolist() )
	print('child_3_index:')
	print( myTree.quadTreeNode.child_3_index.numpy().tolist() )
	print('child_4_index:')
	print( myTree.quadTreeNode.child_4_index.numpy().tolist() )

	validateResult = myTree.validateQuadTreeNodeBBox( myTree.quadTreeNode )
	printBoldUnderLine('Validate myTree bounding box result:', validateResult)
	

	# 
	# Test sample tree
	# 
	printTitle('Test Sampling direction from QuadTree')

	sampler = mi.load_dict({
		'type' : 'independent'
	})
	sampler.seed(0, wavefront_size= 10)
	
	rootIndex = dr.full( mi.UInt32, 0, 10 )
	sampleDir = myTree.sampleQuadTree( rootIndex, sampler )

	printBoldUnderLine( '10 Sampled directions:', sampleDir.numpy() )

	pdf = myTree.pdfQuadTree( rootIndex, sampleDir )

	printBoldUnderLine( '10 Sampled direction pdfs:', pdf.numpy() )


	# 
	# Test copy tree
	# 
	printTitle('Test Copy Tree 2nd time')

	quadTreeNode = myTree.copyTree( rootIndex= mi.UInt32(0) )

	# Print tree structure
	printBoldUnderLine('tree size:', quadTreeNode.getWidth())
	printBoldUnderLine('Copy tree structure:')
	print('child_1_index:')
	print( quadTreeNode.child_1_index.numpy().tolist() )
	print('child_2_index:')
	print( quadTreeNode.child_2_index.numpy().tolist() )
	print('child_3_index:')
	print( quadTreeNode.child_3_index.numpy().tolist() )
	print('child_4_index:')
	print( quadTreeNode.child_4_index.numpy().tolist() )
	
	irradianceArray = quadTreeNode.irradiance.numpy()
	print('Copy tree irradiance node: ', irradianceArray)
	printBoldUnderLine('Copy tree irradiance:', irradianceArray[0] )
	printBoldUnderLine('Copy tree size:', quadTreeNode.getWidth())

	validateResult = myTree.validateQuadTreeNodeBBox( quadTreeNode )
	printBoldUnderLine('Validate copy tree bounding box result:', validateResult)


	# 
	# Test reset tree radiance
	# 
	printTitle('Test Reset Tree Radiance')
	myTree.resetTreeIrradiance( rootIndex )
	printBoldUnderLine('Tree irradiance after reset:', myTree.quadTreeNode.irradiance.numpy())
	leafNodeIndex = myTree.getAllLeafNodeIndex( rootIndex )
	leafNodeIrradiance = dr.gather(mi.Float, myTree.quadTreeNode.irradiance, leafNodeIndex)
	leafNodeIrradianceSum = np.sum(leafNodeIrradiance.numpy())
	printBoldUnderLine('Tree irradiance sum after reset:', leafNodeIrradianceSum)
	printBoldUnderLine('Tree size:', myTree.quadTreeNode.getWidth())


	# 
	# Add data into skeleton tree
	# 
	printTitle('Test Add Data into Skeleton Tree (the reset tree that hold structure but not irradiance)')
	# Generate new data
	surfaceInteractionRecord = generateRandomData()
	dataRootIndex = dr.full( mi.UInt32, 0, N )
	myTree.addDataPropagate( dataRootIndex, surfaceInteractionRecord )
	
	# Check tree irradiance
	print('Tree irradiance node:', myTree.quadTreeNode.irradiance.numpy())

	irradianceArray = myTree.quadTreeNode.irradiance.numpy()
	printBoldUnderLine('Tree irradiance:', irradianceArray[0] )
	printBoldUnderLine('Tree size:', myTree.quadTreeNode.getWidth())


	# 
	# Refine tree second time
	# 
	# Set refinement threshold and refine
	printTitle('Refine Tree 2nd time')
	myTree.setRefinementThreshold( rootIndex= rootIndex, total_flux_prev_quadtree= mi.Float( irradianceArray.tolist()[0] ) )
	print('Refinement threshold:', myTree.quadTreeNode.refinementThreshold.numpy()[0])
	myTree.refine( rootIndex )
	# Check tree irradiance
	irradianceArray = myTree.quadTreeNode.irradiance.numpy()
	printBoldUnderLine('Tree irradiance node after refine:', irradianceArray)
	printBoldUnderLine('Tree irradiance:', irradianceArray[0])
	printBoldUnderLine('Tree size:', myTree.quadTreeNode.getWidth())


	# 
	# Test Clear Tree Unused Node
	# 
	printTitle('Test Clear Tree Unused Node')
	myTree.clearTreeUnusedNode()
	irradianceArray = myTree.quadTreeNode.irradiance.numpy()
	print('Clean Tree irradiance node: ', irradianceArray)
	printBoldUnderLine('Clean Tree irradiance:', irradianceArray[0] )
	printBoldUnderLine('Clean Tree size:', myTree.quadTreeNode.getWidth())


	# 
	# Test create new root node
	# 
	printTitle('Test Create New Root Node')
	newRootIndex = myTree.createRootNode(numRootNode= 2)
	printBoldUnderLine('Created root index', newRootIndex)
	printBoldUnderLine('New tree root', myTree.quadTreeNode.rootNodeIndex.numpy())
	printBoldUnderLine('New tree irradiance', myTree.quadTreeNode.irradiance.numpy())
	printBoldUnderLine('New tree depth', myTree.quadTreeNode.depth.numpy())
	printBoldUnderLine('Clean Tree size:', myTree.quadTreeNode.getWidth())



	# 
	# Test add data into new root node
	# 
	printTitle('Test Add Data into the new root and refine it')
	# Generate new data
	surfaceInteractionRecord = generateRandomData()
	# Add data
	rootIndex = dr.gather( mi.UInt32, newRootIndex, index= mi.UInt32(1) )
	dataRootIndex = dr.repeat( rootIndex, N )
	myTree.addDataPropagate(dataRootIndex, surfaceInteractionRecord)
	# Set refinement threshold
	irradiance = surfaceInteractionRecord.radiance / surfaceInteractionRecord.woPdf
	trueIrradianceSum = np.sum( irradiance.numpy() )
	myTree.setRefinementThreshold( rootIndex= rootIndex, total_flux_prev_quadtree= mi.Float( trueIrradianceSum.tolist() ) )
	# Refine
	myTree.refine( rootIndex= rootIndex )
	irradiance = myTree.quadTreeNode.irradiance.numpy()
	printBoldUnderLine('Tree irradiance:', irradiance)
	printBoldUnderLine('Total tree size:', myTree.quadTreeNode.getWidth())
	rootNodeIndexArray = myTree.quadTreeNode.rootNodeIndex.numpy()
	printBoldUnderLine('First Tree Irradiance:', irradiance[ rootNodeIndexArray[0] ])
	printBoldUnderLine('Custom Tree Irradiance:', irradiance[ rootNodeIndexArray[2] ])


	# 
	# Test append tree
	# 
	printTitle('Test Append Tree')
	printBoldUnderLine('Current Tree rootNodeIndex:', myTree.quadTreeNode.rootNodeIndex.numpy())
	printBoldUnderLine('Current Tree size:', myTree.quadTreeNode.getWidth())
	printBoldUnderLine('Current Tree irradiance:', myTree.quadTreeNode.irradiance.numpy())

	inputQuadTreeNode = myTree.copyTree( rootIndex= mi.UInt32(0) )

	printBoldUnderLine('Input Tree rootNodeIndex:', inputQuadTreeNode.rootNodeIndex.numpy())
	printBoldUnderLine('Input Tree size:', dr.width( inputQuadTreeNode.depth ))
	printBoldUnderLine('Input Tree irradiance:', inputQuadTreeNode.irradiance.numpy())

	newRootIndex = myTree.appendQuadTreeNode( inputQuadTreeNode )

	printBoldUnderLine('Merged Tree rootNodeIndex:', myTree.quadTreeNode.rootNodeIndex.numpy())
	printBoldUnderLine('Merged Tree size:', myTree.quadTreeNode.getWidth())
	printBoldUnderLine('Merged Tree irradiance:', myTree.quadTreeNode.irradiance.numpy())


	# 
	# Test add data into the appended tree
	# 
	printTitle('Test add data into the appended tree')
	# Generate new data
	surfaceInteractionRecord = generateRandomData()
	# Add data
	rootIndex = dr.gather( mi.UInt32, newRootIndex, index= mi.UInt32(0) )
	dataRootIndex = dr.repeat( rootIndex, N )
	myTree.addDataPropagate(dataRootIndex, surfaceInteractionRecord)
	printBoldUnderLine('size:', dr.width(myTree.quadTreeNode.irradiance))
	printBoldUnderLine('sum:', sum(myTree.quadTreeNode.irradiance.numpy()))
	printBoldUnderLine('New Tree irradiance:', myTree.quadTreeNode.irradiance.numpy().tolist())


	# 
	# Performance time
	# 
	endTime = time.perf_counter()
	elapsedTime = endTime - startTime
	printTitle('Elapsed time: ' + str(elapsedTime))
