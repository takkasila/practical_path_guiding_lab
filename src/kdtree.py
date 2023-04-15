from __future__ import annotations

import drjit as dr
import mitsuba as mi
if __name__ == '__main__':
	mi.set_variant('cuda_ad_rgb')

from src.common import *

from src.quadtree import QuadTree

import numpy as np

import math

class KDTreeNode:

	DRJIT_STRUCT = {
		'bbox' : mi.BoundingBox3f,
		'depth' : mi.UInt32,
		'vertCount' : mi.Float,
		'isLeaf' : mi.Bool,
		'quadTreeRootIndex' : mi.UInt32,
		'child_left_index' : mi.UInt32, 
		'child_right_index' : mi.UInt32,
	}
	
	def __init__(self) -> None:
		self.bbox = mi.BoundingBox3f()
		self.depth = mi.UInt32()
		self.vertCount = mi.Float()
		self.isLeaf = mi.Bool()
		self.quadTreeRootIndex = mi.UInt32()
		self.child_left_index = mi.UInt32()
		self.child_right_index = mi.UInt32()

	
	def copyFrom(self, kdtree_node: KDTreeNode) -> None:
		"""
			Copy from a KDTree_node onto itself.
		"""
		self.bbox = mi.BoundingBox3f()
		self.bbox.min = mi.Vector3f( kdtree_node.bbox.min )
		self.bbox.max = mi.Vector3f( kdtree_node.bbox.max )
		self.depth = mi.UInt32( kdtree_node.depth )
		self.vertCount = mi.Float( kdtree_node.vertCount )
		self.isLeaf = mi.Bool( kdtree_node.isLeaf )
		self.quadTreeRootIndex = mi.UInt32( kdtree_node.quadTreeRootIndex )
		self.child_left_index = mi.UInt32( kdtree_node.child_left_index )
		self.child_right_index = mi.UInt32( kdtree_node.child_right_index )

	
	def loadFromFile( self, dataNumpy: np.array ) -> None:
		"""
			Load data from numpy array-like object
		"""
		self.bbox = mi.BoundingBox3f( dataNumpy['kdtree_bbox_min'], dataNumpy['kdtree_bbox_max'] )
		self.depth = mi.UInt32( dataNumpy['kdtree_depth'] )
		self.vertCount = mi.Float( dataNumpy['kdtree_vertCount'] )
		self.isLeaf = mi.Bool( dataNumpy['kdtree_isLeaf'] )
		self.quadTreeRootIndex = mi.UInt32( dataNumpy['kdtree_quadTreeRootIndex'] )
		self.child_left_index = mi.UInt32( dataNumpy['kdtree_child_left_index'] )
		self.child_right_index = mi.UInt32( dataNumpy['kdtree_child_right_index'] )


	def getWidth(self) -> int:
		return dr.width(self.depth)


	def getBBox(self, idx: mi.UInt32) -> mi.BoundingBox3f():
		"""
			Get mitsuba.BoundingBox2f for the given node index
		"""
		bbox_min = dr.gather( mi.Vector3f, self.bbox.min, idx )
		bbox_max = dr.gather( mi.Vector3f, self.bbox.max, idx )
		return mi.BoundingBox3f( bbox_min, bbox_max )


	def resize(self, newSize: mi.UInt32) -> None:
		"""
			Resize the entire data structure except for the rootIndex array.
			If the new size is less than current then cut the tail. Careful, this might corrupt tree structure.
			If the new size is more than current the concat tail with dr.zeros. Except for isLeaf.
		"""
		self.depth = resizeDrJitArray( self.depth, newSize )
		self.vertCount = resizeDrJitArray( self.vertCount, newSize )
		self.isLeaf = resizeDrJitArray( self.isLeaf, newSize, isDefaultZero= False )
		self.quadTreeRootIndex = resizeDrJitArray( self.quadTreeRootIndex, newSize )
		self.child_left_index = resizeDrJitArray( self.child_left_index, newSize )
		self.child_right_index = resizeDrJitArray( self.child_right_index, newSize )

		bbox_min = resizeDrJitArray( self.bbox.min, newSize )
		bbox_max = resizeDrJitArray( self.bbox.max, newSize )
		self.bbox = mi.BoundingBox3f( bbox_min, bbox_max )

		dr.eval(
			self.depth,
			self.vertCount,
			self.isLeaf,
			self.quadTreeRootIndex,
			self.child_left_index,
			self.child_right_index,
			self.bbox.min,
			self.bbox.max
		)


class KDTree:
	"""Special KD-Tree that store a QuadTree at each leaf node.
		- The split happens in the middle of the bounding box.
		- Only the leaf node hold a reference toward a QuadTree.
		- 'max_leaf_size' specifies how many photons can be contain in a node.
		- Does not automatically refine itself when storing data. Only refine when call to do so.
		- 3 Dimensional
		- Alternately splitting x, y, z
	"""
	def __init__(self,
					max_leaf_size: float = 1,
					maxDepth: int = 10
	) -> None:
		# Init KDTreeNode
		self.kdTreeNode: KDTreeNode = dr.zeros( KDTreeNode, shape= 1 )
		self.kdTreeNode.isLeaf = mi.Bool(True)
		self.kdTreeNode.bbox = mi.BoundingBox3f( [0, 0, 0], [1, 1, 1] )

		self.maxLeafSize = max_leaf_size
		self.maxDepth = maxDepth

		# Init QuadTree
		self.quadTree = QuadTree()


	def setup(self, bbox_min: mi.Vector3f, bbox_max: mi.Vector3f) -> None:
		"""
			Delay passing of bounding box. Must be call after the class is init and before any 
			operation with this class.
		"""
		self.kdTreeNode.bbox = mi.BoundingBox3f( bbox_min, bbox_max )
	

	def copyFrom(self, kdtree: KDTree ) -> None:
		"""
			Copy from a KDTree onto itself.
		"""
		# Copy KDTreeNode
		self.kdTreeNode.copyFrom( kdtree.kdTreeNode )

		# Copy self properties
		self.maxLeafSize = kdtree.maxLeafSize
		self.maxDepth = kdtree.maxDepth

		# Copy QuadTree
		self.quadTree.copyFrom( kdtree.quadTree )
	

	def loadFromFile( self, dataNumpy: np.array ) -> None:
		"""
			Load data from numpy array-like object
		"""
		# Load KDTree
		self.maxLeafSize = int(dataNumpy['kdtree_maxLeafSize'])
		self.maxDepth = int(dataNumpy['kdtree_maxDepth'])

		# Load KDTreeNode
		self.kdTreeNode = KDTreeNode()
		self.kdTreeNode.loadFromFile( dataNumpy )

		# Load QuadTree
		self.quadTree = QuadTree()
		self.quadTree.loadFromFile( dataNumpy )

	
	def getAllLeafNodeIndex(self) -> mi.UInt32:
		"""
			Get all leaf node indicies from the given root node index.
		"""
		return dr.compress( self.kdTreeNode.isLeaf )
	

	def addDataPropagate(self, surfaceInteractionRecord: SurfaceInteractionRecord) -> None:
		"""
			Traverse the tree and add data along the way until reach the corresponding leaf node.
			- surfaceInteractionRecord: contains interaction data on the surface.
		"""
		p = surfaceInteractionRecord.position

		size = dr.width(p)
		# 	Start searching from root node
		nodeIndex = dr.zeros( mi.UInt32, size )

		# 	Test if data is within the root node bbox
		rootNodeBBox = self.kdTreeNode.getBBox( 0 )
		active = rootNodeBBox.contains( p )

		loop = mi.Loop( name = 'KDTree propagate data to leaf', state= lambda: (active, nodeIndex) )
		while loop( active ):

			# Increase vertCount
			dr.scatter_reduce( dr.ReduceOp.Add, self.kdTreeNode.vertCount, 1, nodeIndex, active )

			# If at leaf node then stop. Otherwise, continue
			isLeafNode = dr.gather( mi.Bool, self.kdTreeNode.isLeaf, nodeIndex, active )
			
			# Else, check which children node it belongs to and store the node index
			isNotLeafNode = ~isLeafNode
			active &= isNotLeafNode 

			child_left_idx = dr.gather( mi.UInt32, self.kdTreeNode.child_left_index, nodeIndex, active )
			child_right_idx = dr.gather( mi.UInt32, self.kdTreeNode.child_right_index, nodeIndex, active )

			child_left_bbox = self.kdTreeNode.getBBox( child_left_idx )
			child_left_bbox_test = child_left_bbox.contains( p )
			nodeIndex[ child_left_bbox_test & active ] = child_left_idx

			child_right_bbox = self.kdTreeNode.getBBox( child_right_idx )
			child_right_bbox_test = child_right_bbox.contains( p )
			nodeIndex[ child_right_bbox_test & active ] = child_right_idx

		
		dr.eval( self.kdTreeNode.vertCount )

		# After traversing the tree, we now know which leaf node each surfaceInteractionRecord is belong to.
		# Get the corresponding QuadTree root index and continue propagate data into the QuadTree
		quadTreeRoot = dr.gather( mi.UInt32, self.kdTreeNode.quadTreeRootIndex, nodeIndex )
		self.quadTree.addDataPropagate( rootIndex= quadTreeRoot, surfaceInteractionRecord= surfaceInteractionRecord )

	
	
	def split(self, idx: mi.UInt32) -> None:
		"""
			Split leaf KDTreeNode at index 'idx'. We handle splitting here instead of inside KDTreeNode_CUDA class
			because we also need to handle splitting of the QuadTree at leaf node.
			- idx: node index to spit. Need to guaranteed that there is no duplicating index.
		"""
		# 	Allocate new memory
		numSplitNode = dr.width(idx)
		numNewNode = numSplitNode * 2
		oldSize = self.kdTreeNode.getWidth()
		newsize = oldSize + numNewNode
		self.kdTreeNode.resize( newsize )

		# 	Calculate new children index
		childrenNodeIndex = dr.arange( mi.UInt32, numSplitNode )
		child_left_index = childrenNodeIndex * 2 + 0 + oldSize
		child_right_index = childrenNodeIndex * 2 + 1 + oldSize

		# 	Update current node's new children index
		dr.scatter( self.kdTreeNode.child_left_index, child_left_index, idx )
		dr.scatter( self.kdTreeNode.child_right_index, child_right_index, idx )

		# 	Update current node's isLeaf
		dr.scatter( self.kdTreeNode.isLeaf, False, idx )

		# 	Update children's depth
		depth = dr.gather( mi.UInt32, self.kdTreeNode.depth, idx )
		childrenDepth = depth + 1
		dr.scatter( self.kdTreeNode.depth, childrenDepth, child_left_index )
		dr.scatter( self.kdTreeNode.depth, childrenDepth, child_right_index )

		# 	Update children's vertCount
		vertCount = dr.gather( mi.Float, self.kdTreeNode.vertCount, idx )
		vertCount[vertCount > 0] = vertCount / 2
		dr.scatter( self.kdTreeNode.vertCount, vertCount, child_left_index )
		dr.scatter( self.kdTreeNode.vertCount, vertCount, child_right_index )

		# Update children's bbox
		# 		Bounding Box Splitting
		bbox_min = dr.gather( mi.Vector3f, self.kdTreeNode.bbox.min, idx )
		bbox_max = dr.gather( mi.Vector3f, self.kdTreeNode.bbox.max, idx )
		bbox_mid = (bbox_min + bbox_max) / 2


		# 	Selectively pick out x, y, or z component of the bbox_mid vector
		# 	depending on the interested splitting axis.
		# 	And then use them to set children bounding box
		bbox_mid_plain: mi.Float = dr.ravel( bbox_mid )
		splitAxis = depth % 3
		bbox_mid_componentIndex = dr.arange( mi.UInt32, numSplitNode ) * 3 + splitAxis
		bbox_mid_component = dr.gather( mi.Float, bbox_mid_plain, bbox_mid_componentIndex )


		# 		Left child
		bbox_left_min = mi.Vector3f( bbox_min )
		bbox_left_max = mi.Vector3f( bbox_max )

		bbox_left_max_plain = dr.ravel( bbox_left_max )
		dr.scatter( bbox_left_max_plain, bbox_mid_component, bbox_mid_componentIndex  )
		bbox_left_max = dr.unravel( mi.Vector3f, bbox_left_max_plain )


		# 		Right child
		bbox_right_min = mi.Vector3f( bbox_min )
		bbox_right_max = mi.Vector3f( bbox_max )

		bbox_right_min_plain = dr.ravel( bbox_right_min )
		dr.scatter( bbox_right_min_plain, bbox_mid_component, bbox_mid_componentIndex )
		bbox_right_min = dr.unravel( mi.Vector3f, bbox_right_min_plain )

		# 	Save
		dr.scatter( self.kdTreeNode.bbox.min, bbox_left_min, child_left_index )
		dr.scatter( self.kdTreeNode.bbox.max, bbox_left_max, child_left_index )

		dr.scatter( self.kdTreeNode.bbox.min, bbox_right_min, child_right_index )
		dr.scatter( self.kdTreeNode.bbox.max, bbox_right_max, child_right_index )

		# Update the tree
		dr.eval( self.kdTreeNode.vertCount, self.kdTreeNode.bbox.min, self.kdTreeNode.bbox.max )


		# 	QuadTree
		# 	Copy parent QuadTreeNode.
		# 	Since we already have a QuadTreeNode from the parent. We can move ownership 
		# 	from parent to the left child and copy another one for the right child.

		# 		Copy QuadTreeRoot index from parent to child
		quadTreeRootIndex = dr.gather( mi.UInt32, self.kdTreeNode.quadTreeRootIndex, idx )
		dr.scatter( self.kdTreeNode.quadTreeRootIndex, quadTreeRootIndex, child_left_index )

		# 		Copy QuadTreeNode from parent
		child_right_quadTree = self.quadTree.copyTree( rootIndex= quadTreeRootIndex )
		# 		Append children QuadTreeNode into the main QuadTree class
		child_right_quadTree_rootIndex = self.quadTree.appendQuadTreeNode( child_right_quadTree )
		dr.scatter( self.kdTreeNode.quadTreeRootIndex, child_right_quadTree_rootIndex, child_right_index )


	
	def setRefinementThreshold(self, iteration: int) -> None:
		# This constant is from the paper
		c = 12000
		self.maxLeafSize = c * math.sqrt( math.pow(2, iteration) )

	
	def refine(self) -> None:
		""" Refine the tree until conditions are met.
			1. Split if the node's flux is more than the threshold and depth is less than the max depth.
		"""
		# 
		# 	Split condition
		# 
		active = True
		while active:
			# Get all leaf node
			leafNodeIndex = self.getAllLeafNodeIndex()

			# If leaf node vertCount is more than threshold then split
			vertCount = dr.gather( mi.Float, self.kdTreeNode.vertCount, leafNodeIndex )
			depth = dr.gather( mi.UInt32, self.kdTreeNode.depth, leafNodeIndex )
			condition = ( vertCount > self.maxLeafSize ) & ( depth < self.maxDepth )

			# If there is atleast one node to split, then split and continue the loop.
			active = dr.any( condition )
			if active:

				# Get list of leaf node that need to split
				splitNodeIndex = dr.gather( mi.UInt32, leafNodeIndex, dr.compress(condition) )

				# Split
				self.split( splitNodeIndex )


	def validateTreeNodeBBox(self) -> bool:
		nodeIndex = dr.arange(mi.UInt32, dr.width(self.kdTreeNode.isLeaf))

		# If at leaf node then stop, if not then continue
		isNotLeafNode = ~self.kdTreeNode.isLeaf
		active = isNotLeafNode 

		# Else, check which child node it belongs to and store the node index
		child_left_idx = dr.gather(mi.UInt32, self.kdTreeNode.child_left_index, index= nodeIndex, active= active)
		child_right_idx = dr.gather(mi.UInt32, self.kdTreeNode.child_right_index, index= nodeIndex, active= active)

		node_bbox = self.kdTreeNode.getBBox( nodeIndex )

		child_left_bbox = self.kdTreeNode.getBBox( child_left_idx )
		child_right_bbox = self.kdTreeNode.getBBox( child_right_idx )

		valid_child_left = ( (child_left_bbox.min.x >= node_bbox.min.x) & (child_left_bbox.max.x <= node_bbox.max.x) \
							& (child_left_bbox.min.y >= node_bbox.min.y) & (child_left_bbox.max.y <= node_bbox.max.y) \
							& (child_left_bbox.min.z >= node_bbox.min.z) & (child_left_bbox.max.z <= node_bbox.max.z) )

		valid_child_right = ( (child_right_bbox.min.x >= node_bbox.min.x) & (child_right_bbox.max.x <= node_bbox.max.x) \
							& (child_right_bbox.min.y >= node_bbox.min.y) & (child_right_bbox.max.y <= node_bbox.max.y) \
							& (child_right_bbox.min.z >= node_bbox.min.z) & (child_right_bbox.max.z <= node_bbox.max.z) )

		none_bbox_test_fail = active & (~valid_child_left | ~valid_child_right)

		dr.printf_async("failed bbox test n:(%f %f)-(%f %f) c1:(%f %f)-(%f %f) c2:(%f %f)-(%f %f) c3:(%f %f)-(%f %f) c4:(%f %f)-(%f %f)\n",
			node_bbox.min.x, node_bbox.min.y, node_bbox.max.x, node_bbox.max.y,
			child_left_bbox.min.x, child_left_bbox.min.y, child_left_bbox.max.x, child_left_bbox.max.y,
			child_right_bbox.min.x, child_right_bbox.min.y, child_right_bbox.max.x, child_right_bbox.max.y,
			active=none_bbox_test_fail)

		dr.eval()

		if dr.width( dr.compress( none_bbox_test_fail ) ) > 0:
			return False
		else:
			return True
	

	def resetTreeVertCount(self) -> None:
		"""
			Traverse the tree from root to leaf and set all vertCount to zero.
		"""

		# Start from the root node
		nodeIndex = mi.UInt32(0)

		active = True

		while active:

			# Set node irradiance to zero
			dr.scatter( self.kdTreeNode.vertCount, 0, nodeIndex )

			# Check if leaf node, if yes then stop
			isLeafNode = dr.gather( mi.Bool, self.kdTreeNode.isLeaf, nodeIndex )
			isNotLeafNode = ~isLeafNode		# Bit flip to active on the non-leaf node

			# If there is at least one non-leaf node
			active = dr.any( isNotLeafNode )
			if active:

				# Get leaf node index
				nonLeafNodeIndex = dr.gather( mi.UInt32, nodeIndex, dr.compress( isNotLeafNode ) )

				# Get children index
				child_left_idx = dr.gather( mi.UInt32, self.kdTreeNode.child_left_index, nonLeafNodeIndex )
				child_right_idx = dr.gather( mi.UInt32, self.kdTreeNode.child_right_index, nonLeafNodeIndex )

				# Set all children index to be travel in the new iteration
				nodeIndex = concatDrJitArray( child_left_idx, child_right_idx )
	

	def getLeafNodeIndex(self, position: mi.Vector3f, active: mi.Bool = True) -> mi.UInt32:
		"""
			Find the corresponding leaf node index
			- position: search position
			- active: Mitsuba active boolean
		"""
		# 	Start searching from root node
		size = dr.width(position)
		nodeIndex = dr.zeros( mi.UInt32, size )

		# 	Test if data is within the root node bbox
		rootNodeBBox = self.kdTreeNode.getBBox( 0 )
		search_active = rootNodeBBox.contains( position ) & active

		loop = mi.Loop( name = 'KDTree Get Leaf Node Index', state= lambda: (search_active, nodeIndex) )
		while loop( search_active ):

			# If at leaf node then stop. Otherwise, continue
			isLeafNode = dr.gather( mi.Bool, self.kdTreeNode.isLeaf, nodeIndex, search_active )
			
			# Else, check which children node it belongs to and store the node index
			isNotLeafNode = ~isLeafNode
			search_active &= isNotLeafNode 

			child_left_idx = dr.gather( mi.UInt32, self.kdTreeNode.child_left_index, nodeIndex, search_active )
			child_right_idx = dr.gather( mi.UInt32, self.kdTreeNode.child_right_index, nodeIndex, search_active )

			child_left_bbox = self.kdTreeNode.getBBox( child_left_idx )
			child_left_bbox_test = child_left_bbox.contains( position )
			nodeIndex[ child_left_bbox_test & search_active ] = child_left_idx

			child_right_bbox = self.kdTreeNode.getBBox( child_right_idx )
			child_right_bbox_test = child_right_bbox.contains( position )
			nodeIndex[ child_right_bbox_test & search_active ] = child_right_idx
		
		return nodeIndex
		

	def sample(self, position: mi.Vector3f, sampler: mi.Sampler, active: mi.Bool = True) -> List[ mi.Vector3f, mi.Float ]:
		"""
			Sample a direction and pdf at a position from the QuadTree
			- position: position of interest
			- sampler: Mitsuba sampler
			- active: Mitsuba active boolean
		"""
		# Find corresponding quadtree root 
		leafNodeIndex = self.getLeafNodeIndex( position, active )
		quadTreeRoot = dr.gather( mi.UInt32, self.kdTreeNode.quadTreeRootIndex, leafNodeIndex, active )
		sampledDirection = self.quadTree.sampleQuadTree(quadTreeRoot, sampler, active)
		sampledPdf = self.quadTree.pdfQuadTree(quadTreeRoot, sampledDirection, active)

		return sampledDirection, sampledPdf

	
	def pdf( self, position: mi.Vector3f, direction: mi.Vector3f, active: mi.Bool = True ) -> mi.Float:
		
		# Find corresponding quadtree root 
		leafNodeIndex = self.getLeafNodeIndex( position, active )
		quadTreeRoot = dr.gather( mi.UInt32, self.kdTreeNode.quadTreeRootIndex, leafNodeIndex, active )
		sampledPdf = self.quadTree.pdfQuadTree(quadTreeRoot, direction, active)

		return sampledPdf


	"""
		QuadTree Handler
	"""

	def setQuadTreeRefinementThreshold(self) -> None:
		"""
			Set every QuadTrees' refinement threshold
		"""
		leafNodeIndex = self.getAllLeafNodeIndex()
		quadTreeRootIndex = dr.gather( mi.UInt32, self.kdTreeNode.quadTreeRootIndex, leafNodeIndex )

		quadTreeRootNode = dr.gather( mi.UInt32, self.quadTree.quadTreeNode.rootNodeIndex, quadTreeRootIndex )

		quadTreeRootNodeIrradiance = dr.gather( mi.Float, self.quadTree.quadTreeNode.irradiance, quadTreeRootNode )

		self.quadTree.setRefinementThreshold( rootIndex= quadTreeRootIndex, total_flux_prev_quadtree= quadTreeRootNodeIrradiance )

	
	def refineAllQuadTree(self) -> None:
		"""
			Refine every QuadTree at each leaf node
		"""
		leafNodeIndex = self.getAllLeafNodeIndex()
		quadTreeRootIndex = dr.gather( mi.UInt32, self.kdTreeNode.quadTreeRootIndex, leafNodeIndex )

		self.quadTree.refine( rootIndex= quadTreeRootIndex )

	
	def cleanUnusedQuadTree(self) -> None:
		self.quadTree.clearTreeUnusedNode()

	
	def resetAllQuadTreeIrradiance(self) -> None:
		self.quadTree.resetAllTreeIrradiance()

	
	"""
		Utility
	"""

	def saveToFile( self, fileName: str ) -> None:
		"""
			Save both KDTree and QuadTree data into numpy compressed file.
		"""
		# Gather data from KDTree
		kdtree_maxLeafSize = self.maxLeafSize
		kdtree_maxDepth = self.maxDepth
		
		kdtree_bbox_min = self.kdTreeNode.bbox.min.numpy()
		kdtree_bbox_max = self.kdTreeNode.bbox.max.numpy()
		kdtree_depth = self.kdTreeNode.depth.numpy()
		kdtree_vertCount = self.kdTreeNode.vertCount.numpy()
		kdtree_isLeaf = self.kdTreeNode.isLeaf.numpy()
		kdtree_quadTreeRootIndex = self.kdTreeNode.quadTreeRootIndex.numpy()
		kdtree_child_left_index = self.kdTreeNode.child_left_index.numpy()
		kdtree_child_right_index = self.kdTreeNode.child_right_index.numpy()


		# Gather data from QuadTree
		quadtree_maxDepth = self.quadTree.maxDepth
		quadtree_isStoreNEERadiance = self.quadTree.isStoreNEERadiance
		
		quadtree_rootNodeIndex = self.quadTree.quadTreeNode.rootNodeIndex.numpy()
		quadtree_bbox_min = self.quadTree.quadTreeNode.bbox.min.numpy()
		quadtree_bbox_max = self.quadTree.quadTreeNode.bbox.max.numpy()
		quadtree_depth = self.quadTree.quadTreeNode.depth.numpy()
		quadtree_irradiance = self.quadTree.quadTreeNode.irradiance.numpy()
		quadtree_isLeaf = self.quadTree.quadTreeNode.isLeaf.numpy()
		quadtree_refinementThreshold = self.quadTree.quadTreeNode.refinementThreshold.numpy()
		quadtree_child_1_index = self.quadTree.quadTreeNode.child_1_index.numpy()
		quadtree_child_2_index = self.quadTree.quadTreeNode.child_2_index.numpy()
		quadtree_child_3_index = self.quadTree.quadTreeNode.child_3_index.numpy()
		quadtree_child_4_index = self.quadTree.quadTreeNode.child_4_index.numpy()


		# Save to numpy file
		np.savez_compressed(
			file = fileName,
			
			kdtree_maxLeafSize = kdtree_maxLeafSize,
			kdtree_maxDepth = kdtree_maxDepth,
			kdtree_bbox_min = kdtree_bbox_min,
			kdtree_bbox_max = kdtree_bbox_max,
			kdtree_depth = kdtree_depth,
			kdtree_vertCount = kdtree_vertCount,
			kdtree_isLeaf = kdtree_isLeaf,
			kdtree_quadTreeRootIndex = kdtree_quadTreeRootIndex,
			kdtree_child_left_index = kdtree_child_left_index,
			kdtree_child_right_index = kdtree_child_right_index,

			quadtree_maxDepth = quadtree_maxDepth,
			quadtree_isStoreNEERadiance = quadtree_isStoreNEERadiance,
			quadtree_rootNodeIndex = quadtree_rootNodeIndex,
			quadtree_bbox_min = quadtree_bbox_min,
			quadtree_bbox_max = quadtree_bbox_max,
			quadtree_depth = quadtree_depth,
			quadtree_irradiance = quadtree_irradiance,
			quadtree_isLeaf = quadtree_isLeaf,
			quadtree_refinementThreshold = quadtree_refinementThreshold,
			quadtree_child_1_index = quadtree_child_1_index,
			quadtree_child_2_index = quadtree_child_2_index,
			quadtree_child_3_index = quadtree_child_3_index,
			quadtree_child_4_index = quadtree_child_4_index,
		)
	

	def saveOBJ( self, fileName: str ) -> None:
		"""
			Write KDTree bounding boxes to a Wavefront Obj file
		"""

		# Get bounding box data
		bbox_min_arr = self.kdTreeNode.bbox.min.numpy()
		bbox_max_arr = self.kdTreeNode.bbox.max.numpy()
		numBoxes = np.shape(bbox_min_arr)[0]

		# Retrieve sceneName from fileName
		sceneName = fileName.split('/')[-1].split('.')[0]

		vertCount = 1	#	wavefront obj vertex index starts from 1

		# Create a file
		with open( fileName, 'w' ) as file:

			# Header
			file.write( '# OBJ file of KDTree Bounding Boxes\n' )
			
			# Object name
			file.write( f'o {sceneName}\n' )
			
			# Iterate over each bbox
			for i in range( numBoxes ):

				bbox_min = bbox_min_arr[i]
				bbox_max = bbox_max_arr[i]

				# Mitsuba scene coordinate system is: Upward: Y, Forward: -Z
				# Line on x-z bottom plane
				file.write( f'v {bbox_min[0]} {bbox_min[1]} {bbox_min[2]}\n' )	# v0
				file.write( f'v {bbox_max[0]} {bbox_min[1]} {bbox_min[2]}\n' )	# v1
				file.write( f'v {bbox_max[0]} {bbox_min[1]} {bbox_max[2]}\n' )	# v2
				file.write( f'v {bbox_min[0]} {bbox_min[1]} {bbox_max[2]}\n' )	# v3

				file.write( f'l {vertCount + 0} {vertCount + 1} {vertCount + 2} {vertCount + 3} {vertCount + 0}\n' )
				
				# Line on x-z top plane
				file.write( f'v {bbox_min[0]} {bbox_max[1]} {bbox_min[2]}\n' )	# v4
				file.write( f'v {bbox_max[0]} {bbox_max[1]} {bbox_min[2]}\n' )	# v5
				file.write( f'v {bbox_max[0]} {bbox_max[1]} {bbox_max[2]}\n' )	# v6
				file.write( f'v {bbox_min[0]} {bbox_max[1]} {bbox_max[2]}\n' )	# v7

				file.write( f'l {vertCount + 4} {vertCount + 5} {vertCount + 6} {vertCount + 7} {vertCount + 4}\n' )

				# Line connect top and bottom plane
				file.write( f'l {vertCount + 0} {vertCount + 4}\n' )
				file.write( f'l {vertCount + 1} {vertCount + 5}\n' )
				file.write( f'l {vertCount + 2} {vertCount + 6}\n' )
				file.write( f'l {vertCount + 3} {vertCount + 7}\n' )

				# Update vertex counter
				vertCount += 8
			
			
			# Close file
			file.close()



if __name__ == '__main__':
	
	from numpy.random import rand as np_rand
	import time		# Performance timer

	N = 1000000
	
	def generateRandomData() -> SurfaceInteractionRecord:

		print('N:', N)
		surfaceInteractionRecord = dr.zeros( SurfaceInteractionRecord, N)

		surfaceInteractionRecord.position = mi.Vector3f( np_rand(N, 3) * 100 )
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
	myTree = KDTree()
	myTree.setup( bbox_min= [0, 0, 0], bbox_max= [100, 100, 100] )


	# 
	# Split the leaf node once for proper testing of addData
	# 
	printTitle('Test Create Tree and Splitting')
	# 1st split
	leafNodeIndex = myTree.getAllLeafNodeIndex()
	myTree.split( leafNodeIndex )
	# 2nd split
	leafNodeIndex = myTree.getAllLeafNodeIndex()
	myTree.split( leafNodeIndex )

	# Print tree structure
	printBoldUnderLine('tree size:', myTree.kdTreeNode.getWidth())
	printBoldUnderLine('child_left_index:')
	print(myTree.kdTreeNode.child_left_index.numpy().tolist())
	printBoldUnderLine('child_right_index:')
	print(myTree.kdTreeNode.child_right_index.numpy().tolist())

	# Print tree bounding box
	printBoldUnderLine('bbox_min:')
	print(myTree.kdTreeNode.bbox.min.numpy())
	printBoldUnderLine('bbox_max:')
	print(myTree.kdTreeNode.bbox.max.numpy())

	numLeafNode = dr.width( dr.compress( myTree.kdTreeNode.isLeaf ) )
	printBoldUnderLine( 'Num leaf nodes:', numLeafNode )

	# Validate tree bounding box
	validateResult = myTree.validateTreeNodeBBox()
	printBoldUnderLine('Validate myTree bounding box result:', validateResult)


	# 
	# Add data
	# 
	printTitle('Test Add Data')
	myTree.addDataPropagate( surfaceInteractionRecord )

	# Check tree vertCount
	printBoldUnderLine('Tree vertCount:', myTree.kdTreeNode.vertCount.numpy())

	# Check QuadTree irradiance
	printBoldUnderLine('QuadTree irradiance:', myTree.quadTree.quadTreeNode.irradiance.numpy())
	irradiance = surfaceInteractionRecord.radiance / surfaceInteractionRecord.woPdf
	irradianceSum = sum( irradiance.numpy() )
	printBoldUnderLine('Real irradiance sum:', irradianceSum)


	# 
	# Refine tree
	# 
	printTitle('Test Refine Tree')
	myTree.setRefinementThreshold( iteration= 0 )
	printBoldUnderLine( 'Tree max_leaf_size:', myTree.maxLeafSize )

	myTree.refine()

	myTree.setQuadTreeRefinementThreshold()

	myTree.refineAllQuadTree()
	
	myTree.cleanUnusedQuadTree()

	printBoldUnderLine( 'Tree size after refine:', myTree.kdTreeNode.getWidth() )
	printBoldUnderLine( 'Tree vertCount after refine:', myTree.kdTreeNode.vertCount.numpy() )
	
	leafNodeIndex = myTree.getAllLeafNodeIndex()
	leafNodeVertCount = dr.gather( mi.Float, myTree.kdTreeNode.vertCount, leafNodeIndex )
	leafNodeVertCountSum = np.sum( leafNodeVertCount.numpy() )
	printBoldUnderLine( 'Tree vertCount sum after refine:', leafNodeVertCountSum )

	printBoldUnderLine( 'QuadTree rootNodeIndex:', myTree.quadTree.quadTreeNode.rootNodeIndex.numpy() )
	printBoldUnderLine( 'QuadTree size:', myTree.quadTree.quadTreeNode.getWidth() )
	printBoldUnderLine( 'QuadTree irradiance:', myTree.quadTree.quadTreeNode.irradiance.numpy() )

	leafNode = dr.compress( myTree.kdTreeNode.isLeaf )
	numLeafNode = dr.width( leafNode )
	printBoldUnderLine( 'Num leaf nodes:', numLeafNode )
	printBoldUnderLine( 'Leaf Node index:', leafNode.numpy() )

	# Validate tree bounding box
	validateResult = myTree.validateTreeNodeBBox()
	printBoldUnderLine('Validate myTree bounding box result:', validateResult)

	# 
	# Reset Tree vertCount
	# 
	printTitle('Test Reset Tree vertCount')
	myTree.resetTreeVertCount()
	printBoldUnderLine( 'Tree size after reset:', myTree.kdTreeNode.getWidth() )
	printBoldUnderLine( 'Tree vertCount after reset:', myTree.kdTreeNode.vertCount.numpy() )


	# 
	# Test Add Data into skeleton tree and refine
	# 
	printTitle( 'Test add data into skeleton tree' )
	# Generate new data
	N *= 2
	surfaceInteractionRecord = generateRandomData()
	myTree.addDataPropagate( surfaceInteractionRecord )
	myTree.setRefinementThreshold( iteration= 1 )
	myTree.refine()
	myTree.cleanUnusedQuadTree()

	# 
	# 	Test get leaf node index
	# 
	position = mi.Vector3f( 75, 25, 25 )

	leafNodeIndex = myTree.getLeafNodeIndex( position )
	printTitle( 'Test get leaf node index' )
	printBoldUnderLine( 'At position:', position.numpy() )
	printBoldUnderLine( 'leafNodeIndex:', leafNodeIndex.numpy() )

	# 
	# 	Test sample tree
	# 
	sampler = mi.load_dict({
		'type' : 'independent'
	})
	sampler.seed(0, wavefront_size= dr.width(position))
	direction, pdf = myTree.sample( position, sampler )
	printTitle( 'Test sample tree' )
	printBoldUnderLine('Sample direction & pdf:', direction, pdf)


	# 
	# Performance time
	# 
	endTime = time.perf_counter()
	elapsedTime = endTime - startTime
	printTitle('Elapsed time: ' + str(elapsedTime))
