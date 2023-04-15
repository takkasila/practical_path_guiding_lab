import drjit as dr
import mitsuba as mi
if __name__ == '__main__':
	mi.set_variant('cuda_ad_rgb')

from typing import Tuple, List, Union

import csv

# Define frequently used type
Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]

class SurfaceInteractionRecord:
	"""Custom data struct to hold the recordings together
	"""
	DRJIT_STRUCT = {
		'position' : mi.Vector3f,
		'direction' : mi.Vector2f,

		# Storation needed for calculation
		'bsdf' : mi.Color3f,
		'throughputBsdf' : mi.Color3f,
		'throughputRadiance' : mi.Spectrum,
		# 

		'radiance_nee' : mi.Color3f,
		'direction_nee' : mi.Vector2f,

		'radiance' : mi.Float,
		'product' : mi.Spectrum,		# radiance_spectrum * bsdf  (outgoing radiance)
		
		'woPdf' : mi.Float,
		'bsdfPdf' : mi.Float,
		'dTreePdf' : mi.Float,
		'statisticalWeight' : mi.Float,
		'isDelta' : mi.Bool,

		'active' : mi.Bool,
	}

	def __init__(self) -> None:
		self.position = mi.Vector3f()
		self.direction = mi.Vector2f()

		self.bsdf = mi.Color3f()
		self.throughputBsdf = mi.Color3f()
		self.throughputRadiance = mi.Spectrum()
		
		self.radiance = mi.Float()
		self.product = mi.Spectrum()
		
		self.woPdf = mi.Float()
		self.bsdfPdf = mi.Float()
		self.dTreePdf = mi.Float()
		self.statisticalWeight = mi.Float()
		self.isDelta = mi.Bool()

		self.active = mi.Bool()

	# Custom zero initialize callback
	# def zero_(self, size):
	# 	self.radiance += 1


class PerformanceData:

	def __init__( self ) -> None:
		self.time = []
		self.spp = []
		self.cumm_spp = []
		self.iteration = []
		self.variance = []
		self.mse = []


	def append( self, time: float= 0, spp: int= 0, cumm_spp: int= 0, iteration: int= 0, variance: float= 0, mse: float= 0 ) -> None:
		self.time.append(time)
		self.spp.append(spp)
		self.cumm_spp.append(cumm_spp)
		self.iteration.append(iteration)
		self.variance.append(variance)
		self.mse.append(mse)

	
	def saveToFile( self, fileName: str ) -> None:
		
		with open( fileName, 'w', newline='' ) as file:

			writer = csv.writer(file)

			writer.writerow(['time', 'spp', 'cumm_spp', 'iteration', 'variance', 'mse'])

			for i in range( len(self.time) ):
				writer.writerow([ self.time[i], self.spp[i], self.cumm_spp[i], self.iteration[i], self.variance[i], self.mse[i]])
			
			file.close()


def canonicalToDir(p: mi.Vector2f) -> mi.Vector3f:
	"""
		Input: Vector2f: x = Phi, y = CosTheta
			Phi: In normalized range [0, 1]
			CosTheta: In normalized range [0, 1]
		Output: Normalized Vector3f
	"""
	# Unnormalized
	cosTheta = 2 * p.y - 1
	# Compute sin(theta)
	sinTheta = dr.sqrt( 1 - cosTheta * cosTheta )
	
	# Unnormalize
	phi = dr.two_pi * p.x

	sinPhi, cosPhi = dr.sincos(phi)

	dir = mi.Vector3f()

	# 	Phi: xy, CosTheta: z
	dir.x = sinTheta * cosPhi
	dir.y = sinTheta * sinPhi
	dir.z = cosTheta

	# 	Phi: xz, CosTheta: y
	# dir.x = sinTheta * cosPhi
	# dir.z = sinTheta * sinPhi
	# dir.y = cosTheta

	return dir


def dirToCanonical(d: mi.Vector3f) -> mi.Vector2f:
	"""
		Input: Normalized Vector3f
		Output: Vector2f: x = Phi, y = CosTheta
			Phi: In normalized range [0, 1]
			CosTheta: In normalized range [0, 1]
	"""

	# 	Phi: xy, CosTheta: z
	cosTheta = dr.clip( d.z, -1, 1 )
	phi = dr.atan2(d.y, d.x)

	# 	Phi: xz, CosTheta: y
	# cosTheta = dr.clip( d.y, -1, 1 )
	# phi = dr.atan2(d.z, d.x)

	loop = mi.Loop("rotate phi", lambda: (phi))
	while loop( phi < 0 ):
		phi += 2.0 * dr.pi

	p = mi.Vector2f(0)
	p.x = phi / dr.two_pi
	p.y = ( cosTheta + 1 ) / 2

	# Have to check first if the item isfinite, if not then return [0,0]
	flag = dr.isfinite(d.x) & dr.isfinite(d.y) & dr.isfinite(d.z)
	return dr.select( flag, p, mi.Vector2f(0) )


def resizeDrJitArray(obj: dr.ArrayBase, newSize: int, isDefaultZero: bool = True) -> dr.ArrayBase:
	"""
		Create a new array with new size and copy data from the given object into.
		Does not modify the passing objects.
		If the new size is less than current then cut the tail.
		If the new size is more than current the concat tail with dr.zeros or dr.ones depends on the arg 'isDefaultZero'
	"""

	Obj_class = obj.__class__
	oldSize = dr.width(obj)

	new_obj: Obj_class = None

	if oldSize <= newSize:

		if isDefaultZero:
			new_obj = dr.zeros(Obj_class, shape= newSize)
		else:
			new_obj = dr.full(Obj_class, value= 1, shape= newSize)

		# If the starting size is zero then there is nothing to copy
		if oldSize != 0:
			dr.scatter( new_obj, obj, index= dr.arange(mi.UInt32, oldSize) )

	elif oldSize > newSize:
		index = dr.arange(mi.UInt32, newSize)
		new_obj = dr.gather( Obj_class, obj, index= index )
	
	return new_obj


def concatDrJitArray(obj_A: dr.ArrayBase, obj_B: dr.ArrayBase) -> dr.ArrayBase:
	"""
		Concatenate obj_B to the back of obj_A. Both object MUST be of the same type!
		Does not modify the passing objects.

	"""

	Obj_class = obj_A.__class__

	new_obj: Obj_class = None

	obj_A_size = dr.width(obj_A)
	obj_B_size = dr.width(obj_B)

	# Check if the passing arrays are uintiliazed/empty then return
	if obj_A_size == 0 and obj_B_size == 0:
		new_obj = Obj_class()

	elif obj_A_size == 0:
		new_obj = Obj_class( obj_B )
	
	elif obj_B_size == 0:
		new_obj = Obj_class( obj_A )
	
	else:
		# Allocate more room
		newsize = obj_A_size + obj_B_size
		new_obj = resizeDrJitArray( obj_A, newsize )

		# Scatter into
		obj_B_index_in_A = dr.arange(mi.UInt32, obj_B_size) + obj_A_size
		dr.scatter( new_obj, obj_B, obj_B_index_in_A )
	
	return new_obj


def gatherOnlyActive(dtype: dr.ArrayBase, source: dr.ArrayBase, index: mi.UInt32, active: mi.Bool) -> dr.ArrayBase:
	"""
		Gather elements from the array like drjit.gather() but also filter out inactive element.
		The resulting array could have different size with the input.
	"""
	activeIndexIndex = dr.compress(active)
	activeIndex = dr.gather(mi.UInt32, index, activeIndexIndex)
	gatherObj = dr.gather(dtype, source, activeIndex)

	return gatherObj


# Fancy printing
class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKCYAN = '\033[96m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'


def printTitle(text: str) -> None:
	print('\n')
	dashLine = '-' * ( len(text) + 4 )
	colour = bcolors.OKBLUE
	print(bcolors.BOLD + colour + dashLine + bcolors.ENDC)
	print(colour + '| ' + bcolors.BOLD + text + ' |' + bcolors.ENDC)
	print(bcolors.BOLD + colour + dashLine + bcolors.ENDC)


def printBoldUnderLine(highlightText:str, *args):
	argsText = ''
	for arg in args:
		argsText += str(arg) + ' '
	print(bcolors.BOLD + bcolors.UNDERLINE + highlightText + bcolors.ENDC + ' ' + argsText )


# 	Test 
if __name__ == '__main__':

	import numpy as np
	printTitle('Test dir<-->Canonical conversion')
	dirVector = mi.Vector3f( np.array([
		[0, 1, 0],
	]) )
	print('original direction 3D:', dirVector )
	printBoldUnderLine('converted canonical 2D:', dirToCanonical(dirVector) )
	printBoldUnderLine('converted direction 3D:', canonicalToDir( dirToCanonical(dirVector) ) )


	printTitle('Test resizeDrJitArray')
	arr_A = dr.arange(mi.UInt32, 5)
	print('arr_A:', arr_A)
	arr_A = resizeDrJitArray(arr_A, 3)
	printBoldUnderLine('arr_A down size:', arr_A)
	arr_A = resizeDrJitArray(arr_A, 6, isDefaultZero=True)
	printBoldUnderLine('arr_A up size with default zeros:', arr_A)
	arr_A = resizeDrJitArray(arr_A, 9, isDefaultZero=False)
	printBoldUnderLine('arr_A up size with default ones:', arr_A)


	printTitle('Test concatDrJitArray')
	arr_A = dr.arange(mi.UInt32, 3)
	arr_B = dr.arange(mi.UInt32, 5) * 3
	print('arr_A:', arr_A)
	print('arr_B:', arr_B)
	arr_C = concatDrJitArray(arr_A, arr_B)
	printBoldUnderLine('concat arr_A + arr_B:', arr_C)


	printTitle('Test gatherOnlyActive')
	arr_A = dr.arange(mi.UInt32, 7)
	print('arr_A:', arr_A)
	condition = arr_A > 4
	gather_A = gatherOnlyActive(mi.UInt32, arr_A, dr.arange(mi.UInt32, dr.width(arr_A)), active= condition)
	printBoldUnderLine('gather_A:', gather_A)
