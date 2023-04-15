from __future__ import annotations as __annotations__ # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi
if __name__ == '__main__':
	mi.set_variant('cuda_ad_rgb')

from src.common import *

import numpy as np

from src.kdtree import KDTree

epsilon = 0.00001

def mis_weight(pdf_a, pdf_b):
	"""
		Compute the Multiple Importance Sampling (MIS) weight given the densities
		of two sampling strategies according to the power heuristic.
	"""
	a2 = dr.sqr(pdf_a)
	result = dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0)
	result[ dr.isnan( result ) ] = 0
	return result
	

class PathGuidingIntegrator(mi.SamplingIntegrator):

	def __init__(self: mi.SamplingIntegrator, props: mi.Properties) -> None:
		super().__init__(props)

		# Read properties
		# 	Max Depth
		self.max_depth = props.get('max_depth', 30)
		if self.max_depth < 0 and self.max_depth != -1:
			raise Exception("\"max_depth\" must be set to -1 (infinite) or a value >= 0")

		# 	Russian Roulette 
		self.rr_depth = props.get('rr_depth', 8)
		if self.rr_depth < 0:
			raise Exception("\"rr_depth\" must be set to >= 0")

		# Declare properties
		self.numRays: int = 0
		self.array_size: int = 0
		self.isStoreNEERadiance: bool = False
		"""
			When guiding, we perform MIS with the balance heuristic between the guiding
			distribution and the BSDF, combined with probabilistically choosing one of the
			two sampling methods. This factor controls how often the BSDF is sampled
			vs. how often the guiding distribution is sampled.
			Default = 0.5 (50%)
		"""
		self.bsdfSamplingFraction = 0.5
		self.iteration = 0
		self.isFinalIter = False

		# Data recorder for transferring into the SD-Tree
		self.surfaceInteractionRecord: SurfaceInteractionRecord = None

		"""
			Two SDTrees.
			There are two SDTrees: sdTree_current, and sdTree_prev.
			In each iteration, we sample from sdTree_prev and store into sdTree_current.
			At the end of the iteration, we refine the sdTree_current, copy from sdTree_current into sdTree_prev,
			and then remove data in the sdTree_current but keeps the structure, ready to be use again.
		"""
		self.sdTree_prev = KDTree()
		self.sdTree_current = KDTree()


		# For variance calculation
		self.sumL = mi.Spectrum(0)
		self.sumL2 = mi.Spectrum(0)


	def setup( self: mi.SamplingIntegrator, 
		numRays: int, 
		bbox_min: mi.Vector3f, 
		bbox_max: mi.Vector3f,
		sdTreeMaxDepth: int = 10,
		quadTreeMaxDepth: int = 30,
		isStoreNEERadiance: bool = True,
		bsdfSamplingFraction: float = 0.5
	) -> None:
		"""
			Setting SDTree and other self attributes.
			This function need to be called BEFORE starting the rendering process.
		"""

		# Self properties
		self.numRays = numRays
		self.array_size = self.numRays * self.max_depth
		self.isStoreNEERadiance = isStoreNEERadiance
		self.bsdfSamplingFraction = bsdfSamplingFraction
		self.resetRayPathData()

		# SDTrees
		# 	SDTree current
		self.sdTree_current.setup( bbox_min, bbox_max )
		self.sdTree_current.maxDepth = sdTreeMaxDepth
		self.sdTree_current.quadTree.maxDepth = quadTreeMaxDepth
		self.sdTree_current.quadTree.isStoreNEERadiance = isStoreNEERadiance
		# 	SDTree prev
		self.sdTree_prev.copyFrom( self.sdTree_current )
	

	def resetVarianceCounter( self ) -> None:
		self.sumL = mi.Spectrum(0)
		self.sumL2 = mi.Spectrum(0)


	def resetRayPathData(self: mi.SamplingIntegrator):

		if not self.isFinalIter:
			self.surfaceInteractionRecord = dr.zeros(SurfaceInteractionRecord, shape= self.array_size)
		else:
			self.surfaceInteractionRecord = dr.zeros(SurfaceInteractionRecord, shape= 1)
	

	def setIteration(self, iteration: int, isFinalIter: bool) -> None:
		self.iteration = iteration
		self.isFinalIter = isFinalIter
	

	def sample(self: mi.SamplingIntegrator, 
				scene: mi.Scene, 
				sampler: mi.Sampler, 
				ray: mi.RayDifferential3f, 
				medium: mi.Medium = None, 
				active: bool = True,
				aovs: mi.Float = None,
				) -> Tuple[mi.Color3f, bool, List[float]]:

		# Standard BSDF evaluation context for path tracing
		bsdf_ctx = mi.BSDFContext()

		# 
		# 	Configure Loop State
		# 

		# Copy input arguments to avoid mutating the caller's state
		ray = mi.Ray3f(ray)
		# 		Path throughput weight
		throughput_weight = mi.Spectrum(1)
		# 		Depth of current vertex
		depth = mi.UInt32(0)
		# 		Radiance accumulator
		L = mi.Spectrum(0)
		# 		Index of refraction
		ior = mi.Float(1)
		# 		Active SIMD lanes
		active = mi.Bool(active)

		self.resetRayPathData()

		# Generate ray index
		ray_index = dr.arange( mi.UInt32, dr.width(ray) )

		# Variables caching information from the previous bounce
		prev_si = dr.zeros(mi.SurfaceInteraction3f)
		prev_bsdf_pdf = mi.Float(1.0)
		prev_bsdf_delta = mi.Bool(True)

		isFinalIter = mi.Bool( self.isFinalIter )

		# Record the following loop in its entirety
		loop = mi.Loop(name="Path Guiding", 
			state=lambda: (
				sampler, ray, depth, L, throughput_weight, ior, active,
				prev_si, prev_bsdf_pdf, prev_bsdf_delta, isFinalIter
			)
		)

		# Specify the max. number of loop iterations (this can help avoid
		# costly synchronization when when wavefront-style loops are generated)
		loop.set_max_iterations(self.max_depth)

		while loop(active):

			# 
			# 	Direct Emission
			# 
			
			si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=dr.eq(depth, 0))

			bsdf = si.bsdf()

			ds_direct = mi.DirectionSample3f(scene, si=si, ref=prev_si)

			emitter_pdf = scene.pdf_emitter_direction(prev_si, ds_direct, ~prev_bsdf_delta)

			mis = mis_weight(
				prev_bsdf_pdf,
				emitter_pdf
			)

			em_radiance = ds_direct.emitter.eval(si)

			Le = throughput_weight * mis * em_radiance

			# 
			# 	Emitter Sampling (Next Event Estimation)
			# 

			# Should we continue tracing to next vertex?
			active_next = (depth + 1 < self.max_depth) & si.is_valid()

			# Is emitter sampling even possible on the current vertex
			active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

			# If so, randomly sample an emitter without derivative tracking
			ds, em_weight = scene.sample_emitter_direction(
				si, sampler.next_2d(), test_visibility=True, active=active_em
			)
			active_em &= dr.neq(ds.pdf, 0.0)

			# Evaluate BSDF * cos(theta)
			wo = si.to_local(ds.d)
			bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf( bsdf_ctx, si, wo, active_em )

			# Compute PDF of getting non-delta surface (i.e. diffuse surface)
			active_sdtree_and_em = active_em & (self.iteration > 1)

			# 	store current bsdf component
			prev_bsdf_component = bsdf_ctx.component

			# 	sample direction without delta component
			bsdf_ctx.component &= ~mi.BSDFFlags.Delta
			bsdf_sample_non_delta, bsdf_weight_non_delta = bsdf.sample( bsdf_ctx, si, mi.Float( 0.5 ), mi.Vector2f(0.5, 0.5), active_sdtree_and_em )

			# 	pdf(wo | not delta)
			pdf_without_delta = bsdf.pdf( bsdf_ctx, si, bsdf_sample_non_delta.wo, active_sdtree_and_em ) 

			#	pdf(wo | delta)
			# 		set back to previous state which should have non-delta
			bsdf_ctx.component = prev_bsdf_component
			pdf_with_delta = bsdf.pdf( bsdf_ctx, si, bsdf_sample_non_delta.wo, active_sdtree_and_em ) 

			# 	pdf of getting a diffuse surface
			pdf_diffuse = (pdf_with_delta + epsilon) / (pdf_without_delta + epsilon)

			# 	pdf( wo | sdTree )
			sdtree_pdf_em = self.sdTree_prev.pdf( si.p, ds.d, active_sdtree_and_em )

			# 	pdf of sampling wo on a diffuse surface
			surface_pdf_em = self.bsdfSamplingFraction * bsdf_pdf_em + ( 1 - self.bsdfSamplingFraction ) * sdtree_pdf_em * pdf_diffuse

			# 	Also consider if SDTree is avaible to use or not, if not then just use normal bsdf_pdf_em
			surface_pdf_em = dr.select( (self.iteration <= 1), bsdf_pdf_em, surface_pdf_em )
			
			# 	MIS weight with pdf(wo | emission sampling)
			mis_em = dr.select( ds.delta, 1, mis_weight(ds.pdf, surface_pdf_em) )

			# 	evaluate NEE
			Lr_dir = throughput_weight * mis_em * bsdf_value_em * em_weight

			# 
			# 	Accumulate Radiance
			# 
			L += Le + Lr_dir


			# 
			# 	Sample next outgoing ray
			# 

			# TODO check if surface doesn't have delta then no need to ...
			# delta = mi.has_flag( bsdf.flags(), mi.BSDFFlags.Delta )

			# 	First, sample BSDF for an outgoing ray and see if has Delta component
			bsdf_sample, bsdf_weight = bsdf.sample( bsdf_ctx, si, sampler.next_1d( active_next ), sampler.next_2d(active_next), active_next )
			bsdf_pdf = bsdf_sample.pdf
			bsdf_value = bsdf_weight * bsdf_pdf # including cos(theta)
			woPdf = mi.Float( bsdf_pdf )
			wo_local = mi.Vector3f( bsdf_sample.wo )
			wo_world = si.to_world( wo_local )


			# 		If we sampled a delta component, then we have a 0 probability
            # 		of sampling that direction via sdTree
			delta = mi.has_flag( bsdf_sample.sampled_type, mi.BSDFFlags.Delta )
			do_sdtree_bsdf_mis = active_next & ~delta & (self.iteration > 1)
			
			# 	Check if sample with BSDF or with SDTree
			active_sample_sdtree_mis = sampler.next_1d(active_next) > self.bsdfSamplingFraction
			active_sample_sdtree_mis &= do_sdtree_bsdf_mis

			"""
			pdf(result of sampling) = pdf(sample generated from sd tree sampling) * pdf(do sd tree sampling) + pdf(sample generated from bsdf sampling) * (1-pdf(do sd tree sampling))
			"""
			active_sample_bsdf_without_mis = ~do_sdtree_bsdf_mis & ~active_sample_sdtree_mis
			active_sample_bsdf_mis = do_sdtree_bsdf_mis & ~active_sample_sdtree_mis

			active_sample_sdtree_mis &= active_next
			active_sample_bsdf_mis &= active_next
			active_sample_bsdf_without_mis &= active_next


			# 	If sample SDTree MIS
			sdtree_dir, sdtree_pdf = self.sdTree_prev.sample( position= si.p, sampler= sampler, active=active_sample_sdtree_mis )
			wo_world[active_sample_sdtree_mis] = sdtree_dir
			wo_local[active_sample_sdtree_mis] = si.to_local(sdtree_dir)
			bsdf_value[active_sample_sdtree_mis], bsdf_pdf[active_sample_sdtree_mis] = bsdf.eval_pdf(bsdf_ctx, si, wo_local, active_sample_sdtree_mis)

			# 	If sample BSDF MIS
			sdtree_pdf[active_sample_bsdf_mis] = self.sdTree_prev.pdf( si.p, wo_world, active_sample_bsdf_mis )

			# 	Finally, compute pdf and BSDF weight
			woPdf[active_next & do_sdtree_bsdf_mis] = ( self.bsdfSamplingFraction * bsdf_pdf ) + (1 - self.bsdfSamplingFraction) * sdtree_pdf
			bsdf_weight[active_next & do_sdtree_bsdf_mis] = bsdf_value / woPdf # overwrite previous value


			# 
			# 	Store surface interaction data
			# 	Only store if active, valid surface and not in the final iteration
			# 
			globalIndex = (ray_index * self.max_depth) + depth
			storeFlag = active & si.is_valid()
			storeFlag &= ~isFinalIter

			dr.scatter(self.surfaceInteractionRecord.position, value= si.p, index= globalIndex, active= storeFlag)
			
			# 	Convert outgoing dir (Vector3f) to Spherical coordinate 
			dr.scatter(self.surfaceInteractionRecord.direction, value= dirToCanonical(wo_world), index= globalIndex, active= storeFlag)

			dr.scatter(self.surfaceInteractionRecord.active, value= storeFlag, index= globalIndex, active= storeFlag)
			
			dr.scatter(self.surfaceInteractionRecord.bsdf, value= bsdf_weight, index= globalIndex, active= storeFlag)

			dr.scatter(self.surfaceInteractionRecord.throughputBsdf, value= throughput_weight, index= globalIndex, active= storeFlag)

			dr.scatter(self.surfaceInteractionRecord.throughputRadiance, value= L, index= globalIndex, active= storeFlag)

			isStoreNEERadiance = self.isStoreNEERadiance & storeFlag
			dr.scatter(self.surfaceInteractionRecord.radiance_nee, value= Lr_dir / throughput_weight, index= globalIndex, active= isStoreNEERadiance)
			dr.scatter(self.surfaceInteractionRecord.direction_nee, value= dirToCanonical(ds.d), index= globalIndex, active= isStoreNEERadiance)

			# 	The two is only difference if we sample material together with the tree: if(sample.x >= bsdfSamplingFraction) woPdf = bsdfPdf * bsdfSamplingFraction;
			dr.scatter(self.surfaceInteractionRecord.woPdf, value= woPdf, index= globalIndex, active= storeFlag)
			dr.scatter(self.surfaceInteractionRecord.bsdfPdf, value= bsdf_sample.pdf, index= globalIndex, active= storeFlag)

			# dTreePdf: float
			# statisticalWeight: float

			dr.scatter(self.surfaceInteractionRecord.isDelta, value= delta, index= globalIndex, active= storeFlag)


			# 
			# 	Update loop variables based on current interaction
			# 
			ray = si.spawn_ray(wo_world)
			ior *= bsdf_sample.eta
			throughput_weight *= bsdf_weight

            # Information about the current vertex needed by the next iteration
			prev_si = si
			prev_bsdf_pdf = woPdf # bsdf_sample.pdf
			prev_bsdf_delta = delta


			# 
			# 	Stopping criterion
			# 
			
			# Don't run another iteration if the throughput reached zero
			throughput_weight_max = dr.max(throughput_weight)
			active_next &= dr.neq( throughput_weight_max, 0 )

			# Russian roulette stopping probability (must cancel out ior^2
			# to obtain unitless throughput, enforces a minimum probability)
			rr_prob = dr.minimum(throughput_weight_max * ior ** 2, 0.95)

			# Apply only further along the path since, this introduces variance
			rr_active = depth >= self.rr_depth
			throughput_weight_max[rr_active] *= dr.rcp(rr_prob)
			rr_continue = sampler.next_1d() < rr_prob
			active_next &= ~rr_active | rr_continue

			active = active_next
			depth[si.is_valid()] += 1

		dr.schedule(L)
		sampler.schedule_state()
		dr.schedule(self.surfaceInteractionRecord)

		# No need to update tree in the final iteration
		if not self.isFinalIter:

			# Process path data
			Lfinal = mi.Spectrum( L )
			self.processPathData( Lfinal )
		
			# Scatter the stored data into SD-Tree
			self.scatterDataIntoSDTree()


		# Accumulate Radiance for variance calculation

		spp_per_pass = sampler.sample_count()
		
		if spp_per_pass == 1:

			self.sumL += L
			self.sumL2 += L * L

		else:

			# Process sample for each SPP
			total_size = dr.width( L )

			one_sample_size = int( total_size / spp_per_pass )

			# Each sample within render pass is lay out in a concatination fashion next to eachc other.
			# For example, if SPP per pass is 4, then:
			# 	pixel_1_sample_1, pixel_1_sample_2, pixel_1_sample_3, pixel_1_sample_4, pixel_2_sample_1, pixel_2_sample_2, ...
			base_index = dr.arange( mi.UInt32, one_sample_size ) * spp_per_pass		#	0 4 8 12 ...

			for i in range( spp_per_pass ):

				index = base_index + i		# 	e.g. i=1: 1 5 9 13 ...

				one_spp_L = dr.gather( mi.Spectrum, L, index )

				self.sumL += one_spp_L
				self.sumL2 += one_spp_L * one_spp_L
		
		
		dr.schedule( self.sumL, self.sumL2 )

		return (L, dr.neq(depth, 0), [1])
	

	def processPathData(self: mi.SamplingIntegrator, Lfinal : mi.Spectrum) -> None:
		"""	Calculate incoming radiance at each vertex
		"""

		# This launches one thread per vertex:
		globalVertexIndex = dr.arange(mi.UInt32, self.array_size)
		rayIndex = globalVertexIndex // self.max_depth		# 000001111122222...
		LfinalPerVertex = dr.gather(mi.Spectrum, Lfinal, rayIndex)

		outgoingRadiance = ( LfinalPerVertex - self.surfaceInteractionRecord.throughputRadiance ) / self.surfaceInteractionRecord.throughputBsdf
		outgoingRadiance[ dr.isnan(outgoingRadiance) ] = 0

		dr.scatter(self.surfaceInteractionRecord.product, outgoingRadiance, globalVertexIndex)

		incomingRadiance = outgoingRadiance / self.surfaceInteractionRecord.bsdf
		incomingRadiance[ dr.isnan(incomingRadiance) ] = 0

		# Convert to from Spectrum to Luminance
		incomingRadiance = mi.luminance( incomingRadiance )
		dr.scatter( self.surfaceInteractionRecord.radiance, incomingRadiance, globalVertexIndex )


	def scatterDataIntoSDTree(self: mi.SamplingIntegrator) -> None:
		"""
			Scatter surface interaction data into current SDTree.
			This also filter out invalid surface interactions before scattering.
		"""

		# Remove surfaceInteractionRecord that are inactive. This modifies the array size.
		isActive = self.surfaceInteractionRecord.active

		# Filter NaN radiance
		self.surfaceInteractionRecord.radiance[ dr.isnan(self.surfaceInteractionRecord.radiance) ] = 0
		self.surfaceInteractionRecord.radiance_nee[ dr.isnan(self.surfaceInteractionRecord.radiance_nee) ] = 0
		
		# Check if both radiance is zero then also filter out
		radiance_zero = dr.eq( self.surfaceInteractionRecord.radiance, 0 )
		radiance_nee_zero = dr.eq( mi.luminance( self.surfaceInteractionRecord.radiance_nee ), 0 )
		bothRadianceZero = radiance_zero & radiance_nee_zero

		# Remove woPdf = 0 or NaN
		woPdf_zero = dr.eq( self.surfaceInteractionRecord.woPdf, 0 )
		woPdf_nan = dr.isnan( self.surfaceInteractionRecord.woPdf )
		
		activeIndex = dr.compress(isActive & ~bothRadianceZero & ~woPdf_zero & ~woPdf_nan)


		# If there is no active element then exit
		if( dr.width( activeIndex ) == 0 ):
			return
			
		self.surfaceInteractionRecord.position = dr.gather( type(self.surfaceInteractionRecord.position), self.surfaceInteractionRecord.position, activeIndex )
		
		self.surfaceInteractionRecord.direction = dr.gather( type(self.surfaceInteractionRecord.direction), self.surfaceInteractionRecord.direction, activeIndex )
		self.surfaceInteractionRecord.radiance = dr.gather( type(self.surfaceInteractionRecord.radiance), self.surfaceInteractionRecord.radiance, activeIndex )

		if self.isStoreNEERadiance:
			self.surfaceInteractionRecord.radiance_nee = dr.gather( type(self.surfaceInteractionRecord.radiance_nee), self.surfaceInteractionRecord.radiance_nee, activeIndex )
			self.surfaceInteractionRecord.direction_nee = dr.gather( type(self.surfaceInteractionRecord.direction_nee), self.surfaceInteractionRecord.direction_nee, activeIndex )
		
		self.surfaceInteractionRecord.product = dr.gather( type(self.surfaceInteractionRecord.product), self.surfaceInteractionRecord.product, activeIndex )
		self.surfaceInteractionRecord.woPdf = dr.gather( type(self.surfaceInteractionRecord.woPdf), self.surfaceInteractionRecord.woPdf, activeIndex )
		self.surfaceInteractionRecord.bsdfPdf = dr.gather( type(self.surfaceInteractionRecord.bsdfPdf), self.surfaceInteractionRecord.bsdfPdf, activeIndex )
		self.surfaceInteractionRecord.isDelta = dr.gather( type(self.surfaceInteractionRecord.isDelta), self.surfaceInteractionRecord.isDelta, activeIndex )

		# Scatter into current SDTree
		self.sdTree_current.addDataPropagate( self.surfaceInteractionRecord )
		

	def computeMSE( self, spp: float, groundTruth: mi.Color3f ) -> float:
		"""
			Compute Mean Square Error with respect to ground truth.
			If ground truth is not provided then compare to itself.
		"""
		L = self.sumL / spp

		# 		MSE compare to ground truth
		mse = (L - groundTruth) ** 2
		
		mse = mi.luminance( mse )
		mse = dr.minimum( mse, 10000 ) # 	Cutoff outlier for more stable mse
		mse = dr.mean( mse )[0]

		return mse


	def computeVariance( self, spp: float, groundTruth: mi.Color3f = None ) -> float:
		"""
			Compute Variance
			If ground truth is not provided then compare to itself.
		"""

		if groundTruth is not None:
			variance = (self.sumL2 / spp) - (groundTruth * groundTruth)

			variance = mi.luminance( variance )
			variance = dr.minimum( variance, 10000 )
			variance = dr.mean( variance )[0]

			# Population variance
			variance /= spp

		else:
			L = self.sumL / spp
			L2 = self.sumL2 / spp
			
			variance = L2 - (L * L)
		
			variance = mi.luminance( variance )
			variance = dr.minimum( variance, 10000 )
			variance = dr.mean( variance )[0]

			if spp > 1:
				# Sample variance
				variance /= spp - 1

		return variance
	

	def refine(self) -> None:
		"""
			Refine the current SDTree.
		"""

		# Refine the KDTree
		self.sdTree_current.refine()

		# Refind QuadTree
		self.sdTree_current.setQuadTreeRefinementThreshold()
		self.sdTree_current.refineAllQuadTree()
	
	
	def refineAndPrepareSDTreeForNextIteration(self) -> None:
		"""
			Prepare SDTree for the next iteration.
			1. Refine the current SDTree
			2. Clean unused quadtree in current SDTree
			3. Copy from current SDTree to previous SDTree
			4. Reset data of the current SDTree
		"""
		# 1. Set refinement threshold and refine the current SDTree
		self.sdTree_current.setRefinementThreshold( self.iteration )
		self.refine()

		# 2. Clean unused quadtree in current SDTree
		self.sdTree_current.cleanUnusedQuadTree()

		# 3. Copy from current SDTree to previous SDTree
		self.sdTree_prev.copyFrom( self.sdTree_current )

		# 4. Reset current SDTree data
		self.sdTree_current.resetTreeVertCount()
		self.sdTree_current.resetAllQuadTreeIrradiance()
	

	def saveSDTreeToFile(self, fileName: str) -> None:
		"""
			Save previous SDTree data which contains both values and structure of the tree,
			into a file.
		"""
		self.sdTree_prev.saveToFile(fileName)
	

	def loadSDTreeFromFile(self, fileName: str) -> None:
		"""
			Load sdTree data from a given .npz file
		"""
		dataNumpy = np.load( fileName )
		self.sdTree_prev.loadFromFile( dataNumpy )

		self.sdTree_current.copyFrom( self.sdTree_prev )

		# 4. Reset current SDTree data
		self.sdTree_current.resetTreeVertCount()
		self.sdTree_current.resetAllQuadTreeIrradiance()

	
	def saveSDTreeOBJ( self, fileName: str ) -> None:
		"""
			Write KDTree bounding boxes to a Wavefront Obj file
		"""
		self.sdTree_prev.saveOBJ( fileName )


	def aov_names(self):
		# Not really sure what it's for but it's required
		return ["depth.Y"]


	def to_string(self):
		return "path_guiding_integrator"



mi.register_integrator('path_guiding_integrator', lambda props: PathGuidingIntegrator(props))