from __future__ import annotations as __annotations__ # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi
if __name__ == '__main__':
	mi.set_variant('cuda_ad_rgb')

# from common import *

import numpy as np

def mis_weight(pdf_a, pdf_b):
	"""
		Compute the Multiple Importance Sampling (MIS) weight given the densities
		of two sampling strategies according to the power heuristic.
	"""
	a2 = dr.sqr(pdf_a)
	result = dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0)
	result[ dr.isnan( result ) ] = 0
	return result
	

class PathTracingIntegrator(mi.SamplingIntegrator):

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

		# For variance calculation
		self.sumL = mi.Spectrum(0)
		self.sumL2 = mi.Spectrum(0)


	def resetVarianceCounter( self ) -> None:
		self.sumL = mi.Spectrum(0)
		self.sumL2 = mi.Spectrum(0)


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

		# Variables caching information from the previous bounce
		prev_si = dr.zeros(mi.SurfaceInteraction3f)
		prev_bsdf_pdf = mi.Float(1.0)
		prev_bsdf_delta = mi.Bool(True)


		# Record the following loop in its entirety
		loop = mi.Loop(name="Path Tracing with NEE", 
			state=lambda: (
				sampler, ray, depth, L, throughput_weight, ior, active,
				prev_si, prev_bsdf_pdf, prev_bsdf_delta
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

			# 	MIS 
			mis_em = dr.select( ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em) )

			# 	evaluate NEE
			Lr_dir = throughput_weight * mis_em * bsdf_value_em * em_weight

			# 
			# 	Accumulate Radiance
			# 
			L += Le + Lr_dir


			# 
			# 	Sample next outgoing ray
			# 
			bsdf_sample, bsdf_weight = bsdf.sample( bsdf_ctx, si, sampler.next_1d( active_next ), sampler.next_2d(active_next), active_next )


			# 
			# 	Update loop variables based on current interaction
			# 
			ray = si.spawn_ray(si.to_world( bsdf_sample.wo ))
			ior *= bsdf_sample.eta
			throughput_weight *= bsdf_weight

            # Information about the current vertex needed by the next iteration
			prev_si = si
			prev_bsdf_pdf = bsdf_sample.pdf # bsdf_sample.pdf
			prev_bsdf_delta = mi.has_flag( bsdf_sample.sampled_type, mi.BSDFFlags.Delta )


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
	

	def aov_names(self):
		# Not really sure what it's for but it's required
		return ["depth.Y"]


	def to_string(self):
		return "path_tracing_integrator_py"



mi.register_integrator('path_tracing_integrator_py', lambda props: PathTracingIntegrator(props))