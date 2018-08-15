Shader "Hidden/StochasticScreenSpaceReflection" {
	CGINCLUDE
		#include "SSRPassCommon.cginc"
	ENDCG

	SubShader {
		ZTest Always 
		Cull Off 
		ZWrite Off

		Pass {
			Name"RayCastingPass"
			CGPROGRAM
				#pragma vertex vert
				#pragma fragment LinearRayMarching3DSpace
			ENDCG
		} Pass {
			Name"ResolvePass"
			CGPROGRAM
				#pragma vertex vert
				#pragma fragment Resolve
			ENDCG
		} Pass {
			Name"TemporalPass"
			CGPROGRAM
				#pragma vertex vert
				#pragma fragment Temporal
			ENDCG
		} Pass {
			Name"CombienPass"
			CGPROGRAM
				#pragma vertex vert
				#pragma fragment Combien
			ENDCG
		}
		
	}
}
