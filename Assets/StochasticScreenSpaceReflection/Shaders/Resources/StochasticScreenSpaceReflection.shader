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
				#pragma vertex vertMRT
				#pragma fragment LinearRayMarching3DSpace
			ENDCG
		} Pass {
			Name"ResolvePass"
			CGPROGRAM
				#pragma vertex vertSRT
				#pragma fragment Resolve
			ENDCG
		} Pass {
			Name"TemporalPass"
			CGPROGRAM
				#pragma vertex vertSRT
				#pragma fragment Temporal
			ENDCG
		} Pass {
			Name"CombienPass"
			CGPROGRAM
				#pragma vertex vertSRT
				#pragma fragment Combien
			ENDCG
		}
		Pass//4
		{
			Name"HoriBlur"
			CGPROGRAM
				#pragma vertex vertSRT
				#pragma fragment blurHori
			ENDCG
		}
		Pass//5
		{
			Name"VertBlur"
			CGPROGRAM
				#pragma vertex vertSRT
				#pragma fragment blurVert
			ENDCG
		}
	}
}
