Shader "Hidden/BackFaceShader"{
	Properties{
	}
	SubShader{
    	Tags { "RenderType"="Opaque" }
        LOD 100
        Cull Front

    	Pass{
        	CGPROGRAM

			#pragma vertex vert
			#pragma fragment frag
			#include "UnityCG.cginc"

        	struct v2f {
        	float4 position : POSITION;
        	float4 linearDepth : TEXCOORD0;
    	};

    	v2f vert(appdata_base v) {
        	v2f output;
        	output.position = UnityObjectToClipPos(v.vertex);
        	output.linearDepth = float4(0.0, 0.0, COMPUTE_DEPTH_01, 0.0);
        	return output;
    	}

    	float4 frag(v2f input) : COLOR{
        	return input.linearDepth.z;
    	}
        ENDCG
        }
    }
}
