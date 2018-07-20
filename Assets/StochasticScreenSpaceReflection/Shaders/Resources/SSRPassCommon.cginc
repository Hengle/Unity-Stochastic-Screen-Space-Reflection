#include "UnityCG.cginc"
#include "SSRCommon.cginc"
#include "NoiseCommon.cginc"
#include "BRDFCommon.cginc"

sampler2D _CameraDepthTexture, _backDepthTexture, _SceneColor_RT, _Noise, _RayCastRT, _RayMask_RT, _CameraMotionVectorsTexture, _CombienReflection_RT,
		  _TemporalBefore, _TemporalAfter, _CameraGBufferTexture0, _CameraGBufferTexture1, _CameraGBufferTexture2, _CameraReflectionsTexture, _Resolve_RT;

int _NumSteps, _ResolverNum, _TreatBackfaceHitAsMiss, _AllowBackwardsRays, _TraceBehindObjects;
										 
float _BRDFBias, _ScreenFade, _TScale, _TResponse, _Thickness, _RayStepSize, _MaxRayTraceDistance;

float4 _NoiseSize, _ProjInfo,
	   _ResolveSize, _RayCastSize, _JitterSizeAndOffset;

float3 _CameraClipInfo;
																					
float4x4 _ProjectionMatrix, _InverseProjectionMatrix, _ViewProjectionMatrix, _InverseViewProjectionMatrix,
		 _LastFrameViewProjectionMatrix, _WorldToCameraMatrix, _CameraToWorldMatrix, _ProjectToPixelMatrix;


struct VertexInput {
	float4 vertex : POSITION;
	float4 uv : TEXCOORD0;
};

struct PixelInput {
	float4 vertex : SV_POSITION;
	float4 uv : TEXCOORD0;
};
	
PixelInput vertMRT(VertexInput v) {
	PixelInput o;
	o.vertex = v.vertex;
	o.uv = v.uv;
	return o;
}

PixelInput vertSRT(VertexInput v) {
	PixelInput o;
	o.vertex = UnityObjectToClipPos(v.vertex);
	o.uv = v.uv;
	return o;
}
        #define DOWNSAMPLE_DEPTH_MODE 2
        #define UPSAMPLE_DEPTH_THRESHOLD 1.5f
        #define BLUR_DEPTH_FACTOR 0.5
        #define GAUSS_BLUR_DEVIATION 1.5        
        #define FULL_RES_BLUR_KERNEL_SIZE 7
        #define HALF_RES_BLUR_KERNEL_SIZE 5
        #define QUARTER_RES_BLUR_KERNEL_SIZE 6
#define GaussianWeight(offset, deviation2) (deviation2.y * exp(-(offset * offset) / (deviation2.x)))

float4 BilateralBlur(float2 uv, sampler2D mainTex, sampler2D depth, const int2 direction, const int kernelRadius, float screenSize)
		{
			//const float deviation = kernelRadius / 2.5;
			const float dev = kernelRadius / GAUSS_BLUR_DEVIATION; // make it really strong
			const float dev2 = dev * dev * 2;
			const float2 deviation = float2(dev2, 1.0f / (dev2 * PI));
			float4 centerColor = tex2D(mainTex, uv);
			float3 color = centerColor.xyz;
			//return float4(color, 1);
			float centerDepth = (LinearEyeDepth(tex2D(depth, uv)));

			float weightSum = 0;

			// gaussian weight is computed from constants only -> will be computed in compile time
            float weight = GaussianWeight(0, deviation);
			color *= weight;
			weightSum += weight;
						
			[unroll] for (int i = -kernelRadius; i < 0; i += 1)
			{
                float2 offset = (direction * i) / screenSize;
                float3 sampleColor = tex2D(mainTex, uv + offset);
                float sampleDepth = (LinearEyeDepth(tex2D(depth,uv + offset)));

				float depthDiff = abs(centerDepth - sampleDepth);
                float dFactor = depthDiff * BLUR_DEPTH_FACTOR;	//Should be 0.5
				float w = exp(-(dFactor * dFactor));

				// gaussian weight is computed from constants only -> will be computed in compile time
				weight = GaussianWeight(i, deviation) * w;

				color += weight * sampleColor;
				weightSum += weight;
			}

			[unroll] for (i = 1; i <= kernelRadius; i += 1)
			{
				float2 offset = (direction * i) / screenSize;
                float3 sampleColor = tex2D(mainTex, uv + offset);
                float sampleDepth = (LinearEyeDepth(tex2D(depth, uv + offset)));

				float depthDiff = abs(centerDepth - sampleDepth);
                float dFactor = depthDiff * BLUR_DEPTH_FACTOR;	//Should be 0.5
				float w = exp(-(dFactor * dFactor));
				weight = GaussianWeight(i, deviation) * w;
				color += weight * sampleColor;
				weightSum += weight;
			}

			color /= weightSum;
			return float4(color, centerColor.w);
		}

float4 FixedTemporalWhipping(PixelInput i) : SV_Target {	 
	float2 uv = i.uv.xy;
	float4 sceneColor = tex2D(_SceneColor_RT, uv);
	return sceneColor;
}

void LinearRayMarching3DSpace(PixelInput i, out float4 rayCasting : SV_Target0, out half ssrMask : SV_Target1) {	
    float2 uv = i.uv.xy;
	float4 _Screen_TexelSize = float4(1 / _ScreenParams.x, 1 / _ScreenParams.y, _ScreenParams.x, _ScreenParams.y);
    float3 rayPos = GetPosition(_CameraDepthTexture, _Screen_TexelSize, _ProjInfo, uv);
    if (rayPos.z < -100.0) {
            rayCasting = 0;
    }

	float roughness = 1 - tex2D(_CameraGBufferTexture1, uv).a;
	roughness = clamp(roughness, 0.014, 1);
    float4 worldNormal = tex2D(_CameraGBufferTexture2, uv) * 2.0 - 1.0;
	
    float2 jitter = tex2Dlod(_Noise, float4((uv + _JitterSizeAndOffset.zw) * _RayCastSize.xy / _NoiseSize.xy, 0, -255)).xy;
	float2 Xi = jitter;
	Xi.y = lerp(Xi.y, 0, _BRDFBias);
	float4 H;
	if(roughness > 0.1) {
		H = TangentToWorld(worldNormal, ImportanceSampleGGX(Xi, roughness));
	} else {
		H = worldNormal;
	}
	
    float3 viewNormal = mul((float3x3)(_WorldToCameraMatrix), H);
    float3 reflectionDir = csMirrorVector(rayPos, viewNormal);

    if (_AllowBackwardsRays == 0 && reflectionDir.z > 0.0) {
            rayCasting = 0;
    }

    float maxRayTraceDistance = _MaxRayTraceDistance;
    float layerThickness = _Thickness;
    float rayBump = max(-0.01 * rayPos.z, 0.001);
	float constJintter = jitter.x + jitter.y;
	//float stepSize = _RayStepSize * constJintter + _RayStepSize;
	float stepSize = _RayStepSize;
    float2 rayHitPixel;
    float3 rayHitPoint;
    float stepsNum;

    bool wasHit = castDenseScreenSpaceRay(  _CameraDepthTexture,
                                        	rayPos + (viewNormal) * rayBump,
                                        	reflectionDir,
                                        	_ProjectToPixelMatrix,
                                        	_ScreenParams,
                                        	_CameraClipInfo,
                                        	constJintter,
                                        	_NumSteps,
                                        	layerThickness,
                                        	maxRayTraceDistance,
                                        	rayHitPixel,
                                        	stepSize,
                                        	_TraceBehindObjects == 1,
                                        	rayHitPoint,
                                        	stepsNum);

    float2 rayUV = rayHitPixel / _ScreenParams;
    float rayDist = dot(rayHitPoint - rayPos, reflectionDir);
    float rayMask = 0.0;
    if (wasHit) {
            rayMask = sqr(1.0 - max(2.0 * float(stepsNum) / float(_NumSteps) - 1.0, 0.0));
            rayMask *= clamp(((_MaxRayTraceDistance - rayDist)), 0.0, 1.0);
            if (_TreatBackfaceHitAsMiss > 0) {
                    float3 wsHitNormal = tex2Dlod(_CameraGBufferTexture2, float4(rayUV, 0, 0)).rgb * 2.0 - 1.0;
                    float3 wsRayDirection = mul(_CameraToWorldMatrix, float4(reflectionDir, 0)).xyz;

                    if (dot(wsHitNormal, wsRayDirection) > 0) {
                            rayMask = 0.0;
                    }
            }
    }
	float rayDepth = 1 - tex2D(_CameraDepthTexture, rayUV).r;
	float screnFade = GetScreenFadeBord(rayUV, _ScreenFade);
	rayMask = sqr(rayMask) * screnFade;

	rayCasting = float4(rayUV, rayDepth, H.a);
	ssrMask = rayMask;
}

static const float2 offset[4] = {float2(0, 0), float2(2, -2), float2(-2, -2), float2(0, 2)};
float4 Resolve(PixelInput i) : SV_Target {
	float2 uv = i.uv.xy;
	float4 specularColor = tex2D(_CameraGBufferTexture1, uv);
	float roughness = 1 - specularColor.a;
	roughness = clamp(roughness, 0.014, 1);
	float4 Normal = tex2D(_CameraGBufferTexture2, uv);
	float4 worldNormal = Normal * 2 - 1;
	float3 viewNormal = GetViewNormal (worldNormal, _WorldToCameraMatrix);
	float Depth = GetDepth(_CameraDepthTexture, uv);
	float3 screenPos = GetSSRScreenPos(uv, Depth);
	float3 viewPos = GetViewPos(screenPos, _InverseProjectionMatrix);

	float3 worldPos = GetWorlPos(screenPos, _InverseViewProjectionMatrix);
	float3 viewDir = GetViewDir(worldPos, _WorldSpaceCameraPos);
	float NoV = saturate(dot(worldNormal, -viewDir));
	float4 PreintegratedGF = float4(EnvBRDFApprox(specularColor.rgb, roughness, NoV), 1);

	float2 Noise = tex2D(_Noise, (uv + _JitterSizeAndOffset.zw) * _ScreenParams.xy / _NoiseSize.xy) * 2 - 1;
	float2x2 OffsetRotationMatrix = float2x2(Noise.x, Noise.y, -Noise.y, Noise.x);
	float ssrMask = tex2D(_RayMask_RT, uv).r;

	int ResolverCont = 0;
	float weightSum, hitZ, weight, hitPDF;
	float2 offsetUV, neighborUv, hitUv;
	float3 hitViewPos;
	float4 hitPacked, sampleColor, reflecttionColor;

	if(roughness > 0.1) {
		ResolverCont = _ResolverNum;
	} else {
		ResolverCont = 1;
	}

	UNITY_LOOP
    for(int i = 0; i < ResolverCont; i++) {
		offsetUV = offset[i] * (1 / _ResolveSize.xy);
		offsetUV = mul(OffsetRotationMatrix, offsetUV);
		neighborUv = uv + offsetUV;

        hitPacked = tex2Dlod(_RayCastRT, float4(neighborUv, 0, 0));
		hitPDF = hitPacked.a;
        hitUv = hitPacked.xy;
        hitZ = hitPacked.z;
		hitViewPos = GetViewPos(GetSSRScreenPos(hitUv, hitZ), _InverseProjectionMatrix);

		weight = BRDF_UE4(normalize(-viewPos), normalize(hitViewPos - viewPos), viewNormal, max(0.014, roughness)) / max(1e-5, hitPDF);
		weight = weight;

		sampleColor.rgb = tex2Dlod(_SceneColor_RT, float4(hitUv, 0, 0)).rgb;
		sampleColor.rgb /= 1 + Luminance(sampleColor.rgb);

		reflecttionColor += sampleColor * weight;
    	weightSum += weight;
    }
	reflecttionColor /= weightSum;
	reflecttionColor.rgb /= 1 - Luminance(reflecttionColor.rgb);
	reflecttionColor.rgb *= PreintegratedGF;
	reflecttionColor.a = ssrMask;
	return max(1e-5, reflecttionColor);
}

float4 Temporal (PixelInput i) : SV_Target {	
	float2 uv = i.uv.xy;
	float roughness = (1 - tex2D(_CameraGBufferTexture1, uv).a);
	roughness = clamp(roughness, 0.014, 1);
	float3 hitPacked = tex2D(_RayCastRT, uv);
	float2 velocity = GetRayMotionVector(1 - hitPacked.z, uv, _InverseViewProjectionMatrix, _LastFrameViewProjectionMatrix, _ViewProjectionMatrix);
	//float2 velocity = tex2D(_CameraMotionVectorsTexture, uv);

	float2 du = float2(1 / _ScreenParams.x, 0);
	float2 dv = float2(0, 1 / _ScreenParams.y);

	float4 currentTopLeft = tex2D(_Resolve_RT, uv.xy - dv - du);
	float4 currentTopCenter = tex2D(_Resolve_RT, uv.xy - dv);
	float4 currentTopRight = tex2D(_Resolve_RT, uv.xy - dv + du);
	float4 currentMiddleLeft = tex2D(_Resolve_RT, uv.xy - du);
	float4 currentMiddleCenter = tex2D(_Resolve_RT, uv.xy);
	float4 currentMiddleRight = tex2D(_Resolve_RT, uv.xy + du);
	float4 currentBottomLeft = tex2D(_Resolve_RT, uv.xy + dv - du);
	float4 currentBottomCenter = tex2D(_Resolve_RT, uv.xy + dv);
	float4 currentBottomRight = tex2D(_Resolve_RT, uv.xy + dv + du);

	float4 currentMin = min(currentTopLeft, min(currentTopCenter, min(currentTopRight, min(currentMiddleLeft, min(currentMiddleCenter, min(currentMiddleRight, min(currentBottomLeft, min(currentBottomCenter, currentBottomRight))))))));
	float4 currentMax = max(currentTopLeft, max(currentTopCenter, max(currentTopRight, max(currentMiddleLeft, max(currentMiddleCenter, max(currentMiddleRight, max(currentBottomLeft, max(currentBottomCenter, currentBottomRight))))))));

	float4 center = (currentMin + currentMax) * 0.5;
	currentMin = (currentMin - center) * _TScale + center;
	currentMax = (currentMax - center) * _TScale + center;

	float4 afterColor = tex2D(_Resolve_RT, uv);
	float4 beforeColor = tex2D(_TemporalBefore, uv - velocity);
	beforeColor = clamp(beforeColor, currentMin, currentMax);
    return lerp(afterColor, beforeColor, saturate(clamp(0, 0.97, _TResponse) *  (1 - length(velocity) * 8)));
}

float4 blurHori(PixelInput i) : SV_Target
{
	return BilateralBlur(i.uv, _TemporalAfter, _CameraDepthTexture, int2(1,0), 7, _ScreenParams.x);
}

float4 blurVert(PixelInput i) : SV_Target
{
	return BilateralBlur(i.uv, _Resolve_RT, _CameraDepthTexture, int2(0,1), 7, _ScreenParams.y);
}

float4 Combien (PixelInput i) : SV_Target {	
	float2 uv = i.uv.xy;
	float AmbientOcclusion = tex2D(_CameraGBufferTexture0, uv).a;
	float roughness = 1 - tex2D(_CameraGBufferTexture1, uv).a;
	roughness = clamp(roughness, 0.014, 1);

	float4 finalColor = 0;
	float4 ssrColor = tex2D(_TemporalAfter, uv) * AmbientOcclusion;
	float ssrMask = sqr(ssrColor.a);
	float4 Cubemap =  tex2D(_CameraReflectionsTexture, uv);
	float4 reflectionColor = lerp(ssrColor, Cubemap, saturate(roughness - 0.25));
	reflectionColor = lerp(Cubemap, reflectionColor, ssrMask);

	float4 sceneColor = tex2D(_SceneColor_RT, uv);
	sceneColor.rgb = max(1e-5, sceneColor.rgb - Cubemap.rgb);
		finalColor = sceneColor + reflectionColor;

	return finalColor;
}

