#include "UnityCG.cginc"
#include "SSRCommon.cginc"
#include "BRDFCommon.cginc"

sampler2D _CameraDepthTexture, _backDepthTexture, _SceneColor_RT, _Noise, _RayCastRT, _RayMask_RT, _CameraMotionVectorsTexture, _CombienReflection_RT, _Resolve_RT,
		  _TemporalBefore, _TemporalAfter, _CameraGBufferTexture0, _CameraGBufferTexture1, _CameraGBufferTexture2, _CameraReflectionsTexture, _forntDepthTexture;

int _NumSteps, _ResolverNum, _TreatBackfaceHitAsMiss, _AllowBackwardsRays, _TraceBehindObjects, _BilateralScale;
										 
float _BRDFBias, _ScreenFade, _TScale, _TResponse, _Thickness, _RayStepSize, _MaxRayTraceDistance;

float4 _ScreenSize, _NoiseSize, _ProjInfo,
	   _ResolveSize, _RayCastSize, _Jitter;

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
	
PixelInput vert(VertexInput v) {
	PixelInput o;
	o.vertex = v.vertex;
	o.uv = v.uv;
	return o;
}

float4 FixedTemporalWhipping(PixelInput i) : SV_Target {	 
	float2 uv = i.uv.xy;
	float4 sceneColor = tex2D(_SceneColor_RT, uv);
	return sceneColor;
}

void LinearRayMarching3DSpace(PixelInput i, out float4 rayCasting : SV_Target0, out half ssrMask : SV_Target1) {	
    float2 uv = i.uv.xy;
	float4 _Screen_TexelSize = float4(1 / _ScreenSize.x, 1 / _ScreenSize.y, _ScreenSize.x, _ScreenSize.y);
    float3 rayPos = GetPosition(_forntDepthTexture, _Screen_TexelSize, _ProjInfo, uv);
    if (rayPos.z < -100.0) {
            rayCasting = 0;
    }

	float roughness = 1 - tex2D(_CameraGBufferTexture1, uv).a;
	roughness = clamp(roughness, 0.014, 1);
    float4 worldNormal = tex2D(_CameraGBufferTexture2, uv) * 2.0 - 1.0;
	
    float2 jitter = tex2Dlod(_Noise, float4((uv + _Jitter.zw) * _RayCastSize.xy / _NoiseSize.xy, 0, -255)).xy;
	float2 Xi = jitter;
	Xi.y = lerp(Xi.y, 0, _BRDFBias);
	float4 H;
	if(roughness > 0.1) {
		H = TangentToWorld(worldNormal, ImportanceSampleGGX(Xi, roughness));
	} else {
		H = float4(worldNormal.xyz, 100);
	}
	
    float3 viewNormal = mul((float3x3)(_WorldToCameraMatrix), H);
    float3 reflectionDir = GetReflectionDir(rayPos, viewNormal);

    if (_AllowBackwardsRays == 0 && reflectionDir.z > 0.0) {
            rayCasting = 0;
    }

    float maxRayTraceDistance = _MaxRayTraceDistance;
    float layerThickness = _Thickness;
    float rayBump = max(-0.01 * rayPos.z, 0.001);
	float constJintter = jitter.x + jitter.y;
	float stepSize = lerp(_RayStepSize, 28, roughness);
    float2 rayHitPixel;
    float3 rayHitPoint;
    float stepsNum;

    bool wasHit = castDenseScreenSpaceRay(  _forntDepthTexture,
                                        	rayPos + (viewNormal) * rayBump,
                                        	reflectionDir,
                                        	_ProjectToPixelMatrix,
                                        	_ScreenSize,
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

    float2 rayUV = rayHitPixel / _ScreenSize;
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
	float rayDepth = 1 - tex2D(_forntDepthTexture, rayUV).r;
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
	float Depth = GetDepth(_forntDepthTexture, uv);
	float3 screenPos = GetSSRScreenPos(uv, Depth);
	float3 worldPos = GetWorlPos(screenPos, _InverseViewProjectionMatrix);
	float3 viewPos = GetViewPos(screenPos, _InverseProjectionMatrix);
	float3 viewDir = GetViewDir(worldPos, viewPos);

	float2 Noise = tex2D(_Noise, (uv + _Jitter.zw) * _ScreenSize.xy / _NoiseSize.xy) * 2 - 1;
	float2x2 OffsetRotationMatrix = float2x2(Noise.x, Noise.y, -Noise.y, Noise.x);
	float NdotV = saturate(dot(worldNormal, -viewDir));
	float coneTangent = lerp(0, roughness * (1 - _BRDFBias), NdotV * sqrt(roughness));
	float ssrMask = tex2D(_RayMask_RT, uv).r;

	int ResolverCont = 0;
	float weightSum, hitZ, weight, hitPDF, intersectionCircleRadius, mip;
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

		intersectionCircleRadius = coneTangent * length(hitUv - uv);
		mip = clamp(log2(intersectionCircleRadius * max(_ResolveSize.x, _ResolveSize.y)), 0, 4);

		sampleColor.rgb = tex2Dlod(_SceneColor_RT, float4(hitUv, 0, mip)).rgb;
		sampleColor.rgb /= 1 + Luminance(sampleColor.rgb);

		reflecttionColor += sampleColor * weight;
    	weightSum += weight;
    }
	reflecttionColor /= weightSum;
	reflecttionColor.rgb /= 1 - Luminance(reflecttionColor.rgb);
	reflecttionColor.a = ssrMask;
	return max(1e-5, reflecttionColor);
}

float4 Temporal (PixelInput i) : SV_Target {	
	float2 uv = i.uv.xy;
	float4 Normal = tex2D(_CameraGBufferTexture2, uv) * 2 - 1;
	float roughness = 1 - tex2D(_CameraGBufferTexture1, uv).a;
	roughness = clamp(roughness, 0.014, 1);
	float3 hitPacked = tex2D(_RayCastRT, uv);

	float2 depthVelocity = tex2D(_CameraMotionVectorsTexture, uv);
	float2 rayVelocity = GetRayMotionVector(1 - hitPacked.z, uv, _InverseViewProjectionMatrix, _LastFrameViewProjectionMatrix, _ViewProjectionMatrix);
	float2 velocity = 0;

	float2 du = float2(1 / _ScreenSize.x, 0);
	float2 dv = float2(0, 1 / _ScreenSize.y);

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

	float TemporalScaler = 0;
	if(roughness > 0.1) {
		TemporalScaler = _TResponse;
		velocity = lerp(depthVelocity, rayVelocity, 0.85);
	} else {
		TemporalScaler = 0.85;
		velocity = lerp(depthVelocity, rayVelocity, Normal.y);
	}

	float4 afterColor = tex2D(_Resolve_RT, uv);
	float4 beforeColor = tex2D(_TemporalBefore, uv - velocity);
	beforeColor = clamp(beforeColor, currentMin, currentMax);
    return lerp(afterColor, beforeColor, saturate(clamp(0, 0.97, TemporalScaler) *  (1 - length(velocity) * 16)));
}

float4 Combien (PixelInput i) : SV_Target {	
	float2 uv = i.uv.xy;
	float AmbientOcclusion = tex2D(_CameraGBufferTexture0, uv).a;
	float4 specularColor = tex2D(_CameraGBufferTexture1, uv);
	float roughness = 1 - specularColor.a;
	roughness = clamp(roughness, 0.014, 1);

	float4 Normal = tex2D(_CameraGBufferTexture2, uv);
	float4 worldNormal = Normal * 2 - 1;
	float Depth = GetDepth(_forntDepthTexture, uv);
	float3 screenPos = GetSSRScreenPos(uv, Depth);
	float3 worldPos = GetWorlPos(screenPos, _InverseViewProjectionMatrix);
	float3 viewDir = GetViewDir(worldPos, _WorldSpaceCameraPos);
	float NoV = saturate(dot(worldNormal, -viewDir));
	float4 PreintegratedGF = float4(EnvBRDFApprox(specularColor.rgb, roughness, NoV), 1);

	float4 finalColor = 0;
	float offsetU = uv.x / _ScreenSize.x;
	float offsetY = uv.y / _ScreenSize.y;
	float4 Sharpness0 = tex2D(_TemporalAfter, uv);
	float4 Sharpness1 = tex2D(_TemporalAfter, uv + float2(offsetU, 0));
	float4 Sharpness2 = tex2D(_TemporalAfter, uv + float2(-offsetU, 0));
	float4 Sharpness3 = tex2D(_TemporalAfter, uv + float2(0, offsetY));
	float4 Sharpness4 = tex2D(_TemporalAfter, uv + float2(0, -offsetY));
	float4 ssrColor = (5 * Sharpness0 - Sharpness1 - Sharpness2 - Sharpness3 - Sharpness4) * AmbientOcclusion;
	float ssrMask = sqr(ssrColor.a);
	
	float4 Cubemap =  tex2D(_CameraReflectionsTexture, uv);
	float4 reflectionColor = lerp(Cubemap, ssrColor * PreintegratedGF, ssrMask);
	float4 sceneColor = tex2Dlod(_SceneColor_RT, float4(uv, 0, 0));
	sceneColor.rgb = max(1e-5, sceneColor.rgb - Cubemap.rgb);

	finalColor = sceneColor + reflectionColor;
	return finalColor;
}

