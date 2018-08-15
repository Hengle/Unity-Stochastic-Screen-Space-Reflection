#include "UnityStandardBRDF.cginc"
#define PI 3.141592

inline half  Pow2(half  x) { return x * x; }
inline half2 Pow2(half2 x) { return x * x; }
inline half3 Pow2(half3 x) { return x * x; }
inline half4 Pow2(half4 x) { return x * x; }

float sqr(float x) {
	return x * x;
}
	
float fract(float x) {
	return x - floor( x );
}

float ComputeDepth(float4 clippos) {
	#if defined(SHADER_TARGET_GLSL) || defined(SHADER_API_GLES) || defined(SHADER_API_GLES3)
		return (clippos.z / clippos.w) * 0.5 + 0.5;
	#else
		return clippos.z / clippos.w;
	#endif
}

float3 GetViewNormal(float3 normal, float4x4 _WToCMatrix) {
	float3 viewNormal =  mul((float3x3)_WToCMatrix, normal.rgb);
	return normalize(viewNormal);
}

float LinearDepthReverseBack(float Depth){
	float z =  ((1 /  Depth) - _ZBufferParams.y) / _ZBufferParams.x;
	#if defined(UNITY_REVERSED_Z)
		z = 1 - z;
	#endif
	return z;
}

float GetDepth(sampler2D tex, float2 uv) {
	float z = tex2Dlod(tex, float4(uv, 0, 0)).r;
	#if defined(UNITY_REVERSED_Z)
		z = 1 - z;
	#endif
	return z;
}

float Get01Depth(sampler2D tex, float2 uv) {
	float z = Linear01Depth(tex2Dlod(tex, float4(uv, 0, 0)).r);
	#if defined(UNITY_REVERSED_Z)
		z = 1 - z;
	#endif
	return z;
}

float GetEyeDepth(sampler2D tex, float2 uv) {
	float z = LinearEyeDepth(tex2Dlod(tex, float4(uv, 0, 0)).r);
	#if defined(UNITY_REVERSED_Z)
		z = 1 - z;
	#endif
	return z;
}

float3 GetSSRScreenPos(float2 uv, float depth) {
	return float3(uv.xy * 2 - 1, depth.r);
}

float3 GetWorlPos(float3 screenPos, float4x4 _InverseViewProjectionMatrix)
{
	float4 worldPos = mul(_InverseViewProjectionMatrix, float4(screenPos, 1));
	return worldPos.xyz / worldPos.w;
}

float3 GetViewRayFromUV(float2 uv, float4x4	_ProjectionMatrix) {
	float4 _CamScreenDir = float4(1 / _ProjectionMatrix[0][0], 1 / _ProjectionMatrix[1][1], 1, 1);
	float3 ray = float3(uv.x * 2 - 1, uv.y * 2 - 1, 1);
	ray *= _CamScreenDir.xyz;
	ray = ray * (_ProjectionParams.z / ray.z);
	return ray;
}

float3 GetViewPos(float3 screenPos, float4x4 _InverseProjectionMatrix) {
	float4 viewPos = mul(_InverseProjectionMatrix, float4(screenPos, 1));
	return viewPos.xyz / viewPos.w;
}
	
float3 GetViewDir(float3 worldPos, float3 ViewPos) {
	return normalize(worldPos - ViewPos);
}

float GetScreenFadeBord (float2 pos, float value) {
	float borderDist = min(1 - max(pos.x, pos.y), min(pos.x, pos.y));
	return saturate(borderDist > value ? 1 : borderDist / value);
}

half2 GetRayMotionVector(float rayDepth, float2 inUV, float4x4 _InverseViewProjectionMatrix, float4x4 _PrevViewProjectionMatrix, float4x4 _ViewProjectionMatrix) {
	float3 screenPos = GetSSRScreenPos(inUV, rayDepth);
	float4 worldPos = float4(GetWorlPos(screenPos, _InverseViewProjectionMatrix),1);

	float4 prevClipPos = mul(_PrevViewProjectionMatrix, worldPos);
	float4 curClipPos = mul(_ViewProjectionMatrix, worldPos);

	float2 prevHPos = prevClipPos.xy / prevClipPos.w;
	float2 curHPos = curClipPos.xy / curClipPos.w;

	float2 vPosPrev = (prevHPos.xy + 1) / 2;
	float2 vPosCur = (curHPos.xy + 1) / 2;
	return vPosCur - vPosPrev;
}

#define DOWNSAMPLE_DEPTH_MODE 2
#define UPSAMPLE_DEPTH_THRESHOLD 1.5f
#define BLUR_DEPTH_FACTOR 0.5
#define GAUSS_BLUR_DEVIATION 1.5        
#define FULL_RES_BLUR_KERNEL_SIZE 7
#define HALF_RES_BLUR_KERNEL_SIZE 5
#define QUARTER_RES_BLUR_KERNEL_SIZE 6
#define GaussianWeight(offset, deviation2) (deviation2.y * exp(-(offset * offset) / (deviation2.x)))

float4 BilateralBlur(float2 uv, sampler2D mainTex, sampler2D depth, float2 screenSize, int2 direction, int kernelRadius) {
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
						
	[unroll] for (int i = -kernelRadius; i < 0; i += 1) {
        float2 offset = (direction * i / screenSize);
        float3 sampleColor = tex2Dlod(mainTex, float4(uv + offset, 0, 0));
        float sampleDepth = (LinearEyeDepth(tex2D(depth,float4(uv + offset, 0, 0))));

		float depthDiff = abs(centerDepth - sampleDepth);
        float dFactor = depthDiff * BLUR_DEPTH_FACTOR;	//Should be 0.5
		float w = exp(-(dFactor * dFactor));

		// gaussian weight is computed from constants only -> will be computed in compile time
		weight = GaussianWeight(i, deviation) * w;

		color += weight * sampleColor;
		weightSum += weight;
	}

	[unroll] for (i = 1; i <= kernelRadius; i += 1) {
		float2 offset = (direction * i / screenSize);
        float3 sampleColor = tex2Dlod(mainTex, float4(uv + offset, 0, 0));
        float sampleDepth = (LinearEyeDepth(tex2Dlod(depth, float4(uv + offset, 0, 0))));

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

/*
/////////////////////////////////////LInear3DTrace/////////////////////////////////////
float4 LinearTraceRay3DSpace(float4x4 CameraProjacted, sampler2D _forntDepthTexture, float thickness, int NumSteps, float2 screenUV, float2 jitter, float3 reflectionDir, float3 ray, float3 ndcUV) {
	float3 dirProject = float3(
		abs(CameraProjacted._m00), 
		abs(CameraProjacted._m11), 
		((_ProjectionParams.z * _ProjectionParams.y) / (_ProjectionParams.y - _ProjectionParams.z)));

	ray = ray / ray.z;
	float3 rayPos = float3(ndcUV.xy * 0.5 + 0.5, ndcUV.z);
	float rayDepth = Linear01Depth(rayPos.z);
	float eyeDepth = LinearEyeDepth(tex2D(_forntDepthTexture, screenUV));
	float3 rayDir = normalize(float3(reflectionDir.xy - ray.xy * reflectionDir.z, reflectionDir.z / eyeDepth) * dirProject);
	rayDir.xy *= 0.5;

	jitter += 0.5;
	float stepSize = 1 / (float)NumSteps;
	stepSize = stepSize * (jitter.x + jitter.y) + stepSize; 

	float mask, forntDepth, backDepth;
	UNITY_LOOP
	for (int i = 0;  i < NumSteps; i++) {
		forntDepth = Linear01Depth(tex2Dlod(_forntDepthTexture, float4(rayPos.xy, 0, 0)));
		backDepth = forntDepth + thickness;
		if (rayDepth > forntDepth && rayDepth < backDepth) {
			mask = 1;
			break;
		}
		rayPos += rayDir * stepSize;
		rayDepth = Linear01Depth(1 - rayPos.z);
	}
	return float4(rayPos, mask);
}

void LinearRayMarching3DSpace(PixelInput i, out float4 rayCasting : SV_Target0, out half rayMask : SV_Target1) {	
	float2 uv = i.uv.xy;
	float4 normal = tex2D(_CameraGBufferTexture2, uv);
	float roughness = 1 - tex2D(_CameraGBufferTexture1, uv).a;
	roughness = clamp(roughness, 0.014, 1);

	float Thickness = _Thickness;
	float4 worldNormal = normal * 2 - 1;
	
	float depth = GetDepth(_CameraDepthTexture, uv);
	float3 ndcUV = float3(uv * 2 - 1, depth);

	float3 ray = GetViewPos(ndcUV, _InverseProjectionMatrix);
	float3 worldPos = GetWorlPos(ndcUV, _InverseViewProjectionMatrix);
	float3 viewDir = GetViewDir(worldPos, _WorldSpaceCameraPos);

	float2 jitter = tex2Dlod(_Noise, float4((uv + _JitterSizeAndOffset.zw) * _RayCastSize.xy / _NoiseSize.xy, 0, -255)).xy; 
	float2 Xi = jitter;
	Xi.y = lerp(Xi.y, 0, _BRDFBias);
	float4 H;
	if(roughness > 0.1) {
		H = TangentToWorld(worldNormal, ImportanceSampleGGX(Xi, roughness));
	} else {
		H = worldNormal;
	}
	float3 reflectionDir = reflect(viewDir, H.xyz);
	reflectionDir = normalize(mul((float3x3)_WorldToCameraMatrix, reflectionDir));

	float4 rayCast = LinearTraceRay3DSpace(_ProjectionMatrix, _CameraDepthTexture, Thickness, _NumSteps, uv, jitter, reflectionDir, ray, ndcUV);
	rayCasting = float4(rayCast.rgb, H.a);
	rayMask = rayCast.a;
}
*/

/////////////////////////////////////LInear2DTrace/////////////////////////////////////
float distanceSquared(float2 A, float2 B) {
        A -= B;
        return dot(A, A);
}

float distanceSquared(float3 A, float3 B) {
        A -= B;
        return dot(A, A);
}

void swap(inout float v0, inout float v1) {
        float temp = v0;
        v0 = v1;
        v1 = temp;
}


bool isIntersecting(float rayZMin, float rayZMax, float sceneZ, float layerThickness) {
    return (rayZMax >= sceneZ - layerThickness) && (rayZMin <= sceneZ);
}

void rayIterations(sampler2D forntDepth, in bool traceBehindObjects, inout float2 P, inout float stepDirection, inout float end, inout int stepCount, inout int maxSteps, inout bool intersecting,
        inout float sceneZ, inout float2 dP, inout float3 Q, inout float3 dQ, inout float k, inout float dk,
        inout float rayZMin, inout float rayZMax, inout float prevZMaxEstimate, inout bool permute, inout float2 hitPixel,
        inout float2 invSize, inout float layerThickness)
{
        bool stop = intersecting;
        UNITY_LOOP
        for (; (P.x * stepDirection) <= end && stepCount < maxSteps && !stop; P += dP, Q.z += dQ.z, k += dk, stepCount += 1) {
                rayZMin = prevZMaxEstimate;
                rayZMax = (dQ.z * 0.5 + Q.z) / (dk * 0.5 + k);
                prevZMaxEstimate = rayZMax;

                if (rayZMin > rayZMax) {
                        swap(rayZMin, rayZMax);
                }
                
                hitPixel = permute ? P.yx : P;
                sceneZ = tex2Dlod(forntDepth, float4(hitPixel * invSize,0,0)).r;
                sceneZ = -LinearEyeDepth(sceneZ);
                bool isBehind = (rayZMin <= sceneZ);
                intersecting = isBehind && (rayZMax >= sceneZ - layerThickness);
                stop = traceBehindObjects ? intersecting : isBehind;
        } 
        P -= dP, Q.z -= dQ.z, k -= dk;
}

bool castDenseScreenSpaceRay
   (sampler2D forntDepth,
    float3          csOrigin,
    float3          csDirection,
    float4x4        projectToPixelMatrix,
    float2          csZBufferSize,
    float3          clipInfo,
    float           jitterFraction,
    int             maxSteps,
    float           layerThickness,
    float           maxRayTraceDistance,
    in out float2      hitPixel,
    int             stepRate,
    bool            traceBehindObjects,
    in out float3      csHitPoint,
    in out float       stepCount) {

    float2 invSize = float2(1.0 / csZBufferSize.x, 1.0 / csZBufferSize.y);
    hitPixel = float2(-1, -1);

    float nearPlaneZ = -0.01;
    float rayLength = ((csOrigin.z + csDirection.z * maxRayTraceDistance) > nearPlaneZ) ? ((nearPlaneZ - csOrigin.z) / csDirection.z) : maxRayTraceDistance;
    float3 csEndPoint = csDirection * rayLength + csOrigin;
    float4 H0 = mul(projectToPixelMatrix, float4(csOrigin, 1.0));
    float4 H1 = mul(projectToPixelMatrix, float4(csEndPoint, 1.0));
    float k0 = 1.0 / H0.w;
    float k1 = 1.0 / H1.w;
    float2 P0 = H0.xy * k0;
    float2 P1 = H1.xy * k1;
    float3 Q0 = csOrigin * k0;
    float3 Q1 = csEndPoint * k1;

#if 1
    float yMax = csZBufferSize.y - 0.5;
    float yMin = 0.5;
    float xMax = csZBufferSize.x - 0.5;
    float xMin = 0.5;
    float alpha = 0.0;

    if (P1.y > yMax || P1.y < yMin) {
        float yClip = (P1.y > yMax) ? yMax : yMin;
        float yAlpha = (P1.y - yClip) / (P1.y - P0.y); 
        alpha = yAlpha;
    }
    if (P1.x > xMax || P1.x < xMin) {
        float xClip = (P1.x > xMax) ? xMax : xMin;
        float xAlpha = (P1.x - xClip) / (P1.x - P0.x); 
        alpha = max(alpha, xAlpha);
    }

    P1 = lerp(P1, P0, alpha);
    k1 = lerp(k1, k0, alpha);
    Q1 = lerp(Q1, Q0, alpha);
#endif
    P1 = (distanceSquared(P0, P1) < 0.0001) ? P0 + float2(0.01, 0.01) : P1;
    float2 delta = P1 - P0;
    bool permute = false;

    if (abs(delta.x) < abs(delta.y)) {
        permute = true;
        delta = delta.yx;
        P1 = P1.yx;
        P0 = P0.yx;
    }

    float stepDirection = sign(delta.x);
    float invdx = stepDirection / delta.x;
    float2 dP = float2(stepDirection, invdx * delta.y);
    float3 dQ = (Q1 - Q0) * invdx;
    float   dk = (k1 - k0) * invdx;
    dP *= stepRate;
    dQ *= stepRate;
    dk *= stepRate;
    P0 += dP * jitterFraction;
    Q0 += dQ * jitterFraction;
    k0 += dk * jitterFraction;
    float3 Q = Q0;
    float  k = k0;
    float prevZMaxEstimate = csOrigin.z;
    stepCount = 0.0;
    float rayZMax = prevZMaxEstimate, rayZMin = prevZMaxEstimate;
    float sceneZ = 100000;
    float end = P1.x * stepDirection;
    bool intersecting = isIntersecting(rayZMin, rayZMax, sceneZ, layerThickness);
    float2 P = P0;
    int originalStepCount = 0;

    rayIterations(forntDepth, traceBehindObjects, P, stepDirection, end,  originalStepCount,  maxSteps, intersecting,
         sceneZ, dP, Q, dQ,  k,  dk,
         rayZMin,  rayZMax,  prevZMaxEstimate, permute, hitPixel,
         invSize,  layerThickness);

    stepCount = originalStepCount;
    Q.xy += dQ.xy * stepCount;
    csHitPoint = Q * (1.0 / k);
    return intersecting;
}

float3 ReconstructCSPosition(float4 _MainTex_TexelSize, float4 _ProjInfo, float2 S, float z) {
        float linEyeZ = -LinearEyeDepth(z);
        return float3(( (( S.xy * _MainTex_TexelSize.zw) ) * _ProjInfo.xy + _ProjInfo.zw) * linEyeZ, linEyeZ);
}

float3 GetPosition(sampler2D depth, float4 _MainTex_TexelSize, float4 _ProjInfo, float2 ssP) {
        float3 P;
        P.z = SAMPLE_DEPTH_TEXTURE(depth, ssP.xy).r;
        P = ReconstructCSPosition(_MainTex_TexelSize, _ProjInfo, float2(ssP), P.z);
        return P;
}

float3 GetReflectionDir(float3 csPosition, float3 csN) {
        float3 csE = -normalize(csPosition.xyz);
        float cos_o = dot(csN, csE);
        float3 c_mi = normalize((csN * (2.0 * cos_o)) - csE);
        return c_mi;
}

/*
//////////////////////////////////LinearDepth3DTrace/////////////////////////////////////
float4 LinearTraceRay3DSpace(float4x4 CameraProjacted, sampler2D _forntDepthTexture, float thickness, int NumSteps, float2 screenUV, float2 jitter, float3 reflectionDir, float3 ray, float3 ndcUV) {
	float3 dirProject = float3(
		abs(CameraProjacted._m00), 
		abs(CameraProjacted._m11), 
		((_ProjectionParams.z * _ProjectionParams.y) / (_ProjectionParams.y - _ProjectionParams.z)));

	ray = ray / ray.z;
	float3 rayPos = float3(ndcUV.xy * 0.5 + 0.5, ndcUV.z);
	float rayDepth = Linear01Depth(rayPos.z);
	float eyeDepth = LinearEyeDepth(tex2D(_forntDepthTexture, screenUV));
	float3 rayDir = normalize(float3(reflectionDir.xy - ray.xy * reflectionDir.z, reflectionDir.z / eyeDepth) * dirProject);
	rayDir.xy *= 0.5;

	jitter += 0.5;
	float stepSize = 1 / (float)NumSteps;
	stepSize = stepSize * (jitter.x + jitter.y) + stepSize; 

	float mask, forntDepth, backDepth;
	UNITY_LOOP
	for (int i = 0;  i < NumSteps; i++) {
		forntDepth = Linear01Depth(tex2Dlod(_forntDepthTexture, float4(rayPos.xy, 0, 0)));
		backDepth = forntDepth + thickness;
		if (rayDepth > forntDepth && rayDepth < backDepth) {
			mask = 1;
			break;
		}
		rayPos += rayDir * stepSize;
		rayDepth = Linear01Depth(1 - rayPos.z);
	}
	return float4(rayPos, mask);
}
*/