Shader "Hidden/SSR" {

	CGINCLUDE
		#include "UnityCG.cginc"
                #include "TraceLiberary.cginc"

                int _NumSteps, _TreatBackfaceHitAsMiss, _AllowBackwardsRays, _TraceBehindObjects;

                float _RayStepSize, _MaxRayTraceDistance, _Thickness;

                float4 _ScreenSize, _Jitter, _NoiseSize, _ProjInfo, _CameraClipInfo;

                sampler2D _CameraDepthTexture, _CameraGBufferTexture0, _CameraGBufferTexture1, _CameraGBufferTexture2, _SceneColor, _SceneDepth, _Noise;

                float4x4 _ProjectionMatrix, _InverseProjectionMatrix, _ViewProjectionMatrix, _InverseViewProjectionMatrix,
                         _WorldToCameraMatrix, _CameraToWorldMatrix, _ProjectToPixelMatrix;

		struct appdata {
			float4 vertex : POSITION;
			float4 uv : TEXCOORD0;
		};

		struct v2f {
                        float4 vertex : SV_POSITION;
                        float2 uv : TEXCOORD0;
		};

		v2f vert (appdata v) {
			v2f o;
			o.vertex = UnityObjectToClipPos(v.vertex);
			o.uv = v.uv.xy;
			return o;
		}

                float4 LinearTrace2D(v2f i) : SV_Target {
                        float2 uv = i.uv.xy;
	                float4 _Screen_TexelSize = float4(1 / _ScreenSize.x, 1 / _ScreenSize.y, _ScreenSize.x, _ScreenSize.y);
                        float3 rayPos = GetPosition(_SceneDepth, _Screen_TexelSize, _ProjInfo, uv);
                        if (rayPos.z < -100.0) {
                                return 0;
                        }

                        float4 worldNormal = tex2D(_CameraGBufferTexture2, uv) * 2.0 - 1.0;
                        float2 jitter = tex2Dlod(_Noise, float4((uv + _Jitter.zw) * _ScreenSize / _NoiseSize.xy, 0, -255)).xy;
                        float3 viewNormal = mul((float3x3)(_WorldToCameraMatrix), worldNormal);
                        float3 reflectionDir = GetReflectionDir(rayPos, viewNormal);

                        if (_AllowBackwardsRays == 0 && reflectionDir.z > 0.0) {
                                return 0;
                        }

                        float maxRayTraceDistance = _MaxRayTraceDistance;
                        float layerThickness = _Thickness;
                        float rayBump = max(-0.01 * rayPos.z, 0.001);
                        float2 rayHitPixel;
                        float3 rayHitPoint;
                        float stepsNum;

                        bool wasHit = castDenseScreenSpaceRay(  _SceneDepth,
                                                                rayPos + (viewNormal) * rayBump,
                                                                reflectionDir,
                                                                _ProjectToPixelMatrix,
                                                                _ScreenSize.xy,
                                                                _CameraClipInfo,
                                                                jitter.x + jitter.y,
                                                                _NumSteps,
                                                                layerThickness,
                                                                maxRayTraceDistance,
                                                                rayHitPixel,
                                                                _RayStepSize,
                                                                _TraceBehindObjects == 1,
                                                                rayHitPoint,
                                                                stepsNum);

                        float2 reflectionUV = rayHitPixel / _ScreenSize.xy;
                        float rayDist = dot(rayHitPoint - rayPos, reflectionDir);
                        float rayMask = 0.0;
                        if (wasHit) {
                                rayMask = sqr(1.0 - max(2.0 * float(stepsNum) / float(_NumSteps) - 1.0, 0.0));
                                rayMask *= clamp(((_MaxRayTraceDistance - rayDist)), 0.0, 1.0);
                                if (_TreatBackfaceHitAsMiss > 0) {
                                        float3 wsHitNormal = tex2Dlod(_CameraGBufferTexture2, float4(reflectionUV, 0, 0)).rgb * 2.0 - 1.0;
                                        float3 wsRayDirection = mul(_CameraToWorldMatrix, float4(reflectionDir, 0)).xyz;
                                        if (dot(wsHitNormal, wsRayDirection) > 0) {
                                                rayMask = 0.0;
                                        }
                                }
                        }
                        float screenFade = LinearTrace3D_Alpha(reflectionUV, 0.1);
                        rayMask *= screenFade;

                        float4 reflectionColor = tex2D(_SceneColor, reflectionUV) * rayMask;
                        float4 sceneColor = tex2Dlod(_SceneColor, float4(uv, 0, 0));

                        return float4(reflectionUV, 1, rayMask);
                        //return reflectionColor;
                        //return sceneColor;
                        //return float4(rayPos + (viewNormal) * rayBump, 1);
                }

                float4 HiZTrace3D(v2f i) : SV_Target {
                        float4 finalColor = 0;
                        float2 uv = i.uv.xy;
                        float2 jitter = tex2Dlod(_Noise, float4((uv + _Jitter.zw) * _ScreenSize / _NoiseSize.xy, 0, -255)).xy;
                        float4 worldNormal = tex2D(_CameraGBufferTexture2, uv) * 2.0 - 1.0;
                        float3 viewNormal = mul((float3x3)(_WorldToCameraMatrix), worldNormal);
                        float Depth = tex2Dlod(_SceneDepth, float4(uv, 0, 0)).r;
                        float LinearDepth = Linear01Depth(Depth);
	                float EyeDepth = LinearEyeDepth(Depth);
                        float4 ray = float4(uv * 2 - 1, 1, 1);
                        float4 worldRay = mul(_InverseProjectionMatrix, ray);
                        float3 viewRay = worldRay / worldRay.w; 

                        float3 rayVS = viewRay * Depth;
                        float3 rayPos = float3(uv, 1 - Depth);
                        float3 reflectionDir = reflect(normalize(rayVS), viewNormal);

                        float3 rayProject = float3(abs(_ProjectionMatrix._m00 * 0.5), abs(_ProjectionMatrix._m11 * 0.5), ((_ProjectionParams.z * _ProjectionParams.y) / (_ProjectionParams.y - _ProjectionParams.z)) * 0.5);
                        float3 rayDir = normalize(float3(reflectionDir.xy - (viewRay / viewRay.z) * reflectionDir.z, reflectionDir.z / EyeDepth) * rayProject);
                        rayDir.xy *= 0.5;
                        //float4 rayProject = mul(_ProjectionMatrix, float4((viewRay * LinearDepth) + reflectionDir, 1));
                        //float3 rayDir = normalize(((rayProject.xyz / rayProject.w) * 0.5 + 0.5) - rayPos) * 0.5;

                        //float3 rayUV = HiZTrace(4, 2, 2, 128, _ScreenSize.xy, rayPos, rayDir, _CameraDepthTexture);
                 
                        finalColor = float4(rayDir, 1);
                        //finalColor = tex2Dlod(_SceneColor, float4(rayPos.xy, 0, 0));
                        return finalColor;
                }
////////////////////////////////////////////////////////////////////////Unity HiZTrace////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////Unity HiZTrace////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////Unity HiZTrace////////////////////////////////////////////////////////////////////////
#define HIZINTERSECTIONKIND_NONE (0)
#define HIZINTERSECTIONKIND_CELL (1)
#define HIZINTERSECTIONKIND_DEPTH (2)
#define ZERO_INITIALIZE(type, name) name = (type)0;
#define LOAD_TEXTURE2D_LOD(textureName, unCoord2, lod)  textureName.Load(int3(unCoord2, lod))
const float DepthPlaneBias = 1E-5;

struct ScreenSpaceRayHit
{
    uint2 positionSS;         
    float2 positionNDC;        
    float linearDepth;         
};

struct ScreenSpaceRaymarchInput
{
    float3 rayOriginWS;    
    float3 rayDirWS;           
};

float4 ComputeClipSpacePosition(float3 position, float4x4 clipSpaceTransform)
{
    return mul(clipSpaceTransform, float4(position, 1.0));
}

float3 ComputeNormalizedDeviceCoordinatesWithZ(float3 position, float4x4 clipSpaceTransform)
{
    float4 positionCS = ComputeClipSpacePosition(position, clipSpaceTransform);
    positionCS *= rcp(positionCS.w);
    positionCS.xy = positionCS.xy * 0.5 + 0.5;
    return positionCS.xyz;
}

float2 ComputeNormalizedDeviceCoordinates(float3 position, float4x4 clipSpaceTransform)
{
    return ComputeNormalizedDeviceCoordinatesWithZ(position, clipSpaceTransform).xy;
}

void CalculateRaySS(
    float3 rayOriginWS,             // Ray origin (World Space)
    float3 rayDirWS,                // Ray direction (World Space)
    uint2 bufferSize,               // Texture size of screen buffers
    out float3 positionSS,          // (x, y, 1/linearDepth)
    out float3 raySS,               // (dx, dy, d(1/linearDepth))
    out float rayEndDepth           // Linear depth of the end point used to calculate raySS
)
{
    const float kNearClipPlane = -0.01;
    const float kMaxRayTraceDistance = 1000;

    float3 rayOriginVS = mul(_WorldToCameraMatrix, float4(rayOriginWS, 1.0)).xyz;
    float3 rayDirVS = mul((float3x3)_WorldToCameraMatrix, rayDirWS);
    float rayLength = ((rayOriginVS.z + rayDirVS.z * kMaxRayTraceDistance) > kNearClipPlane)
        ? ((kNearClipPlane - rayOriginVS.z) / rayDirVS.z)
        : kMaxRayTraceDistance;

    float3 positionWS = rayOriginWS;
    float3 rayEndWS = rayOriginWS + rayDirWS * rayLength;

    float4 positionCS = ComputeClipSpacePosition(positionWS, _ProjectionMatrix);
    float4 rayEndCS = ComputeClipSpacePosition(rayEndWS, _ProjectionMatrix);

    float2 positionNDC = ComputeNormalizedDeviceCoordinates(positionWS, _ProjectionMatrix);
    float2 rayEndNDC = ComputeNormalizedDeviceCoordinates(rayEndWS, _ProjectionMatrix);
    rayEndDepth = rayEndCS.w;

    float3 rayStartSS = float3(
        positionNDC.xy * bufferSize,
        1.0 / positionCS.w); 

    float3 rayEndSS = float3(
        rayEndNDC.xy * bufferSize,
        1.0 / rayEndDepth); 

    positionSS = rayStartSS;
    raySS = rayEndSS - rayStartSS;
}

float2 LoadDepth(float2 positionSS, int level, Texture2D _DepthPyramidTexture)
{
    float2 pyramidDepth = LOAD_TEXTURE2D_LOD(_DepthPyramidTexture, int2(positionSS.xy) >> level, level).rg;
    float2 linearDepth = float2(LinearEyeDepth(pyramidDepth.r), LinearEyeDepth(pyramidDepth.g));
    return linearDepth;
}

float2 LoadInvDepth(float2 positionSS, int level, Texture2D _DepthPyramidTexture)
{
    float2 linearDepth = LoadDepth(positionSS, level, _DepthPyramidTexture);
    float2 invLinearDepth = 1 / linearDepth;
    return invLinearDepth;
}

bool CellAreEquals(int2 cellA, int2 cellB)
{
    return cellA.x == cellB.x && cellA.y == cellB.y;
}

float3 IntersectDepthPlane(float3 positionSS, float3 raySS, float invDepth)
{
    float t = (invDepth - positionSS.z) / raySS.z;
    t = t >= 0.0f ? (t + DepthPlaneBias) : 1E5;
    return positionSS + raySS * t;
}

float2 CalculateDistanceToCellPlanes(
    float3 positionSS,              
    float2 invRaySS,              
    int2 cellId,                   
    uint2 cellSize,                
    int2 cellPlanes               
)
{

    int2 planes = (cellId + cellPlanes) * cellSize;
    float2 distanceToCellAxes = float2(planes - positionSS.xy) * invRaySS; 
    return distanceToCellAxes;
}

float3 IntersectCellPlanes(
    float3 positionSS,              
    float3 raySS,                  
    float2 invRaySS,               
    int2 cellId,                   
    uint2 cellSize,                
    int2 cellPlanes,               
    float2 crossOffset              
)
{
    float2 distanceToCellAxes = CalculateDistanceToCellPlanes(
        positionSS,
        invRaySS,
        cellId,
        cellSize,
        cellPlanes
    );
    float t = min(distanceToCellAxes.x, distanceToCellAxes.y) + 0.1;
    float3 testHitPositionSS = positionSS + raySS * t;
    return testHitPositionSS;
}

float CalculateHitWeight(
    ScreenSpaceRayHit hit,
    float2 startPositionSS,
    float minLinearDepth,
    float settingsDepthBufferThickness,
    float settingsRayMaxScreenDistance,
    float settingsRayBlendScreenDistance
)
{
    float2 screenDistanceNDC = abs(hit.positionSS.xy - startPositionSS) * _ScreenSize.zw;
    float2 screenDistanceWeights = clamp((settingsRayMaxScreenDistance - screenDistanceNDC) / settingsRayBlendScreenDistance, 0, 1);
    float screenDistanceWeight = min(screenDistanceWeights.x, screenDistanceWeights.y);
    return screenDistanceWeight;
}

float SampleBayer4(uint2 positionSS)
{
    const float4x4 Bayer4 = float4x4(0,  8,  2,  10,
                                         12, 4,  14, 6,
                                         3,  11, 1,  9,
                                         15, 7,  13, 5) / 16;

    return Bayer4[positionSS.x % 4][positionSS.y % 4];
}


int _DepthPyramidScale = 1;
int2 _DepthPyramidSize = 1;

bool ScreenSpaceLinearRaymarch(
    // Settings
    int settingRayLevel,                            // Mip level to use to ray march depth buffer
    uint settingsRayMaxIterations,                  // Maximum number of iterations (= max number of depth samples)
    float settingsDepthBufferThickness,              // Bias to use when trying to detect whenever we raymarch behind a surface
    float settingsRayMaxScreenDistance,             // Maximum screen distance raymarched
    float settingsRayBlendScreenDistance,           // Distance to blend before maximum screen distance is reached
    int settingsDebuggedAlgorithm,                  // currently debugged algorithm (see PROJECTIONMODEL defines)
    // Precomputed properties
    float3 startPositionSS,                         // Start position in Screen Space (x in pixel, y in pixel, z = 1/linearDepth)
    float3 raySS,                                   // Ray direction in Screen Space (dx in pixel, dy in pixel, z = 1/endPointLinearDepth - 1/startPointLinearDepth)
    float rayEndDepth,                              // Linear depth of the end point used to calculate raySS.
    uint2 bufferSize,                               // Texture size of screen buffers
    // Out
    out ScreenSpaceRayHit hit,
    out float hitWeight,
    out uint iteration, 
    Texture2D _DepthPyramidTexture) {
        ZERO_INITIALIZE(ScreenSpaceRayHit, hit);
        bool hitSuccessful = false;
        iteration = 0u;
        hitWeight = 0;
        int mipLevel = min(max(settingRayLevel, 0), _DepthPyramidScale);
        uint maxIterations = settingsRayMaxIterations;

        float3 positionSS = startPositionSS;
        raySS /= max(abs(raySS.x), abs(raySS.y));
        raySS *= 1 << mipLevel;

        float2 invLinearDepth = float2(0.0, 0.0);
        float minLinearDepth                = 0;
        float minLinearDepthWithThickness   = 0;
        float positionLinearDepth           = 0;

        for (iteration = 0u; iteration < maxIterations; ++iteration)
        {
                positionSS += raySS;
                invLinearDepth = LoadInvDepth(positionSS.xy, mipLevel, _DepthPyramidTexture);
                minLinearDepth                  = 1 / invLinearDepth.r;
                minLinearDepthWithThickness     = minLinearDepth + settingsDepthBufferThickness;
                positionLinearDepth             = 1 / positionSS.z;
                bool isAboveDepth               = positionLinearDepth < minLinearDepth;
                bool isAboveThickness           = positionLinearDepth < minLinearDepthWithThickness;
                bool isBehindDepth              = !isAboveThickness;
                bool intersectWithDepth         = !isAboveDepth && isAboveThickness;
                if (intersectWithDepth)
                {
                hitSuccessful = true;
                break;
                }
                if (any(int2(positionSS.xy) > int2(bufferSize)) || any(positionSS.xy < 0) )
                {
                hitSuccessful = false;
                break;
                }
        }
        if (iteration >= maxIterations)
                hitSuccessful = false;
        hit.linearDepth = 1 / positionSS.z;
        hit.positionNDC = float2(positionSS.xy) / float2(bufferSize);
        hit.positionSS = uint2(positionSS.xy);
        hitWeight = CalculateHitWeight(
                hit,
                startPositionSS.xy,
                invLinearDepth.r,
                settingsDepthBufferThickness,
                settingsRayMaxScreenDistance,
                settingsRayBlendScreenDistance
        );
        if (hitWeight <= 0)
                hitSuccessful = false;
        return hitSuccessful;
}

bool ScreenSpaceHiZRaymarch(
    float3 rayOriginWS,
    float3 rayDirWS,
    // Settings
    uint settingsRayMinLevel,                       // Minimum mip level to use for ray marching the depth buffer in HiZ
    uint settingsRayMaxLevel,                       // Maximum mip level to use for ray marching the depth buffer in HiZ
    uint settingsRayMaxIterations,                  // Maximum number of iteration for the HiZ raymarching (= number of depth sample for HiZ)
    float settingsDepthBufferThickness,             // Bias to use when trying to detect whenever we raymarch behind a surface
    float settingsRayMaxScreenDistance,             // Maximum screen distance raymarched
    float settingsRayBlendScreenDistance,           // Distance to blend before maximum screen distance is reached
    bool settingsRayMarchBehindObjects,             // Whether to raymarch behind objects
    int settingsDebuggedAlgorithm,                  // currently debugged algorithm (see PROJECTIONMODEL defines)
    // out
    out ScreenSpaceRayHit hit,
    out float hitWeight, 
    Texture2D _DepthPyramidTexture
)
{
    const float2 CROSS_OFFSET = float2(1, 1);
    ZERO_INITIALIZE(ScreenSpaceRayHit, hit);
    hitWeight = 0;
    bool hitSuccessful = false;
    uint iteration = 0u;
    int minMipLevel = max(settingsRayMinLevel, 0u);
    int maxMipLevel = min(settingsRayMaxLevel, _DepthPyramidScale);
    uint2 bufferSize = _DepthPyramidSize;
    uint maxIterations = settingsRayMaxIterations;
    float3 startPositionSS;
    float3 raySS;
    float rayEndDepth;
    CalculateRaySS(
        rayOriginWS,
        rayDirWS,
        bufferSize,
        startPositionSS,
        raySS,
        rayEndDepth
    );
    iteration = 0u;
    int intersectionKind = 0;
    float raySSLength = length(raySS.xy);
    raySS /= raySSLength;
    float2 invRaySS = float2(1, 1) / raySS.xy;
    int2 cellPlanes = sign(raySS.xy);
    float2 crossOffset = CROSS_OFFSET * cellPlanes;
    cellPlanes = clamp(cellPlanes, 0, 1);
    int currentLevel = minMipLevel;
    uint2 cellCount = bufferSize >> currentLevel;
    uint2 cellSize = uint2(1, 1) << currentLevel;
    float3 positionSS = startPositionSS;
    float2 invLinearDepth = float2(0.0, 0.0);
    float positionLinearDepth           = 0;
    float minLinearDepth                = 0;
    float minLinearDepthWithThickness   = 0;
    {
        const float epsilon = 1E-3;
        const float minTraversal = 2 << currentLevel;
        float2 distanceToCellAxes = CalculateDistanceToCellPlanes(
            positionSS,
            invRaySS,
            int2(positionSS.xy) / cellSize,
            cellSize,
            cellPlanes
        );

        float t = min(distanceToCellAxes.x * minTraversal + epsilon, distanceToCellAxes.y * minTraversal + epsilon);
        positionSS = positionSS + raySS * t;
    }
    bool isBehindDepth = false;
    while (currentLevel >= minMipLevel)
    {
        hitSuccessful = true;
        if (iteration >= maxIterations)
        {
            hitSuccessful = false;
            break;
        }
        cellCount = bufferSize >> currentLevel;
        cellSize = uint2(1, 1) << currentLevel;
        int mipLevelDelta = -1;
        invLinearDepth = LoadInvDepth(positionSS.xy, currentLevel, _DepthPyramidTexture);
        positionLinearDepth                 = 1 / positionSS.z;
        minLinearDepth                      = 1 / invLinearDepth.r;
        minLinearDepthWithThickness         = minLinearDepth + settingsDepthBufferThickness;
        bool isAboveDepth                   = positionLinearDepth < minLinearDepth;
        bool isAboveThickness               = positionLinearDepth < minLinearDepthWithThickness;
        isBehindDepth                       = !isAboveThickness;
        bool intersectWithDepth             = minLinearDepth >= positionLinearDepth && isAboveThickness;
        intersectionKind = HIZINTERSECTIONKIND_NONE;
        if (isAboveDepth)
        {
            float3 candidatePositionSS = IntersectDepthPlane(positionSS, raySS, invLinearDepth.r);
            intersectionKind = HIZINTERSECTIONKIND_DEPTH;
            const int2 cellId = int2(positionSS.xy) / cellSize;
            const int2 candidateCellId = int2(candidatePositionSS.xy) / cellSize;
            if (!CellAreEquals(cellId, candidateCellId))
            {
                candidatePositionSS = IntersectCellPlanes(
                    positionSS,
                    raySS,
                    invRaySS,
                    cellId,
                    cellSize,
                    cellPlanes,
                    crossOffset
                );
                intersectionKind = HIZINTERSECTIONKIND_CELL;
                mipLevelDelta = 1;
            }
            positionSS = candidatePositionSS;
        }
        else if (settingsRayMarchBehindObjects && isBehindDepth && currentLevel <= (minMipLevel + 1))
        {
            const int2 cellId = int2(positionSS.xy) / cellSize;
            positionSS = IntersectCellPlanes(
                positionSS,
                raySS,
                invRaySS,
                cellId,
                cellSize,
                cellPlanes,
                crossOffset
            );
            intersectionKind = HIZINTERSECTIONKIND_CELL;
            mipLevelDelta = 1;
        }
        currentLevel = min(currentLevel + mipLevelDelta, maxMipLevel);
        float4 distancesToBorders = float4(positionSS.xy, bufferSize - positionSS.xy);
        float distanceToBorders = min(min(distancesToBorders.x, distancesToBorders.y), min(distancesToBorders.z, distancesToBorders.w));
        int minLevelForBorders = int(log2(distanceToBorders));
        currentLevel = min(currentLevel, minLevelForBorders);
        if (any(int2(positionSS.xy) > int2(bufferSize)) || any(positionSS.xy < 0))
        {
            hitSuccessful = false;
            break;
        }
        ++iteration;
    }
    hit.linearDepth = positionLinearDepth;
    hit.positionSS = uint2(positionSS.xy);
    hit.positionNDC = float2(hit.positionSS) / float2(bufferSize);
    hitWeight = CalculateHitWeight(
        hit,
        startPositionSS.xy,
        minLinearDepth,
        settingsDepthBufferThickness,
        settingsRayMaxScreenDistance,
        settingsRayBlendScreenDistance
    );
    if (hitWeight <= 0 || isBehindDepth)
        hitSuccessful = false;
    return hitSuccessful;
}
           
float4 HiZTrace_Unity(v2f i) : SV_Target {
        float4 finalColor = 0;
        float2 uv = i.uv.xy;
        return finalColor;
}
////////////////////////////////////////////////////////////////////////Unity HiZTrace////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////Unity HiZTrace////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////Unity HiZTrace////////////////////////////////////////////////////////////////////////
	ENDCG

	SubShader
	{
        ZTest Always Cull Off ZWrite Off
		Pass {
			CGPROGRAM
			#pragma vertex vert
			//#pragma fragment LinearTrace2D
            #pragma fragment HiZTrace3D
		ENDCG
		}
	}
}
