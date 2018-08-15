#define PI 3.14159265359

inline half  Pow2(half  x) { return x * x; }
inline half2 Pow2(half2 x) { return x * x; }
inline half3 Pow2(half3 x) { return x * x; }
inline half4 Pow2(half4 x) { return x * x; }

float4 TangentToWorld(float3 N, float4 H) {
        float3 UpVector = abs(N.z) < 0.999 ? float3(0,0,1) : float3(1,0,0);
	float3 T = normalize( cross( UpVector, N ) );
	float3 B = cross( N, T );		 
	return float4((T * H.x) + (B * H.y) + (N * H.z), H.w);
}

float4 ImportanceSampleBlinn( float2 Xi, float Roughness ) {
	float m = Roughness*Roughness;
	float m2 = m*m;
	float n = 2 / m2 - 2;
	float Phi = 2 * PI * Xi.x;
	float CosTheta = pow( max(Xi.y, 0.001f), 1 / (n + 1) );
	float SinTheta = sqrt( 1 - CosTheta * CosTheta );
	float3 H;
	H.x = SinTheta * cos( Phi );
	H.y = SinTheta * sin( Phi );
	H.z = CosTheta;
	float D = (n+2)/ (2*PI) * saturate(pow( CosTheta, n ));
	float pdf = D * CosTheta;
	return float4(H, pdf); 
}
		
float4 ImportanceSampleGGX( float2 Xi, float Roughness ) {
	float m = Roughness*Roughness;
	float m2 = m*m;
	float Phi = 2 * PI * Xi.x;		 
	float CosTheta = sqrt( (1 - Xi.y) / ( 1 + (m2 - 1) * Xi.y ) );
	float SinTheta = sqrt( 1 - CosTheta * CosTheta );  		 
	float3 H;
	H.x = SinTheta * cos( Phi );
	H.y = SinTheta * sin( Phi );
	H.z = CosTheta;
	float d = ( CosTheta * m2 - CosTheta ) * CosTheta + 1;
	float D = m2 / ( PI*d*d );
	float pdf = D * CosTheta;
	return float4(H, pdf); 
}

float2 Noise(float2 pos, float random) {
        return frac(sin(dot(pos.xy * random, float2(12.9898f, 78.233f))) * float2(43758.5453f, 28001.8384f));
}

float HaltonSequence (uint index, uint base = 3) {
        float result = 0;
	float f = 1;
	int i = index;
	[unroll]
	while (i > 0)  {
		f = f / base;
		result = result + f * (i % base);
		i = floor(i / base);
	}
	return result;
}

float2 Hammersley(int i, int N) {
	return float2(float(i) * (1.0/float( N )), HaltonSequence(i, 3) );
}

float calcLOD(int cubeSize, float pdf, int NumSamples) {
	float lod = (0.5 * log2( (cubeSize*cubeSize)/float(NumSamples) ) + 2.0) - 0.5*log2(pdf); 
	return lod;
}

float specularPowerToConeAngle(float specularPower) {
	 const float xi = 0.244f;
	 float exponent = 1.0f / (specularPower + 1.0f);
	 return acos(pow(xi, exponent));
}

float isoscelesTriangleOpposite(float adjacentLength, float coneTheta) {
	 return 2.0f * tan(coneTheta) * adjacentLength;
}
 
float isoscelesTriangleInRadius(float a, float h) {
	 float a2 = a * a;
	 float fh2 = 4.0f * h * h;
	 return (a * (sqrt(a2 + fh2) - a)) / (4.0f * h);
}

float isoscelesTriangleNextAdjacent(float adjacentLength, float incircleRadius) {
	 return adjacentLength - (incircleRadius * 2.0f);
}

float LinearTrace3D_Alpha (float2 pos, float value) {
	float borderDist = min(1 - max(pos.x, pos.y), min(pos.x, pos.y));
	return saturate(borderDist > value ? 1 : borderDist / value);
}

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

float3 GetPosition(sampler2D _CameraDepthTexture, float4 _MainTex_TexelSize, float4 _ProjInfo, float2 ssP) {
        float3 P;
        P.z = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, ssP.xy);
        P = ReconstructCSPosition(_MainTex_TexelSize, _ProjInfo, float2(ssP), P.z);
        return P;
}

float sqr(float x) {
        return x*x;
}

float3 GetReflectionDir(float3 csPosition, float3 csN) {
        float3 csE = -normalize(csPosition.xyz);
        float cos_o = dot(csN, csE);
        float3 c_mi = normalize((csN * (2.0 * cos_o)) - csE);
        return c_mi;
}

////////////////////////////////////////HiZTrace////////////////////////////////////////

float2 cell(float2 ray, float2 cell_count) {
	return floor(ray.xy * cell_count);
}

float2 cell_count(float level, float2 ScreenSize) {
	return ScreenSize / (level == 0 ? 1 : exp2(level));
}

float3 intersect_cell_boundary(float3 pos, float3 dir, float2 cell_id, float2 cell_count, float2 cross_step, float2 cross_offset) {
	float2 cell_size = 1 / cell_count;
	float2 planes = cell_id / cell_count + cell_size * cross_step;
	float2 solutions = (planes - pos)/dir.xy;
	float3 intersection_pos = pos + dir * min(solutions.x, solutions.y);
	intersection_pos.xy += (solutions.x < solutions.y) ? float2(cross_offset.x, 0) : float2(0, cross_offset.y);
	return intersection_pos;
}

bool crossed_cell_boundary(float2 cell_id_one, float2 cell_id_two) {
	return (int)cell_id_one.x != (int)cell_id_two.x || (int)cell_id_one.y != (int)cell_id_two.y;
}

float minimum_depth_plane(float2 ray, float level, sampler2D forntDepth) {
        float depth =  tex2Dlod(forntDepth, float4(ray, 0, level)).r;
	return 1 - depth;
}

float3 HiZTrace(int HiZ_Max_Level, int HiZ_Start_Level, int HiZ_Stop_Level, int Steps, float2 ScreenSize, float3 rayPos, float3 rayDir, sampler2D forntDepth) {
	float level = HiZ_Start_Level;
	float3 rayDir_DividZ = rayDir.xyz / rayDir.z;
	float2 hi_z_size = cell_count(level, ScreenSize);
	
	float2 cross_step = float2(rayDir.x >= 0 ? 1 : -1, rayDir.y >= 0 ? 1 : -1);
	float2 cross_offset = cross_step * 0.00001;
	cross_step = saturate(cross_step);
	float2 ray_cell = cell(rayPos.xy, hi_z_size.xy);
	rayPos = intersect_cell_boundary(rayPos, rayDir, ray_cell, hi_z_size, cross_step, cross_offset);
        int iterations = 0;

        [unroll]
	while(level >= HiZ_Stop_Level && iterations < Steps){
		float2 current_cell_count = cell_count(level, ScreenSize);
		float2 old_cell_id = cell(rayPos.xy, current_cell_count);
		float min_z = minimum_depth_plane(rayPos.xy, level, forntDepth);
		float3 temporalRay = rayPos;
		if(rayDir.z > 0) {
			float min_minus_ray = min_z - rayPos.z;
			temporalRay = min_minus_ray > 0 ? rayPos + rayDir_DividZ * min_minus_ray : temporalRay;
			float2 new_cell_id = cell(temporalRay.xy, current_cell_count);
			if(crossed_cell_boundary(old_cell_id, new_cell_id)) {
				temporalRay = intersect_cell_boundary(rayPos, rayDir, old_cell_id, current_cell_count, cross_step, cross_offset);
				level = min(HiZ_Max_Level, level + 2);
			} else {
				if(level == 1 && abs(min_minus_ray) > 0.0001) {
				        temporalRay = intersect_cell_boundary(rayPos, rayDir, old_cell_id, current_cell_count, cross_step, cross_offset);
				        level = 2;
				}
			}
		} else if(rayPos.z < min_z) {
		        temporalRay = intersect_cell_boundary(rayPos, rayDir, old_cell_id, current_cell_count, cross_step, cross_offset);
		        level = min(HiZ_Max_Level, level + 2);
		}
		rayPos.xyz = temporalRay.xyz;
		--level;
                ++iterations;
 	}
        return rayPos;
}




















