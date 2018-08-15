float ClampedPow(float X, float Y) {
	return pow(max(abs(X), 0.000001), Y);
}

float D_GGX(float Roughness, float NdotH) {
	float m = Roughness * Roughness;
	float m2 = m * m;
	
	float D = m2 / (PI * sqr(sqr(NdotH) * (m2 - 1) + 1));
	
	return D;
}

	
float G_GGX(float Roughness, float NdotL, float NdotV) {
	float m = Roughness * Roughness;
	float m2 = m * m;

	float G_L = 1 / (NdotL + sqrt(m2 + (1 - m2) * NdotL * NdotL));
	float G_V = 1 / (NdotV + sqrt(m2 + (1 - m2) * NdotV * NdotV));
	float G = G_L * G_V;
	
	return G;
}

float BRDF_UE4(float3 V, float3 L, float3 N, float Roughness) {
	float3 H = normalize(L + V);

	float NdotH = saturate(dot(N,H));
	float NdotL = saturate(dot(N,L));
	float NdotV = saturate(dot(N,V));

	float D = D_GGX(Roughness, NdotH);
	float G = G_GGX(Roughness, NdotL, NdotV);

	return D * G;
}

float BRDF_Unity_Weight(float3 V, float3 L, float3 N, float Roughness) {
	float3 H = normalize(L + V);

	float NdotH = saturate(dot(N,H));
	float NdotL = saturate(dot(N,L));
	float NdotV = saturate(dot(N,V));

	half G = SmithJointGGXVisibilityTerm (NdotL, NdotV, Roughness);
	half D = GGXTerm (NdotH, Roughness);

	return (D * G) * (UNITY_PI / 4);
}

float4 TangentToWorld(float3 N, float4 H) {
	float3 UpVector = abs(N.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
	float3 T = normalize( cross( UpVector, N ) );
	float3 B = cross( N, T );
				 
	return float4((T * H.x) + (B * H.y) + (N * H.z), H.w);
}

float4 ImportanceSampleGGX( float2 E, float Roughness ) {
	float m = Roughness * Roughness;
	float m2 = m * m;

	float Phi = 2 * PI * E.x;
	float CosTheta = sqrt( (1 - E.y) / ( 1 + (m2 - 1) * E.y ) );
	float SinTheta = sqrt( 1 - CosTheta * CosTheta );

	float3 H;
	H.x = SinTheta * cos( Phi );
	H.y = SinTheta * sin( Phi );
	H.z = CosTheta;
	
	float d = ( CosTheta * m2 - CosTheta ) * CosTheta + 1;
	float D = m2 / ( PI*d*d );
	float PDF = D * CosTheta;

	return float4( H, PDF );
}

float4 ImportanceSampleBlinn(float2 E, float Roughness) {
	float m = Roughness * Roughness;
	float n = 2 / (m*m) - 2;

	float Phi = 2 * PI * E.x;
	float CosTheta = ClampedPow( E.y, 1 / (n + 1) );
	float SinTheta = sqrt( 1 - CosTheta * CosTheta );

	float3 H;
	H.x = SinTheta * cos( Phi );
	H.y = SinTheta * sin( Phi );
	H.z = CosTheta;

	float D = (n+2)/ (2*PI) * ClampedPow( CosTheta, n );
	float PDF = D * CosTheta;

	return float4( H, PDF );
}

float3 FresnelSchlick(float NoV, float3 F0) {
    return F0 + (1 - F0) * pow(1.0 - NoV, 5);
}

half3 EnvBRDFApprox( half3 SpecularColor, half Roughness, half NoV ) {
	const half4 c0 = { -1, -0.0275, -0.572, 0.022 };
	const half4 c1 = { 1, 0.0425, 1.04, -0.04 };
	half4 r = Roughness * c0 + c1;
	half a004 = min( r.x * r.x, exp2( -9.28 * NoV ) ) * r.x + r.y;
	half2 AB = half2( -1.04, 1.04 ) * a004 + r.zw;
	AB.y *= saturate( 50.0 * SpecularColor.g );
	return SpecularColor * AB.x + AB.y;
}



