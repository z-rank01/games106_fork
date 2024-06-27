#version 450

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec4 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inViewVec;
layout (location = 4) in vec3 inLightVec;
layout (location = 5) in vec3 inWorldPos;

layout (set = 1, binding = 0) uniform sampler2D colorMap;
layout (set = 1, binding = 1) uniform sampler2D metallicRoughnessMap;
layout (set = 1, binding = 2) uniform sampler2D normalMap;
layout (set = 1, binding = 3) uniform sampler2D emissiveMap;
layout (set = 1, binding = 4) uniform sampler2D occlusionMap;

layout (set = 3, binding = 0) uniform UBOParams {
	vec4 lightDir;
	float exposure;
	float gamma;
	float prefilteredCubeMipLevels;
	float scaleIBLAmbient;
} uboParams;

layout (set = 3, binding = 1) uniform samplerCube samplerIrradiance;
layout (set = 3, binding = 2) uniform samplerCube prefilteredMap;
layout (set = 3, binding = 3) uniform sampler2D samplerBRDFLUT;

layout(push_constant) uniform PushConsts {
	layout(offset = 0)  vec4 baseColorFactor;
	layout(offset = 16) vec4 emissiveFactor;
	layout(offset = 32) float metallicFactor;
	layout(offset = 36) float roughnessFactor;
	layout(offset = 40) float alphaMode;
	layout(offset = 44) float alphaCutoff;

	layout(offset = 48) int colorTextureSet;
	layout(offset = 52) int PhysicalDescriptorTextureSet;
	layout(offset = 56) int normalTextureSet;
	layout(offset = 60) int occlusionTextureSet;
	layout(offset = 64) int emissiveTextureSet;
} material;

layout (location = 0) out vec4 outFragColor;

struct PBRpara
{
	float NdotL;
	float NdotV;
	float NdotH;
	float LdotH;
	float VdotH;
	float roughness;
	float matalness;
	vec3 reflectance0;
	vec3 reflectance90;
	float alphaRoughness;
	vec3 diffuseColor;
	vec3 specularColor;
};

const float PI = 3.141592653589793;
const float minRoughness = 0.04;
const float gamma = 2.2;

// tone map
vec3 Uncharted2Tonemap(vec3 color)
{
	float A = 0.15;
	float B = 0.50;
	float C = 0.10;
	float D = 0.20;
	float E = 0.02;
	float F = 0.30;
	float W = 11.2;
	return ((color*(A*color+C*B)+D*E)/(color*(A*color+B)+D*F))-E/F;
}

vec4 tonemap(vec4 color)
{
	vec3 outcol = Uncharted2Tonemap(color.rgb * uboParams.exposure);
	outcol = outcol * (1.0f / Uncharted2Tonemap(vec3(11.2f)));	
	return vec4(pow(outcol, vec3(1.0f / uboParams.gamma)), color.a);
}

// get linear color space value
vec4 SRGBtoLINEAR(vec4 srgb)
{
	vec3 linear = pow(srgb.xyz, vec3(gamma));
	return vec4(linear, srgb.w);
}

// get TBN matrix for normal map
mat3 getTBNmaxtrix()
{
	vec3 q1 = dFdx(inWorldPos);
	vec3 q2 = dFdy(inWorldPos);
	vec2 st1 = dFdx(inUV);
	vec2 st2 = dFdy(inUV);

	vec3 N = normalize(inNormal);
	vec3 T = normalize(q1 * st2.t - q2 * st1.t);
	vec3 B = -normalize(cross(N, T));
	mat3 TBN = mat3(T, B, N);

	return TBN;
}

// PBR functions:
// --------------
// lambertian diffuse
vec3 diffuse(PBRpara pbrInputs)
{
	return pbrInputs.diffuseColor / PI;
}

// fresnel reflectance: F() function
vec3 fresnelReflection(PBRpara pbrInputs)
{
	return pbrInputs.reflectance0 + (pbrInputs.reflectance90 - pbrInputs.reflectance0) * pow(clamp(1.0 - pbrInputs.VdotH, 0.0, 1.0), 5.0);
}

// geometric attenuation: G() function
float geometricOcclusion(PBRpara pbrInputs)
{
	float NdotL = pbrInputs.NdotL;
	float NdotV = pbrInputs.NdotV;
	float r = pbrInputs.alphaRoughness;

	float attenuationL = 2.0 * NdotL / (NdotL + sqrt(r * r + (1.0 - r * r) * (NdotL * NdotL)));
	float attenuationV = 2.0 * NdotV / (NdotV + sqrt(r * r + (1.0 - r * r) * (NdotV * NdotV)));
	return attenuationL * attenuationV;
}

// microfacet distribution: D() function
float microfacetDistribution(PBRpara pbrInputs)
{
	float roughnessSq = pbrInputs.alphaRoughness * pbrInputs.alphaRoughness;
	float f = (pbrInputs.NdotH * roughnessSq - pbrInputs.NdotH) * pbrInputs.NdotH + 1.0;
	return roughnessSq / (PI * f * f);
}

// ibl indirect lighting 
vec3 getIBLContribution(PBRpara pbrInputs, vec3 n, vec3 reflection)
{
	float lod = (pbrInputs.roughness * uboParams.prefilteredCubeMipLevels);
	vec2 brdf_uv = vec2(pbrInputs.NdotV, 1.0 - pbrInputs.roughness);
	brdf_uv.x = -brdf_uv.x;
	vec3 brdf = (texture(samplerBRDFLUT, brdf_uv)).rgb;
	vec3 diffuseLight = SRGBtoLINEAR(tonemap(texture(samplerIrradiance, n))).rgb;
	vec3 specularLight = SRGBtoLINEAR(tonemap(textureLod(prefilteredMap, reflection, lod))).rgb;

	vec3 diffuse = diffuseLight * pbrInputs.diffuseColor;
	vec3 specular = specularLight * (pbrInputs.specularColor * brdf.x + brdf.y);

	// For presentation, this allows us to disable IBL terms
	// For presentation, this allows us to disable IBL terms
	diffuse *= uboParams.scaleIBLAmbient;
	specular *= uboParams.scaleIBLAmbient;

	return diffuse + specular;
}

void main() 
{
	float roughness;
	float metallic;
	vec3 diffuseColor;
	vec4 baseColor;

	vec3 f0 = vec3(0.04);

	// base color
	if(material.alphaMode == 2.0f)
	{
		if(material.colorTextureSet > -1)
			baseColor = SRGBtoLINEAR(texture(colorMap, inUV)) * material.baseColorFactor;
		else
			baseColor = material.baseColorFactor;

		// cull off part
		if(baseColor.a < material.alphaCutoff)
			discard;
	}
	else
	{
		if(material.colorTextureSet > -1)
			baseColor = SRGBtoLINEAR(texture(colorMap, inUV)) * material.baseColorFactor;
		else
			baseColor = material.baseColorFactor;
	}
	baseColor *= inColor;

	// metallic and roughness
	roughness = material.roughnessFactor;
	metallic = material.metallicFactor;
	if(material.PhysicalDescriptorTextureSet > -1)
	{
		vec4 metallicRoughnessTexture = texture(metallicRoughnessMap, inUV);
		roughness = metallicRoughnessTexture.g * roughness;
		metallic = metallicRoughnessTexture.b * metallic;
	}
	else
	{
		roughness = clamp(roughness, minRoughness, 1.0);
		metallic = clamp(metallic, 0.0, 1.0);
	}

	
	// pbr parameter
	diffuseColor = baseColor.rgb * (vec3(1.0) - f0);
	diffuseColor *= 1.0 - metallic;

	float alphaRoughness = roughness * roughness;

	vec3 specularColor = mix(f0, baseColor.rgb, metallic);

	// reflectance
	float reflectance = max(max(specularColor.r, specularColor.g), specularColor.b);
	float reflectance90 = clamp(reflectance * 25.0, 0.0, 1.0);
	vec3 specularEnvironmentR0 = specularColor.rgb;
	vec3 specularEnvironmentR90 = vec3(1.0) * reflectance90;
	
	
	mat3 TBN = getTBNmaxtrix();
	vec3 normalTexture = texture(normalMap, inUV).xyz * 2.0 - 1.0;

	vec3 n = (material.normalTextureSet > -1) ? TBN * normalTexture : normalize(inNormal);
	vec3 v = normalize(inViewVec);
	vec3 l = normalize(inLightVec);
	vec3 h = normalize(l + v);
	vec3 reflection = -normalize(reflect(v, n));
	reflection.y *= -1.0f;

	float NdotL = clamp(dot(n, l), 0.001, 1.0);
	float NdotV = clamp(abs(dot(n, v)), 0.001, 1.0);
	float NdotH = clamp(dot(n, h), 0.0, 1.0);
	float LdotH = clamp(dot(l, h), 0.0, 1.0);
	float VdotH = clamp(dot(v, h), 0.0, 1.0);

	PBRpara pbrInputs = PBRpara(
		NdotL,
		NdotV,
		NdotH,
		LdotH,
		VdotH,
		roughness,
		metallic,
		specularEnvironmentR0,
		specularEnvironmentR90,
		alphaRoughness,
		diffuseColor,
		specularColor
	);

	// shading function
	vec3 F = fresnelReflection(pbrInputs);
	float G = geometricOcclusion(pbrInputs);
	float D = microfacetDistribution(pbrInputs);

	// calculate lighting
	const vec3 lightColor = vec3(1.0);
	vec3 diffuseContribution = (1.0 - F) * diffuse(pbrInputs);
	vec3 specularContribution = F * G * D / (4.0 * NdotL * NdotV);
	vec3 color = NdotL * lightColor * (diffuseContribution + specularContribution);
	color += getIBLContribution(pbrInputs, n, reflection);

	// occlusion
	const float occlusionStrength = 0.5f;
	if (material.occlusionTextureSet > -1) {
		float ao = texture(occlusionMap, inUV).r;
		color = mix(color, color * ao, occlusionStrength);
	}
	
	// emissive
	const float emissiveStrength = 10.0f;
	if (material.emissiveTextureSet > -1) {
		vec3 emissive = SRGBtoLINEAR(texture(emissiveMap, inUV)).rgb * material.emissiveFactor.xyz * emissiveStrength;
		color += emissive;
	}
	else
	{
		vec3 emissive = material.emissiveFactor.xyz * emissiveStrength;
		color += emissive;
	}

	outFragColor = vec4(color, baseColor.a);
	//outFragColor = vec4(material.emissiveFactor, 1.0);
}