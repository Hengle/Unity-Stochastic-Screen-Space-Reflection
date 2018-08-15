using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

//[ExecuteInEditMode]
[RequireComponent(typeof(Camera))]
public class SSR : MonoBehaviour {
        public int MaxSteps = 64;
        public float StepSize = 28;
        public float RayDistance = 32;
        public float Thickness = 0.1f;

        private Material SSRMaterial;
        private Camera RenderCamera;
        private Vector4 CameraSize;
        private Matrix4x4 projectionMatrix;
        private Matrix4x4 viewProjectionMatrix;
        private Matrix4x4 _lastFrame_viewProjectionMatrix;
        private Matrix4x4 inverseViewProjectionMatrix;
        private Matrix4x4 worldToCameraMatrix;
        private Matrix4x4 cameraToWorldMatrix;
        private int _SceneColor = Shader.PropertyToID("_SceneColor");
        private int _SceneDepth = Shader.PropertyToID("_SceneDepth");
        private RenderTexture _SceneColorRT;
        private RenderTexture _SceneDepthRT;
        private CommandBuffer RenderBuffer;

        public Texture2D BlueNoise;

        private void Awake() {
                RenderCamera = gameObject.GetComponent<Camera>();
                SSRMaterial = new Material(Shader.Find("Hidden/SSR"));

                _SceneColorRT = new RenderTexture(RenderCamera.pixelWidth, RenderCamera.pixelHeight, 0, RenderTextureFormat.DefaultHDR);
                _SceneColorRT.filterMode = FilterMode.Bilinear;
                //_SceneColorRT.useMipMap = true;
                //_SceneColorRT.autoGenerateMips = false;

                _SceneDepthRT = new RenderTexture(RenderCamera.pixelWidth, RenderCamera.pixelHeight, 0, RenderTextureFormat.RHalf);
                _SceneDepthRT.filterMode = FilterMode.Bilinear;
                //_SceneDepthRT.useMipMap = true;
                //_SceneDepthRT.autoGenerateMips = false;

                RenderBuffer = new CommandBuffer();
                RenderBuffer.name = "SSR";
                RenderCamera.AddCommandBuffer(CameraEvent.BeforeImageEffectsOpaque, RenderBuffer);
        }

        private void OnEnable() {
                RenderCamera.AddCommandBuffer(CameraEvent.BeforeImageEffectsOpaque, RenderBuffer);
        }

        private void OnPreRender() {
                CameraSize = new Vector2(RenderCamera.pixelWidth, RenderCamera.pixelHeight);
                SSRMaterial.SetVector("_ScreenSize", CameraSize);
                worldToCameraMatrix = RenderCamera.worldToCameraMatrix;
                cameraToWorldMatrix = worldToCameraMatrix.inverse;
                projectionMatrix = RenderCamera.projectionMatrix;
                viewProjectionMatrix = projectionMatrix * worldToCameraMatrix;

                SSRMaterial.SetTexture("_Noise", BlueNoise);
                SSRMaterial.SetVector("_NoiseSize", new Vector2(BlueNoise.width, BlueNoise.height));
                Vector2 jitterSample = GenerateRandomOffset();
                SSRMaterial.SetVector("_Jitter", new Vector4((float)BlueNoise.width, (float)BlueNoise.height, jitterSample.x, jitterSample.y));
                //////////////////////////////////////////////////////////////////////////////
                float sWidth = CameraSize.x;
                float sHeight = CameraSize.y;
                float sx = sWidth / 2.0f;
                float sy = sHeight / 2.0f;
                Matrix4x4 warpToScreenSpaceMatrix = new Matrix4x4();
                warpToScreenSpaceMatrix.SetRow(0, new Vector4(sx, 0.0f, 0.0f, sx));
                warpToScreenSpaceMatrix.SetRow(1, new Vector4(0.0f, sy, 0.0f, sy));
                warpToScreenSpaceMatrix.SetRow(2, new Vector4(0.0f, 0.0f, 1.0f, 0.0f));
                warpToScreenSpaceMatrix.SetRow(3, new Vector4(0.0f, 0.0f, 0.0f, 1.0f));
                Matrix4x4 projectToPixelMatrix = warpToScreenSpaceMatrix * projectionMatrix;
                SSRMaterial.SetMatrix("_ProjectToPixelMatrix", projectToPixelMatrix);
                SSRMaterial.SetInt("_TreatBackfaceHitAsMiss", false ? 1 : 0);
                SSRMaterial.SetInt("_AllowBackwardsRays", true ? 1 : 0);
                SSRMaterial.SetInt("_TraceBehindObjects", true ? 1 : 0);
                Vector4 projInfo = new Vector4
                        ((-2.0f / (sWidth * projectionMatrix[0])),
                        (-2.0f / (sHeight * projectionMatrix[5])),
                        ((1.0f - projectionMatrix[2]) / projectionMatrix[0]),
                        ((1.0f + projectionMatrix[6]) / projectionMatrix[5]));
                SSRMaterial.SetVector("_ProjInfo", projInfo);
                Vector3 cameraClipInfo = (float.IsPositiveInfinity(RenderCamera.farClipPlane)) ?
                        new Vector3(RenderCamera.nearClipPlane, -1.0f, 1.0f) :
                        new Vector3(RenderCamera.nearClipPlane * RenderCamera.farClipPlane, RenderCamera.nearClipPlane - RenderCamera.farClipPlane, RenderCamera.farClipPlane);
                SSRMaterial.SetVector("_CameraClipInfo", cameraClipInfo);
                SSRMaterial.SetInt("_NumSteps", MaxSteps);
                SSRMaterial.SetFloat("_RayStepSize", StepSize);
                SSRMaterial.SetFloat("_MaxRayTraceDistance", RayDistance);
                SSRMaterial.SetFloat("_Thickness", Thickness);
                //////////////////////////////////////////////////////////////////////////////
                SSRMaterial.SetMatrix("_WorldToCameraMatrix", worldToCameraMatrix);
                SSRMaterial.SetMatrix("_CameraToWorldMatrix", cameraToWorldMatrix);
                SSRMaterial.SetMatrix("_ProjectionMatrix", projectionMatrix);
                SSRMaterial.SetMatrix("_InverseProjectionMatrix", projectionMatrix.inverse);
                SSRMaterial.SetMatrix("_ViewProjectionMatrix", viewProjectionMatrix);
                SSRMaterial.SetMatrix("_InverseViewProjectionMatrix", viewProjectionMatrix.inverse);


                RenderBuffer.Clear();

                RenderBuffer.Blit(BuiltinRenderTextureType.CameraTarget, _SceneColorRT);
                //RenderBuffer.GenerateMips(_SceneColorRT);
                RenderBuffer.SetGlobalTexture(_SceneColor, _SceneColorRT);

                RenderBuffer.Blit(BuiltinRenderTextureType.ResolvedDepth, _SceneDepthRT);
                //RenderBuffer.GenerateMips(_SceneDepthRT);
                RenderBuffer.SetGlobalTexture(_SceneDepth, _SceneDepthRT);

                RenderBuffer.Blit(null, BuiltinRenderTextureType.CameraTarget, SSRMaterial, 0);
        }

        void OnDisable() {
                if (_SceneColorRT != null)
                {
                        RenderBuffer.ReleaseTemporaryRT(_SceneColor);
                }
                if (_SceneDepthRT != null)
                {
                        RenderBuffer.ReleaseTemporaryRT(_SceneDepth);
                }
                RenderCamera.RemoveCommandBuffer(CameraEvent.BeforeImageEffectsOpaque, RenderBuffer);
                RenderBuffer.Release();
                RenderBuffer.Dispose();
        }

        private float GetHaltonValue(int index, int radix)
        {
                float result = 0f;
                float fraction = 1f / (float)radix;

                while (index > 0)
                {
                result += (float)(index % radix) * fraction;
                index /= radix;
                fraction /= (float)radix;
                }
                return result;
        }

        private int SampleCount = 64;
        private int SampleIndex = 0;
        private Vector2 GenerateRandomOffset()
        {
                var offset = new Vector2(GetHaltonValue(SampleIndex & 1023, 2), GetHaltonValue(SampleIndex & 1023, 3));
                if (SampleIndex++ >= SampleCount)
                SampleIndex = 0;
                return offset;
        }
}
