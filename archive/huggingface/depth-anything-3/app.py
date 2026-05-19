# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Hugging Face Spaces App for Depth Anything 3.

This app uses the @spaces.GPU decorator to dynamically allocate GPU resources
for model inference on Hugging Face Spaces.
"""

import os
import spaces
from depth_anything_3.app.gradio_app import DepthAnything3App
from depth_anything_3.app.modules.model_inference import ModelInference

# Apply @spaces.GPU decorator to run_inference method
# This ensures GPU operations happen in isolated subprocess
# Model loading and inference will occur in GPU subprocess, not main process
original_run_inference = ModelInference.run_inference

@spaces.GPU(duration=1200)  # Request GPU for up to 120 seconds per inference
def gpu_run_inference(self, *args, **kwargs):
    """
    GPU-accelerated inference with Spaces decorator.
    
    This function runs in a GPU subprocess where:
    - Model is loaded and moved to GPU (safe)
    - CUDA operations are allowed
    - All CUDA tensors are moved to CPU before return (for pickle safety)
    """
    return original_run_inference(self, *args, **kwargs)

# Replace the original method with the GPU-decorated version
ModelInference.run_inference = gpu_run_inference

# Initialize and launch the app
if __name__ == "__main__":
    # Configure directories for Hugging Face Spaces
    model_dir = os.environ.get("DA3_MODEL_DIR", "depth-anything/DA3NESTED-GIANT-LARGE")
    workspace_dir = os.environ.get("DA3_WORKSPACE_DIR", "workspace/gradio")
    gallery_dir = os.environ.get("DA3_GALLERY_DIR", "workspace/gallery")
    
    # Create directories if they don't exist
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(gallery_dir, exist_ok=True)
    
    # Initialize the app
    app = DepthAnything3App(
        model_dir=model_dir,
        workspace_dir=workspace_dir,
        gallery_dir=gallery_dir
    )
    
    # Check if examples directory exists
    examples_dir = os.path.join(workspace_dir, "examples")
    examples_exist = os.path.exists(examples_dir)
    
    # Check if caching is enabled via environment variable (default: True if examples exist)
    # Allow disabling via environment variable: DA3_CACHE_EXAMPLES=false
    cache_examples_env = os.environ.get("DA3_CACHE_EXAMPLES", "").lower()
    if cache_examples_env in ("false", "0", "no"):
        cache_examples = False
    elif cache_examples_env in ("true", "1", "yes"):
        cache_examples = True
    else:
        # Default: enable caching if examples directory exists
        cache_examples = examples_exist
    
    # Get cache_gs_tag from environment variable (default: "dl3dv")
    cache_gs_tag = os.environ.get("DA3_CACHE_GS_TAG", "dl3dv")
    
    # Launch with Spaces-friendly settings
    print("🚀 Launching Depth Anything 3 on Hugging Face Spaces...")
    print(f"📦 Model Directory: {model_dir}")
    print(f"📁 Workspace Directory: {workspace_dir}")
    print(f"🖼️  Gallery Directory: {gallery_dir}")
    print(f"💾 Cache Examples: {cache_examples}")
    if cache_examples:
        if cache_gs_tag:
            print(f"🏷️  Cache GS Tag: '{cache_gs_tag}' (scenes matching this tag will use high-res + 3DGS)")
        else:
            print("🏷️  Cache GS Tag: None (all scenes will use low-res only)")
    
    # Pre-cache examples if requested
    if cache_examples:
        print("\n" + "=" * 60)
        print("Pre-caching mode enabled")
        if cache_gs_tag:
            print(f"Scenes containing '{cache_gs_tag}' will use HIGH-RES + 3DGS")
            print("Other scenes will use LOW-RES only")
        else:
            print("All scenes will use LOW-RES only")
        print("=" * 60)
        app.cache_examples(
            show_cam=True,
            filter_black_bg=False,
            filter_white_bg=False,
            save_percentage=5.0,
            num_max_points=1000,
            cache_gs_tag=cache_gs_tag,
            gs_trj_mode="smooth",
            gs_video_quality="low",
        )
    
    # Launch with minimal, Spaces-compatible configuration
    # Some parameters may cause routing issues, so we use minimal config
    app.launch(
        host="0.0.0.0",  # Required for Spaces
        port=7860,       # Standard Gradio port
        share=False      # Not needed on Spaces
    )
