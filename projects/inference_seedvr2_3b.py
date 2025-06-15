# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import os
import torch
import mediapy
from einops import rearrange
from omegaconf import OmegaConf
print(os.getcwd())
import datetime
from tqdm import tqdm
import gc


from data.image.transforms.divisible_crop import DivisibleCrop
from data.image.transforms.na_resize import NaResize
from data.video.transforms.rearrange import Rearrange
if os.path.exists("./projects/video_diffusion_sr/color_fix.py"):
    from projects.video_diffusion_sr.color_fix import wavelet_reconstruction
    use_colorfix=True
else:
    use_colorfix = False
    print('Note!!!!!! Color fix is not avaliable!')
from torchvision.transforms import Compose, Lambda, Normalize
from torchvision.io.video import read_video
import argparse


from common.distributed import (
    get_device,
    init_torch,
)

from common.distributed.advanced import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    init_sequence_parallel,
)

from projects.video_diffusion_sr.infer import VideoDiffusionInfer
from common.config import load_config
from common.distributed.ops import sync_data
from common.seed import set_seed
from common.partition import partition_by_groups, partition_by_size


def configure_sequence_parallel(sp_size):
    if sp_size > 1:
        init_sequence_parallel(sp_size)

def configure_runner(sp_size):
    config_path = os.path.join('./configs_3b', 'main.yaml')
    config = load_config(config_path)
    runner = VideoDiffusionInfer(config)
    OmegaConf.set_readonly(runner.config, False)
    
    init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
    configure_sequence_parallel(sp_size)
    runner.configure_dit_model(device="cuda", checkpoint='./ckpts/seedvr2_ema_3b.pth')
    runner.configure_vae_model()
    # Set memory limit.
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
    return runner

def generation_step(runner, text_embeds_dict, cond_latents):
    def _move_to_cuda(x):
        return [i.to(get_device()) for i in x]

    noises = [torch.randn_like(latent) for latent in cond_latents]
    aug_noises = [torch.randn_like(latent) for latent in cond_latents]
    print(f"Generating with noise shape: {noises[0].size()}.")
    noises, aug_noises, cond_latents = sync_data((noises, aug_noises, cond_latents), 0)
    noises, aug_noises, cond_latents = list(
        map(lambda x: _move_to_cuda(x), (noises, aug_noises, cond_latents))
    )
    cond_noise_scale = 0.0

    def _add_noise(x, aug_noise):
        t = (
            torch.tensor([1000.0], device=get_device())
            * cond_noise_scale
        )
        shape = torch.tensor(x.shape[1:], device=get_device())[None]
        t = runner.timestep_transform(t, shape)
        print(
            f"Timestep shifting from"
            f" {1000.0 * cond_noise_scale} to {t}."
        )
        x = runner.schedule.forward(x, aug_noise, t)
        return x

    conditions = [
        runner.get_condition(
            noise,
            task="sr",
            latent_blur=_add_noise(latent_blur, aug_noise),
        )
        for noise, aug_noise, latent_blur in zip(noises, aug_noises, cond_latents)
    ]

    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
        video_tensors = runner.inference(
            noises=noises,
            conditions=conditions,
            dit_offload=True,
            **text_embeds_dict,
        )

    samples = [
        (
            rearrange(video[:, None], "c t h w -> t c h w")
            if video.ndim == 3
            else rearrange(video, "c t h w -> t c h w")
        )
        for video in video_tensors
    ]
    del video_tensors

    return samples

# video_path is now a file, output_dir is replaced by output_video_path
def generation_loop(runner, video_path, output_video_path, batch_size=1, cfg_scale=1.0, cfg_rescale=0.0, sample_steps=1, seed=666, res_h=1280, res_w=720, sp_size=1):

    def _build_pos_and_neg_prompt():
        # read positive prompt
        positive_text = "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, \
        hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, \
        skin pore detailing, hyper sharpness, perfect without deformations."
        # read negative prompt
        negative_text = "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, \
        CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, \
        signature, jpeg artifacts, deformed, lowres, over-smooth"
        return positive_text, negative_text

    def _build_test_prompts(input_video_file_path): # Takes a single video file path
        positive_text, negative_text = _build_pos_and_neg_prompt()
        # We are processing a single video, so original_videos will contain just its path
        original_video_paths_list = [input_video_file_path] # Store the full path
        prompts = {os.path.basename(input_video_file_path): positive_text}
        print(f"Processing video: {input_video_file_path}")
        return original_video_paths_list, prompts, negative_text

    # original_videos_local_paths is expected to be a list of lists of file paths, e.g., [['/path/to/video.mp4']]
    def _extract_text_embeds(original_videos_local_paths_arg):
        # Text encoder forward.
        positive_prompts_embeds = []
        # The loop `original_videos_local_paths_arg` will iterate over batches of video file paths.
        for batch_of_video_paths in tqdm(original_videos_local_paths_arg):
            # We only care about the existence of a video to process for loading these generic embeddings.
            if batch_of_video_paths: # Check if the batch is not empty (it should contain one video path)
                # Load embeddings from the ./ckpts directory (relative to CWD of script)
                # This assumes snapshot_download in handler.py places them in MODEL_DIR (./ckpts)
                text_pos_embeds = torch.load(os.path.join('./ckpts', 'pos_emb.pt'))
                text_neg_embeds = torch.load(os.path.join('./ckpts', 'neg_emb.pt'))
                positive_prompts_embeds.append(
                    {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}
                )
        gc.collect()
        torch.cuda.empty_cache()
        return positive_prompts_embeds

    def cut_videos(videos, sp_size):
        t = videos.size(1)
        if t <= 4 * sp_size:
            print(f"Cut input video size: {videos.size()}")
            padding = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - t + 1)
            padding = torch.cat(padding, dim=1)
            videos = torch.cat([videos, padding], dim=1)
            return videos
        if (t - 1) % (4 * sp_size) == 0:
            return videos
        else:
            padding = [videos[:, -1].unsqueeze(1)] * (
                4 * sp_size - ((t - 1) % (4 * sp_size))
            )
            padding = torch.cat(padding, dim=1)
            videos = torch.cat([videos, padding], dim=1)
            assert (videos.size(1) - 1) % (4 * sp_size) == 0
            return videos

    # classifier-free guidance
    runner.config.diffusion.cfg.scale = cfg_scale
    runner.config.diffusion.cfg.rescale = cfg_rescale
    # sampling steps
    runner.config.diffusion.timesteps.sampling.steps = sample_steps
    runner.configure_diffusion()

    # set random seed
    set_seed(seed, same_across_ranks=True)
    # output_dir is now output_video_path's directory. Ensure its parent directory exists.
    # This was output_dir, now it's the directory of output_video_path
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    # tgt_path is removed, use output_video_path directly later

    # get test prompts - video_path is now a single file path
    # original_video_filepaths_list will be a list with one item: the full video_path
    original_video_filepaths_list, _, _ = _build_test_prompts(video_path)

    # divide the video file path(s) into different groups for data parallelism.
    # For a single video, and typical serverless single-GPU setup, this simplifies.
    # get_data_parallel_world_size() and get_sequence_parallel_world_size() are likely 1.
    # The code expects original_videos_local_paths to be a list of lists (batches).
    if get_data_parallel_world_size() > 1 or get_sequence_parallel_world_size() > 1:
        # Keep original partitioning logic if distributed; original_video_filepaths_list is like ['/path/to/video.mp4']
        original_videos_group = partition_by_groups(
            original_video_filepaths_list, # List containing the single video path
            get_data_parallel_world_size() // get_sequence_parallel_world_size(),
        )
        # original_videos_local_paths will be this node's share of video paths
        original_videos_local_paths = original_videos_group[
            get_data_parallel_rank() // get_sequence_parallel_world_size()
        ]
        # partition_by_size further breaks it into batches. If batch_size=1, it's like [['/path/to/video.mp4']]
        original_videos_local_paths = partition_by_size(original_videos_local_paths, batch_size)
    else:
        # Non-distributed case: a list containing one batch, which contains the one video path
        # video_path is the direct string path to the video.
        original_videos_local_paths = [[video_path]] # e.g. [['/app/output/input_video_xyz.mp4']]

    # pre-extract the text embeddings
    positive_prompts_embeds = _extract_text_embeds(original_videos_local_paths)

    video_transform = Compose(
        [
            NaResize(
                resolution=(
                    res_h * res_w
                )
                ** 0.5,
                mode="area",
                # Upsample image, model only trained for high res.
                downsample_only=False,
            ),
            Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            DivisibleCrop((16, 16)),
            Normalize(0.5, 0.5),
            Rearrange("t c h w -> c t h w"),
        ]
    )

    # generation loop
    # original_videos_local_paths is e.g. [['/app/output/input_video_xyz.mp4']]
    # positive_prompts_embeds corresponds to this structure.
    for video_filepath_batch, text_embeds_for_batch in tqdm(zip(original_videos_local_paths, positive_prompts_embeds)):
        # video_filepath_batch is a list of video file paths, e.g., ['/app/output/input_video_xyz.mp4']
        cond_latents = []
        # This inner loop will iterate once if batch_size is 1 (as it is by default from handler.py)
        for single_video_filepath in video_filepath_batch:
            video_frames_tensor = ( # Renamed 'video' (original name of loop var) to 'video_frames_tensor'
                read_video(
                   single_video_filepath, output_format="TCHW" # Use the direct full filepath
                )[0]
                / 255.0
            )
            print(f"Read video size: {video_frames_tensor.size()}")
            cond_latents.append(video_transform(video_frames_tensor.to(get_device())))

        ori_lengths = [vf_tensor.size(1) for vf_tensor in cond_latents] # Use renamed var from loop
        input_videos = cond_latents # These are the transformed, original video tensors for the batch
        # Apply cut_videos to each tensor in cond_latents
        cond_latents = [cut_videos(vf_tensor, sp_size) for vf_tensor in cond_latents] # Use renamed var

        runner.dit.to("cpu")
        print(f"Encoding videos: {list(map(lambda x: x.size(), cond_latents))}")
        runner.vae.to(get_device())
        cond_latents = runner.vae_encode(cond_latents)
        runner.vae.to("cpu")
        runner.dit.to(get_device())

        # Pass text_embeds_for_batch to generation_step
        for i, emb in enumerate(text_embeds_for_batch["texts_pos"]):
            text_embeds_for_batch["texts_pos"][i] = emb.to(get_device())
        for i, emb in enumerate(text_embeds_for_batch["texts_neg"]):
            text_embeds_for_batch["texts_neg"][i] = emb.to(get_device())

        samples = generation_step(runner, text_embeds_for_batch, cond_latents=cond_latents)
        runner.dit.to("cpu")
        del cond_latents

        # dump samples to the output directory
        if get_sequence_parallel_rank() == 0:
            # video_filepath_batch contains the file path(s) of the current batch (e.g., ['/app/output/input_video_xyz.mp4'])
            # input_videos contains the corresponding original video tensors (transformed) for the batch.
            # samples contains the generated video tensors for the batch.
            # ori_lengths contains the original lengths for the batch.
            for original_video_filepath, processed_input_tensor, generated_sample_tensor, original_length in zip(
                video_filepath_batch, input_videos, samples, ori_lengths
            ):
                if original_length < generated_sample_tensor.shape[0]:
                    generated_sample_tensor = generated_sample_tensor[:original_length]

                # filename is now the exact output_video_path specified in args
                # This assumes one video processed per call to generation_loop as per overall design with handler.py
                output_filename = output_video_path

                # color fix using processed_input_tensor (which is an item from input_videos list)
                input_tensor_for_colorfix = (
                    rearrange(processed_input_tensor[:, None], "c t h w -> t c h w")
                    if processed_input_tensor.ndim == 3
                    else rearrange(processed_input_tensor, "c t h w -> t c h w")
                )
                if use_colorfix:
                    generated_sample_tensor = wavelet_reconstruction(
                        generated_sample_tensor.to("cpu"), input_tensor_for_colorfix[: generated_sample_tensor.size(0)].to("cpu")
                    )
                else:
                    generated_sample_tensor = generated_sample_tensor.to("cpu")
                generated_sample_tensor = ( # Renaming 'sample' to 'generated_sample_tensor' for clarity
                    rearrange(generated_sample_tensor[:, None], "t c h w -> t h w c")
                    if generated_sample_tensor.ndim == 3
                    else rearrange(generated_sample_tensor, "t c h w -> t h w c")
                )
                generated_sample_tensor = generated_sample_tensor.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round()
                # This is the line that was duplicated due to incorrect search block end.
                # The following lines correctly use generated_sample_tensor and output_filename.
                generated_sample_tensor = generated_sample_tensor.to(torch.uint8).numpy()

                if generated_sample_tensor.shape[0] == 1:
                    mediapy.write_image(output_filename, generated_sample_tensor.squeeze(0))
                else:
                    mediapy.write_video(
                        output_filename, generated_sample_tensor, fps=24
                    )
        # gc.collect() and torch.cuda.empty_cache() should be outside the loop over samples in a batch
        # but inside the loop over batches. The original indentation seems correct for this.
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Changed video_path to expect a file, not a directory
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    # Changed output_dir to output_video_path for a specific file output
    parser.add_argument("--output_video_path", type=str, required=True, help="Path to save the processed video file")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--res_h", type=int, default=720)
    parser.add_argument("--res_w", type=int, default=1280)
    parser.add_argument("--sp_size", type=int, default=1)
    args = parser.parse_args()

    # The generation_loop call will be changed in a subsequent diff application
    runner = configure_runner(args.sp_size)
    # generation_loop signature is now updated, so call it with the new argument names.
    args_dict = vars(args).copy()
    video_path_arg = args_dict.pop('video_path')
    output_video_path_arg = args_dict.pop('output_video_path')
    # output_dir argument is no longer part of generation_loop or args

    runner = configure_runner(args.sp_size) # sp_size is still in args_dict
    generation_loop(runner, video_path=video_path_arg, output_video_path=output_video_path_arg, **args_dict)
