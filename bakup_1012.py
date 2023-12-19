import time
from contextlib import closing
import torch
import gc

import modules.scripts
from modules import processing
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.shared import opts, cmd_opts
import modules.shared as shared
from modules.ui import plaintext_to_html
import gradio as gr

import modules.sd_models as ch

#optimize these codes

def txt2img(id_task: str, prompt: str, negative_prompt: str,  prompt_styles, steps: int, sampler_name: str, n_iter: int, batch_size: int, cfg_scale: float, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, hr_checkpoint_name: str, hr_sampler_name: str, hr_prompt: str, hr_negative_prompt, override_settings_texts, request: gr.Request, *args):
    override_settings = create_override_settings_dict(override_settings_texts)

    gc.collect()
    torch.cuda.empty_cache()
    I = ['Elegant', 'Mysterious', 'Enchanting', 'Romantic', 'Fantastic']
    J = ['Amusementpark', 'Park', 'Beach', 'Sunset', 'Flower garden']
    K = ['(1 Airplane:1.2)', '(1 Cake:1.3)', '(Firework spectacle:1.3)', '(beautiful Fairy tale Castle:1.3)', '(Gift boxes:1.3)']
    V = ['Purple', 'Pink', 'Yellow', 'Green', 'Blue']
    W = ['Happiness', 'Sadness', 'Anger', 'Love', 'Surprise']

    if not prompt:  # WebUI에서 입력된 prompt 값이 없을 경우
        start_i, start_j, start_k, start_v, start_w = 0, 0, 0, 0, 0

        for i in range(start_i, 5):
            for j in range(start_j if i == start_i else 0, 5):
                for k in range(start_k if j == start_j and i == start_i else 0, 5):
                    for v in range(start_v if k == start_k and j == start_j and i == start_i else 0, 5):
                        for w in range(
                                start_w if v == start_v and k == start_k and j == start_j and i == start_i else 0, 5):
                                prompt = f' Illustration, (masterpiece, best quality, CGI:1.2), the {I[i]} scene at a {J[j]}, sense of {W[w]}, (without any people:1.4), with {K[k]} in {V[v]}, breathtaking 8k, masterpiece, detailed, Illustration, 8k uhd, wallpaper, no humans, no text, high quality, HDR, best quality, highres, adorable, kawaii, highly detailed, beautiful, <lora:add_detail:1>'
                                print(prompt)
                                print(i, j, k, v, w)  # 출력하거나 필요한 작업을 수행
                                width = 960
                                sampler_name = "DPM++ 3M SDE Karras"
                                steps = 100  # 생성 이미지 sampling steps 설정
                                negative_prompt = "(low quality, worst quality:1.4), (NSFW:1.5),Easynegative, watermark, ugly, naked, nude, bad-hands-5, deformation, (bad quality), (worst quality), watermark, (blurry), bad quality, deformed hands, deformed fingers, nostalgic, drawing, painting, bad anatomy, worst quality, blurry, blurred, normal quality, bad focus, tripod, three legs, weird legs, short legs, people, human, chridren, ng_deepnegative_v1_75t, easynegative, bad-picture-chill-75v,Multiple people,lowres, low quality, normal quality, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, (low quality, worst quality:1.4), bad_prompt"
                                p = processing.StableDiffusionProcessingTxt2Img(
                                    sd_model=shared.sd_model,
                                    outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
                                    outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
                                    prompt=prompt,
                                    styles=prompt_styles,
                                    negative_prompt=negative_prompt,
                                    sampler_name=sampler_name,
                                    batch_size=batch_size,
                                    n_iter=n_iter,
                                    steps=steps,
                                    cfg_scale=cfg_scale,
                                    width=width,
                                    height=height,
                                    enable_hr=enable_hr,
                                    denoising_strength=denoising_strength if enable_hr else None,
                                    hr_scale=hr_scale,
                                    hr_upscaler=hr_upscaler,
                                    hr_second_pass_steps=hr_second_pass_steps,
                                    hr_resize_x=hr_resize_x,
                                    hr_resize_y=hr_resize_y,
                                    hr_checkpoint_name=None if hr_checkpoint_name == 'Use same checkpoint' else hr_checkpoint_name,
                                    hr_sampler_name=None if hr_sampler_name == 'Use same sampler' else hr_sampler_name,
                                    hr_prompt=hr_prompt,
                                    hr_negative_prompt=hr_negative_prompt,
                                    override_settings=override_settings,
                                )

                                p.scripts = modules.scripts.scripts_txt2img
                                p.script_args = args

                                p.user = request.username

                                if cmd_opts.enable_console_prompts:
                                    print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

                                with closing(p):
                                    processed = modules.scripts.scripts_txt2img.run(p, *args)

                                    if processed is None:
                                        processed = processing.process_images(p)

                                shared.total_tqdm.clear()

                                generation_info_js = processed.js()
                                if opts.samples_log_stdout:
                                    print(generation_info_js)

                                if opts.do_not_show_images:
                                    processed.images = []

                                del p
                                gc.collect()
                                torch.cuda.empty_cache() # GPU 캐시 정리


        return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")

    else: #WebUI로 전달된 prompt값 존재할 경우
        p = processing.StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
            outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
            prompt=prompt,
            styles=prompt_styles,
            negative_prompt=negative_prompt,
            sampler_name=sampler_name,
            batch_size=batch_size,
            n_iter=n_iter,
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            enable_hr=enable_hr,
            denoising_strength=denoising_strength if enable_hr else None,
            hr_scale=hr_scale,
            hr_upscaler=hr_upscaler,
            hr_second_pass_steps=hr_second_pass_steps,
            hr_resize_x=hr_resize_x,
            hr_resize_y=hr_resize_y,
            hr_checkpoint_name=None if hr_checkpoint_name == 'Use same checkpoint' else hr_checkpoint_name,
            hr_sampler_name=None if hr_sampler_name == 'Use same sampler' else hr_sampler_name,
            hr_prompt=hr_prompt,
            hr_negative_prompt=hr_negative_prompt,
            override_settings=override_settings,
        )

        p.scripts = modules.scripts.scripts_txt2img
        p.script_args = args

        p.user = request.username

        if cmd_opts.enable_console_prompts:
            print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

        with closing(p):
            processed = modules.scripts.scripts_txt2img.run(p, *args)

            if processed is None:
                processed = processing.process_images(p)

        shared.total_tqdm.clear()

        generation_info_js = processed.js()
        if opts.samples_log_stdout:
            print(generation_info_js)

        if opts.do_not_show_images:
            processed.images = []

        return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")