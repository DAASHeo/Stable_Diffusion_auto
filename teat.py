import time
from contextlib import closing

import modules.scripts
from modules import processing
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.shared import opts, cmd_opts
import modules.shared as shared
from modules.ui import plaintext_to_html
import gradio as gr

import modules.sd_models as ch


def txt2img(id_task: str, prompt: str, negative_prompt: str,  prompt_styles, steps: int, sampler_name: str, n_iter: int, batch_size: int, cfg_scale: float, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, hr_checkpoint_name: str, hr_sampler_name: str, hr_prompt: str, hr_negative_prompt, override_settings_texts, request: gr.Request, *args):
    override_settings = create_override_settings_dict(override_settings_texts)

    Lora = ['<lora:jianying:1>', '<lora:bioluminescent_dress-1.0:1>, <lora:Vivid_Setting:1>, <lora:fanxing:0.7>']
    euler = ["rabbit_v15.safetensors [8e2cc3ee15]", "duchaitenAiartSDXL_v10.safetensors [882d0a68e2]"]
    D2MSK = ["sparklingDreamland_v10.safetensors [b49a736d7c]","LahMysteriousSDXL_v40.safetensors [da5ddce194]", "revAnimated_v122EOL.safetensors [4199bcdd14]"]
    D2SaK = ["hansChaosModelV2_v20.safetensors [d0477788ad]"]
    DSK = ["spectrumblendx_v20.safetensors [7cf9ed7d67]"]

    I = ['Elegant', 'Mysterious', 'Enchanting', 'Romantic', 'Fantastic']
    J = ['Camera Object', 'Cake', 'Gift boxes', 'Balloon', 'Castle']
    K = ['Party', 'Park', 'Beach', 'Sunset', 'Flower garden']
    V = ['Purple', 'Pink', 'Yellow', 'Green', 'Blue']
    W = ['Happiness', 'Sadness', 'Anger', 'Love', 'Surprise']

    # sen = ['happiness, flowers, yellow, star, bright, Amusement Park, with Balloons, Pink, Yellow',
    #        'dark, rain, bridge, top, between, under, water, lightning, Rain, Dark Clouds, Navy Blue',
    #        'red, mountain, dark, night, Sunset, Red, Airplane',
    #        'blue, water, pastel, raining, cloud, calm, cool, Snow, Mountain, Window',
    #        'pink, comfortable, enjoyable, yacht, sailboat, blue, shining, Flower, Grass, Day, Sun, Tree, Green',
    #        'surprise, colorful, right, firework, tower, top, top-view, Firework, Flame, at Night, Stars, Light']
    # W = ['Tranquility', 'Happiness', 'Surprise', 'Fear', 'Anger', 'Sadness']


    if not prompt:  # WebUI에서 입력된 prompt 값이 없을 경우

        # if(ch.select_checkpoint_name() in D2MSK): # checkpoint에 맞는 sampler 할당
        #     sampler_name = "DPM++ 2M SDE Karras"
        # elif(ch.select_checkpoint_name() in euler):
        #     sampler_name = "Euler a"
        # elif (ch.select_checkpoint_name() in D2SaK):
        #     sampler_name = "DPM++ 2S a Karras"
        # elif (ch.select_checkpoint_name() in DSK):
        #     sampler_name = "DPM++ SDE Karras"
        # else:
        #     sampler_name = "DPM++ 2M Karras"

        # for i in range(5):
        #     for j in range(5):
        #         for k in range(5):
        #             for v in range(5):
        #                 for w in range(5):
        #                     prompt = f'{I[i]},{J[j]},{K[k]},{V[v]},{W[w]},breathtaking 8k, masterpiece, detailed, realistic, 8k uhd, high quality, HDR, best quality, highres'
        start_i, start_j, start_k, start_v, start_w = 1, 2, 2, 4, 2

        for i in range(start_i, 5):
            for j in range(start_j if i == start_i else 0, 5):
                for k in range(start_k if j == start_j and i == start_i else 0, 5):
                    for v in range(start_v if k == start_k and j == start_j and i == start_i else 0, 5):
                        for w in range(start_w if v == start_v and k == start_k and j == start_j and i == start_i else 0, 5):
                            prompt = f'{I[i]},{J[j]},{K[k]},{V[v]},{W[w]},breathtaking 8k, masterpiece, detailed, realistic, 8k uhd, high quality, HDR, best quality, highres'
                            print(prompt)  # 출력하거나 필요한 작업을 수행
                            width = 960
                            sampler_name = "DPM++ 2M SDE Karras"
                            # height = 512
                            steps = 100 # 생성 이미지 sampling steps 설정
                            negative_prompt = "(low quality, worst quality:1.4), (NSFW:1.5),easynegative, naked, nude, bad-hands-5, deformation, (bad quality), (worst quality), watermark, (blurry), bad quality, deformed hands, deformed fingers, nostalgic, drawing, painting, bad anatomy, worst quality, blurry, blurred, normal quality, bad focus, tripod, three legs, weird legs, short legs, people, human, chridren, ng_deepnegative_v1_75t, easynegative, bad-picture-chill-75v,Multiple people,lowres, low quality, normal quality, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, (low quality, worst quality:1.4), bad_prompt"
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

                            time.sleep(5)

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