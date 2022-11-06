import os
from webui import wrap_gradio_gpu_call
from modules import scripts, script_callbacks
from modules import shared, devices, sd_hijack, processing, sd_models, images, ui
from modules.shared import opts, cmd_opts, restricted_opts
from modules.ui import create_output_panel, setup_progressbar, create_refresh_button
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, \
    StableDiffusionProcessingImg2Img, process_images
from modules.ui import plaintext_to_html
from modules.textual_inversion.textual_inversion import save_embedding
import gradio as gr
import gradio.routes
import gradio.utils
import torch

# ISSUES
# distribution shouldn't be fetched until the first embedding is opened
# most functions need to verify that an embedding is selected
# vector numbers aren't verified (might be better as a slider)
# weight slider values are lost when changing vector number
# remove unused imports
#
# TODO
# add tagged positions on sliders from user-supplied words (and unique symbols & colours)
# add a word->substrings printout for use with the above for words which map to multiple embeddings (e.g. "computer" = "compu" and "ter")
# add the ability to create embeddings which are a mix of other embeddings (with ratios), e.g. 0.5 * skunk + 0.5 * puppy is a valid embedding
# add the ability to shift all weights towards another embedding with a master slider
# add a strength slider (multiply all weights)
# print out the closest word(s) in the original embeddings list to the current embedding, with torch.abs(embedding1.vec - embedding2.vec).mean() or maybe sum
# also maybe print a mouseover or have an expandable info box per weight slider for the closest embedding(s) for that weight value
# maybe add per-weight notes, and possibly a way to save them per embedding (and vector), or save them under a class name (e.g. 'animal', 'clothing', 'face')
# add option to vary individual weights one at a time and generate outputs, potentially also varied combinations of weights. Potentially use scoring system to determine size of change (maybe latents or clip interrogator)
# add option to 'move' around current embedding position and generate outputs (a 768-dimensional vector spiral)?
# potentially represent all weights 2 or 3 magnitudes larger, so that it's more obvious when a weight is larger than most or very small


def determine_embedding_distribution():
    cond_model = shared.sd_model.cond_stage_model
    embedding_layer = cond_model.wrapped.transformer.text_model.embeddings
    
    distribution_floor = torch.zeros(768)
    distribution_ceiling = torch.zeros(768)
    
    for i in range(49405): # guessing that's the range of CLIP tokens given that 49406 and 49407 are special tokens presumably appended to the end
        embedding = embedding_layer.token_embedding.wrapped(torch.LongTensor([i]).to(devices.device)).squeeze(0)
        if i == 0:
            distribution_floor = embedding
            distribution_ceiling = embedding
        else:
            distribution_floor = torch.minimum(distribution_floor, embedding)
            distribution_ceiling = torch.maximum(distribution_ceiling, embedding)
    
    return distribution_floor, distribution_ceiling

def build_slider(index, default, distribution_floor, distribution_ceiling, weight_sliders):
    floor = distribution_floor[index].item()
    ceil = distribution_ceiling[index].item()
    slider = gr.Slider(minimum=floor, maximum=ceil, step=0.00001, label=f"w{index}", value=default, interactive=True)
    
    weight_sliders.append(slider)

def on_ui_tabs():
    distribution_floor, distribution_ceiling = determine_embedding_distribution()
    weight_sliders = []

    with gr.Blocks(analytics_enabled=False) as embedding_editor_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel', scale=1.5):
                with gr.Column():
                    with gr.Row():
                        embedding_name = gr.Dropdown(label='Embedding', elem_id="edit_embedding", choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys()), interactive=True)
                        vector_num = gr.Number(label='Vector', value=0, step=1, interactive=True)
                        refresh_embeddings_button = gr.Button(value="Refresh Embeddings", variant='secondary')
                        save_embedding_button = gr.Button(value="Save Embedding", variant='primary')
                    
                    for i in range(0, 128):
                        with gr.Row():
                            build_slider(i*6+0, 0, distribution_floor, distribution_ceiling, weight_sliders)
                            build_slider(i*6+1, 0, distribution_floor, distribution_ceiling, weight_sliders)
                            build_slider(i*6+2, 0, distribution_floor, distribution_ceiling, weight_sliders)
                            build_slider(i*6+3, 0, distribution_floor, distribution_ceiling, weight_sliders)
                            build_slider(i*6+4, 0, distribution_floor, distribution_ceiling, weight_sliders)
                            build_slider(i*6+5, 0, distribution_floor, distribution_ceiling, weight_sliders)
            
            with gr.Column(scale=1):
                gallery = gr.Gallery(label='Output', show_label=False, elem_id="embedding_editor_gallery").style(grid=4)
                prompt = gr.Textbox(label="Prompt", elem_id=f"embedding_editor_prompt", show_label=False, lines=2, placeholder="e.g. A portrait photo of embedding_name" )
                batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1)
                steps = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=20)
                cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0)
                seed =(gr.Textbox if cmd_opts.use_textbox_seed else gr.Number)(label='Seed', value=-1)
                
                with gr.Row():
                    generate_preview = gr.Button(value="Generate Preview", variant='primary')
                
                generation_info = gr.HTML()
                html_info = gr.HTML()
    
        preview_args = dict(
            fn=wrap_gradio_gpu_call(generate_embedding_preview),
            _js="submit",
            inputs=[
                embedding_name,
                vector_num,
                prompt,
                steps,
                cfg_scale,
                seed,
                batch_count,
            ] + weight_sliders,
            outputs=[
                gallery,
                generation_info,
                html_info
            ],
            show_progress=False,
        )
        
        generate_preview.click(**preview_args)
        
        selection_args = dict(
            fn=select_embedding,
            inputs=[
                embedding_name,
                vector_num,
            ],
            outputs = weight_sliders,
        )
        
        embedding_name.change(**selection_args)
        vector_num.change(**selection_args)
        
        def refresh_embeddings():
            sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings() # refresh_method
            
            refreshed_args = lambda: {"choices": sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())} # refreshed_args
            args = refreshed_args() if callable(refreshed_args) else refreshed_args
            
            for k, v in args.items():
                setattr(embedding_name, k, v)
            
            return gr.update(**(args or {}))
        
        refresh_embeddings_button.click(
            fn=refresh_embeddings,
            inputs=[],
            outputs=[embedding_name]
        )
        
        save_embedding_button.click(
            fn=save_embedding_weights,
            inputs=[
                embedding_name,
                vector_num,
            ] + weight_sliders,
            outputs=[],
        )
    
    return [(embedding_editor_interface, "Embedding Editor", "embedding_editor_interface")]

def select_embedding(embedding_name, vector_num):
    embedding = sd_hijack.model_hijack.embedding_db.word_embeddings[embedding_name]
    vec = embedding.vec[int(vector_num)]
    weights = []
    
    for i in range(0, 768):
        weights.append( vec[i].item() )
    
    return weights

def update_embedding_weights(embedding_name, vector_num, weights):
    embedding = sd_hijack.model_hijack.embedding_db.word_embeddings[embedding_name]
    vec = embedding.vec[int(vector_num)]
    old_weights = []
    
    for i in range(0, 768):
        old_weights.append(vec[i].item())
        vec[i] = weights[i]
    
    return old_weights

def generate_embedding_preview(embedding_name, vector_num, prompt: str, steps: int, cfg_scale: float, seed: int, batch_count: int, *weights):
    old_weights = update_embedding_weights(embedding_name, vector_num, weights)
    
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        seed=seed,
        steps=steps,
        cfg_scale=cfg_scale,
        n_iter=batch_count,
    )
    
    if cmd_opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)
    
    processed = process_images(p)

    p.close()

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)
    
    update_embedding_weights(embedding_name, vector_num, old_weights) # restore
    
    return processed.images, generation_info_js, plaintext_to_html(processed.info)

def save_embedding_weights(embedding_name, vector_num, *weights):
    update_embedding_weights(embedding_name, vector_num, weights)
    embedding = sd_hijack.model_hijack.embedding_db.word_embeddings[embedding_name]
    checkpoint = sd_models.select_checkpoint()
    
    filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')
    save_embedding(embedding, checkpoint, embedding_name, filename, remove_cached_checksum=True)



script_callbacks.on_ui_tabs(on_ui_tabs)
