import streamlit as st
import torch
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from PIL import Image

# Load Model and Processor
@st.cache_resource
def load_model_and_processor(model_path="deepseek-ai/Janus-1.3B"):
    config = AutoConfig.from_pretrained(model_path)
    language_config = config.language_config
    language_config._attn_implementation = 'eager'
    
    vl_gpt_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        language_config=language_config,
        trust_remote_code=True
    )
    vl_gpt_model = vl_gpt_model.to(torch.bfloat16 if torch.cuda.is_available() else torch.float16)
    if torch.cuda.is_available():
        vl_gpt_model = vl_gpt_model.cuda()

    vl_chat_proc = VLChatProcessor.from_pretrained(model_path)
    return vl_gpt_model, vl_chat_proc

vl_gpt, vl_chat_processor = load_model_and_processor()
tokenizer = vl_chat_processor.tokenizer
cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Image Generation Support Functions
@torch.inference_mode()
def generate(
    input_ids,
    width,
    height,
    temperature=1.0,
    parallel_size=5,
    cfg_weight=5.0,
    image_token_num_per_image=576,
    patch_size=16
):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(cuda_device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(cuda_device)
    
    pkv = None
    for i in range(image_token_num_per_image):
        outputs = vl_gpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=pkv
        )
        pkv = outputs.past_key_values
        hidden_states = outputs.last_hidden_state
        logits = vl_gpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)
    
    patches = vl_gpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, width // patch_size, height // patch_size]
    )
    return generated_tokens.to(dtype=torch.int), patches

@torch.inference_mode()
def generate_image(prompt, seed=None, guidance=5):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    
    width, height, parallel_size = 384, 384, 5
    messages = [{'role': 'User', 'content': prompt}, {'role': 'Assistant', 'content': ''}]
    text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=messages,
        sft_format=vl_chat_processor.sft_format,
        system_prompt=''
    )
    text += vl_chat_processor.image_start_tag
    
    input_ids = torch.LongTensor(tokenizer.encode(text))
    _, patches = generate(
        input_ids,
        width // 16 * 16,
        height // 16 * 16,
        cfg_weight=guidance,
        parallel_size=parallel_size
    )
    
    images = patches.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    images = np.clip((images + 1) / 2 * 255, 0, 255).astype(np.uint8)
    pil_images = [Image.fromarray(images[i]).resize((1024, 1024), Image.LANCZOS) for i in range(parallel_size)]
    return pil_images

# Streamlit UI

def main():
    st.title("Janus - Text-to-Image Generation")
    st.subheader("Generate Images From Text")
    prompt = st.text_area("Prompt", value="A cute baby fox in autumn leaves, digital art, cinematic lighting...")
    
    seed_t2i = st.number_input("Seed (Optional)", min_value=0, value=12345, step=1)
    cfg_weight = st.slider("CFG Weight", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
    
    if st.button("Generate Images"):
        with st.spinner('Generating images... This may take a minute.'):
            images = generate_image(prompt=prompt, seed=seed_t2i, guidance=cfg_weight)
        
        st.write("Generated Images:")
        cols = st.columns(2)
        idx = 0
        for i in range(2):
            for j in range(2):
                if idx < len(images):
                    with cols[j]:
                        st.image(images[idx], use_column_width=True)
                idx += 1

if __name__ == "__main__":
    main()
