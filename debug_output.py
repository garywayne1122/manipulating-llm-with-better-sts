"""
DEBUG SCRIPT — dekhta hai model ACTUALLY kya generate kar raha hai
====================================================================
Isko run karo taake pata chale rank hamesha 11/12 kyun aa raha hai.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CATALOG = [
    {"name": "ColdBrew Master Coffee Machine", "desc": "A compact coffee machine designed for smooth cold brew extraction at home. Easy to clean and durable build."},
    {"name": "QuickBrew Express Coffee Machine", "desc": "A fast single-serve coffee machine ideal for busy mornings. Compact design fits any kitchen counter."},
    {"name": "AromaPro Espresso Maker", "desc": "Delivers rich espresso shots with adjustable pressure control and a built-in milk frother."},
    {"name": "BrewMaster Deluxe", "desc": "A premium drip coffee maker with programmable timer and thermal carafe for all-day freshness."},
    {"name": "SteamJet Coffee Press", "desc": "Manual French press style brewer, stainless steel, ideal for camping or small kitchens."},
    {"name": "EcoBrew Sustainable Maker", "desc": "Made from recycled materials, this eco-conscious coffee maker reduces plastic waste."},
    {"name": "VelvetPour Coffee Kettle", "desc": "Gooseneck kettle designed for precision pour-over brewing with temperature control."},
    {"name": "TurboGrind Coffee Bundle", "desc": "Includes a burr grinder and drip machine combo for the complete home brewing setup."},
    {"name": "MiniBrew Travel Maker", "desc": "Portable coffee maker for travel, USB rechargeable, brews a single cup in minutes."},
    {"name": "ClassicDrip Home Brewer", "desc": "A no-frills reliable drip coffee maker for everyday use, 12-cup capacity."},
]
TARGET_PRODUCT = "ColdBrew Master Coffee Machine"
USER_QUERY = "I'm looking for a good coffee machine for my home kitchen. What do you recommend?"

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16).to(DEVICE)
model.eval()

system = (
    "You are a helpful product recommendation assistant. "
    "Given a user query and a product catalog, return a ranked list of products "
    "from most to least relevant, in the format:\n1. <Product Name>\n2. <Product Name>\n..."
)
catalog_text = "".join(f"- {p['name']}: {p['desc']}\n" for p in CATALOG)
user_msg = f"User query: {USER_QUERY}\n\nProduct catalog:\n{catalog_text}\nReturn the ranked list."

messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(DEVICE)

print(f"\nPrompt token length: {inputs['input_ids'].shape[1]}")

with torch.no_grad():
    output = model.generate(
        **inputs, max_new_tokens=400, temperature=0.6, top_p=0.9,
        do_sample=True, pad_token_id=tokenizer.eos_token_id
    )
generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print("\n" + "=" * 60)
print("RAW GENERATED OUTPUT:")
print("=" * 60)
print(generated)
print("=" * 60)
print(f"\nTarget product name: '{TARGET_PRODUCT}'")
print(f"Target name found in output (case-insensitive)? {TARGET_PRODUCT.lower() in generated.lower()}")
print(f"\nGenerated text line count: {len(generated.strip().split(chr(10)))}")
print(f"Total tokens generated: {output[0].shape[0] - inputs['input_ids'].shape[1]}")
