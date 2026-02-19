# التحقق من نوع النموذج
print(f"Model type: {type(model).__name__}")
print(f"Processor type: {type(processor).__name__}")

# تجربة صيغ مختلفة
prompts = [
    "USER: You are a radiologist. Analyze this chest X-ray and write a short medical report. State if pneumonia is suspected.\nASSISTANT:",
    "USER: <image>\nYou are a radiologist. Analyze this chest X-ray and write a short medical report. State if pneumonia is suspected.\nASSISTANT:",
    "Analyze this chest X-ray and write a short medical report. State if pneumonia is suspected."
]

for i, prompt in enumerate(prompts):
    try:
        print(f"\nTrying prompt {i+1}: {prompt[:50]}...")
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

        # إذا نجحت، استخدمها
        outputs = model.generate(**inputs, max_new_tokens=200)
        response = processor.decode(outputs[0], skip_special_tokens=True)
        print("Success! Response:", response)
        break
    except Exception as e:
        print(f"Failed with error: {type(e).__name__}")
        continue
