from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "sshleifer/tiny-gpt2"   # Correct tiny model

print("Loading model… this may take a few minutes. Please wait!")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Chatbot ready! Type 'exit' to stop.\n")

while True:
    prompt = input("You: ")
    if prompt.lower() == "exit":
        break
        
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )
    
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Bot:", reply)
