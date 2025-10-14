import onnxruntime_genai as og

model = og.Model("gemma-3-270m")
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

# Set the max length to something sensible by default,
# since otherwise it will be set to the entire context length
search_options = {}
search_options["max_length"] = 2048
search_options["batch_size"] = 1

text = input("Input: ")
if not text:
    print("Error, input cannot be empty")
    exit()

prompt = tokenizer.apply_chat_template(
    messages=f"""[{{"role": "user", "content": "{text}"}}]""",
    add_generation_prompt=True,
)

input_tokens = tokenizer.encode(prompt)

params = og.GeneratorParams(model)
params.set_search_options(**search_options)
generator = og.Generator(model, params)

print("Output: ", end="", flush=True)

try:
    generator.append_tokens(input_tokens)
    while not generator.is_done():
        generator.generate_next_token()

        new_token = generator.get_next_tokens()[0]
        print(tokenizer_stream.decode(new_token), end="", flush=True)
except KeyboardInterrupt:
    print("  --control+c pressed, aborting generation--")

print()
