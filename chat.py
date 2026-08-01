import mmap
import os
import struct
import time
from io import BytesIO

import numpy as np
from flask import Flask, Response, render_template_string, request
from PIL import Image
from transformers import AutoProcessor

app = Flask(__name__)
print("Loading Moonshot Kimi-K3 Processor...")
processor = AutoProcessor.from_pretrained("moonshotai/kimi-k3", trust_remote_code=True)

SHM_SIZE = 64 * 1024 * 1024  # 64MB
shm_path = "/dev/shm/kimi_k3_shm" if os.name != "nt" else "/tmp/kimi_k3_shm"
try:
    with open(shm_path, "wb") as f:
        f.write(b"\x00" * SHM_SIZE)
except Exception:
    pass

shm_file = open(shm_path, "r+b")
shm = mmap.mmap(shm_file.fileno(), SHM_SIZE)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Kimi-K3 Chat</title>
    <style>
        body { font-family: sans-serif; margin: 40px; }
        #chat { border: 1px solid #ccc; padding: 20px; height: 400px; overflow-y: auto; margin-bottom: 20px; white-space: pre-wrap; }
        #controls { display: flex; gap: 10px; }
        #textInput { flex-grow: 1; padding: 10px; }
        button { padding: 10px 20px; cursor: pointer; }
    </style>
</head>
<body>
    <h2>Kimi-K3 tensor_graphs Integration</h2>
    <div id="chat"></div>
    <div id="controls">
        <input type="file" id="imageInput" accept="image/*">
        <input type="text" id="textInput" placeholder="Message...">
        <button onclick="send()">Send</button>
    </div>
    <script>
        async function send() {
            let text = document.getElementById('textInput').value;
            let fileInput = document.getElementById('imageInput');
            let formData = new FormData();
            formData.append("text", text);
            if (fileInput.files.length > 0) {
                formData.append("image", fileInput.files[0]);
            }
            
            document.getElementById('chat').innerHTML += "\\n<b>User:</b> " + text;
            if (fileInput.files.length > 0) document.getElementById('chat').innerHTML += " [Image Attached]";
            document.getElementById('chat').innerHTML += "\\n<b>Kimi:</b> ";
            
            let response = await fetch('/chat', { method: 'POST', body: formData });
            let reader = response.body.getReader();
            let decoder = new TextDecoder();
            
            while (true) {
                const {value, done} = await reader.read();
                if (done) break;
                document.getElementById('chat').innerHTML += decoder.decode(value);
                document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
            }
            document.getElementById('textInput').value = '';
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/chat", methods=["POST"])
def chat():
    text = request.form.get("text", "")
    has_image = "image" in request.files

    if has_image:
        image_file = request.files["image"]
        img = Image.open(BytesIO(image_file.read())).convert("RGB")
        inputs = processor(
            text=text, medias=[{"type": "image", "image": img}], return_tensors="np"
        )
    else:
        inputs = processor(text=text, return_tensors="np")

    input_ids = inputs["input_ids"][0].astype(np.int32)
    num_tokens = len(input_ids)

    # Write metadata into SHM
    shm.seek(0)
    shm.write(struct.pack("<i", 1))  # state = 1 (Ready)
    shm.write(struct.pack("<i", num_tokens))
    shm.write(input_ids.tobytes())
    shm.seek(4 + 4 + 8192 * 4)  # padding

    shm.write(struct.pack("<i", 0))  # output_token offset
    shm.write(struct.pack("<i", 1 if has_image else 0))

    if has_image:
        grid_thw = inputs["grid_thws"][0]
        shm.write(struct.pack("<iii", grid_thw[0], grid_thw[1], grid_thw[2]))
        pixel_values = inputs["pixel_values"]
        num_patches = pixel_values.shape[0]
        shm.write(struct.pack("<i", num_patches))
        shm.write(pixel_values.astype(np.float32).tobytes())
    else:
        shm.write(struct.pack("<iiii", 0, 0, 0, 0))

    def generate():
        while True:
            shm.seek(0)
            state = struct.unpack("<i", shm.read(4))[0]
            if state == 3:
                shm.seek(32776)
                token = struct.unpack("<i", shm.read(4))[0]
                if token == 163586:  # EOS
                    shm.seek(0)
                    shm.write(struct.pack("<i", 5))  # Stop C++ loop for this generation
                    break
                word = processor.decode([token])
                yield word
                shm.seek(0)
                shm.write(struct.pack("<i", 4))  # Acknowledge
            elif state == 5:
                break
            time.sleep(0.01)

    return Response(generate(), mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
