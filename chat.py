import mmap
import struct
import sys
import time

from flask import Flask, Response, render_template_string, request
from transformers import AutoTokenizer

from encoding_dsv4 import encode_messages

app = Flask(__name__)

# Constants matching C++
SHM_NAME = "/tg_chat_shm" if sys.platform != "win32" else "tg_chat_shm"
SHM_SIZE = 4 + 8192 * 4 + 4 + 4

try:
    if sys.platform == "win32":
        shm = mmap.mmap(-1, SHM_SIZE, tagname=SHM_NAME)
    else:
        import posix_ipc

        memory = posix_ipc.SharedMemory(SHM_NAME, posix_ipc.O_RDWR)
        shm = mmap.mmap(memory.fd, memory.size)
except Exception as e:
    print("Failed to open shared memory. Ensure chat.cpp is running. Error:", e)
    sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V4-Flash-0731")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>DeepSeek V4 Flash - Tensor Graphs</title>
    <style>
        body { font-family: sans-serif; margin: 40px; background: #f4f4f9; }
        #chat { max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .message { margin-bottom: 15px; }
        .user { font-weight: bold; color: #2c3e50; }
        .assistant { color: #34495e; white-space: pre-wrap; }
        #input-box { width: 100%; height: 100px; padding: 10px; box-sizing: border-box; }
        button { margin-top: 10px; padding: 10px 20px; cursor: pointer; }
    </style>
</head>
<body>
    <div id="chat">
        <h2>DeepSeek V4 Flash Chat</h2>
        <div id="history"></div>
        <textarea id="input-box" placeholder="Ask something..."></textarea>
        <button onclick="send()">Send</button>
    </div>
    <script>
        let messages = [];
        
        function send() {
            let input = document.getElementById('input-box').value;
            if (!input) return;
            messages.push({role: 'user', content: input});
            document.getElementById('input-box').value = '';
            render();
            
            let assistantDiv = document.createElement('div');
            assistantDiv.className = 'message assistant';
            assistantDiv.innerHTML = '<i>Thinking...</i>';
            document.getElementById('history').appendChild(assistantDiv);
            
            fetch('/generate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({messages: messages})
            }).then(response => {
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullText = "";
                
                function read() {
                    reader.read().then(({done, value}) => {
                        if (done) {
                            messages.push({role: 'assistant', content: fullText});
                            return;
                        }
                        const chunk = decoder.decode(value);
                        fullText += chunk;
                        assistantDiv.innerText = fullText;
                        read();
                    });
                }
                read();
            });
        }
        
        function render() {
            document.getElementById('history').innerHTML = messages.map(m => 
                `<div class="message ${m.role}"><b>${m.role}:</b> ${m.content}</div>`
            ).join('');
        }
    </script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = encode_messages(data["messages"], thinking_mode="chat")
    tokens = tokenizer.encode(prompt)

    def token_generator():
        # Wait for Idle
        while struct.unpack("i", shm[:4])[0] not in (0, 4):
            time.sleep(0.01)

        # Write length and tokens
        struct.pack_into("i", shm, 4 + 8192 * 4, len(tokens))
        for i, t in enumerate(tokens):
            struct.pack_into("i", shm, 4 + i * 4, t)

        # Set State = 1 (Generate)
        struct.pack_into("i", shm, 0, 1)

        while True:
            state = struct.unpack("i", shm[:4])[0]
            if state == 2:
                # Token is ready
                out_tok = struct.unpack("i", shm[4 + 8192 * 4 + 4 : 4 + 8192 * 4 + 8])[
                    0
                ]
                if out_tok == tokenizer.eos_token_id:
                    struct.pack_into("i", shm, 0, 4)
                    break
                text = tokenizer.decode([out_tok])
                yield text
                struct.pack_into("i", shm, 0, 3)  # Acknowledge
            elif state == 4:
                break
            time.sleep(0.001)

    return Response(token_generator(), mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=False)
