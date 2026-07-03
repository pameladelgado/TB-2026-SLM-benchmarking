# LLM Experiment Setup

## Step 1 — Create project folder

Command:
```bash
mkdir llm-experiments
```
Result:
Folder created successfully.

---

## Step 2 — Install build tools

Command:
```bash

sudo zypper install git cmake gcc-c++ make
```

Result:
Packages installed.

---

## Step 3 — Clone llama.cpp

Command:
```bash

git clone https://github.com/ggml-org/llama.cpp.git
```

Result:
Repository downloaded.

---

## Step 4 — Build llama.cpp

Command:
```bash
cd llama.cpp
cmake -B build
cmake --build build -j4
```

Result:
Compilation completed.

## Step 5  — Create models directory


Command:
```bash
cd ..
mkdir models
```

Result:
Directory created to store GGUF model files.
 
## Step 6 — Download TinyLlama model

Command:
```bash
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Result:
Model downloaded successfully.

Size:
~670 MB

Purpose:
Used for first inference test with llama.cpp.

## Step 7 - Run the model

From inside llama.cpp:


```bash
./build/bin/llama-cli -m ../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p "What is the capital of France?"
```
Expted output:

The capital of France is Paris.

Observed output:
The model launched successfully and generated a response.

Result:
llama.cpp successfully loaded the GGUF model and executed text generation on CPU.

Note:
The response quality was imperfect, but the setup worked correctly.

## Step 8 - Exit

```bash
/exit
```
Result:
Exit successful


## Step 9 — Create Python virtual environment

Command:
```bash
cd ~/llm-experiments
python3 -m venv .venv
source .venv/bin/activate
````

Result:
A Python virtual environment was created to isolate project dependencies.

Purpose:
Avoid conflicts with system Python packages.

---

## Step 10 — Install Python dependencies

Command:

```bash
pip install requests pyyaml psutil pandas
````

Result:

Required Python libraries installed successfully inside the virtual environ


Command:

```bash
pip install -U huggingface_hub
````


We use a GGUF-quantized (Q8_0) version of the LLaMA-3.2-1B-Instruct model.
The quantized model was obtained from a Hugging Face repository and is derived from the original Meta model using llama.cpp-compatible conversion.

Command:

```bash
hf download lmstudio-community/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q8_0.gguf --local-dir .
````


We use a GGUF-quantized (Q4_K_M) version of the Phi-4-mini-instruct model. The quantized model was obtained from a community repository on Hugging Face (lmstudio-community) and is derived from the original Microsoft release using llama.cpp-compatible conversion.
Command:

```bash
hf download lmstudio-community/Phi-4-mini-instruct-GGUF Phi-4-mini-instruct-Q4_K_M.gguf --local-dir .
````





.
├── configs
│   ├── llama_cpp.yaml
│   └── vllm.yaml
├── docs
│   ├── experiment-log.md
│   └── methodology.md
├── engines
│   └── llama.cpp
├── README.md
├── requirements.txt
├── models
│   ├── huggingface-cli
│   ├── Llama-3.1-8B-Instruct-UD-IQ2_XXS.gguf
│   ├── Llama-3.2-1B-Instruct-Q8_0.gguf
│   ├── Phi-4-mini-instruct-Q4_K_M.gguf
│   └── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
├── runner
│   ├── basic_bench.py
│   ├── benchmark.py
│   ├── dataset_loader.py
│   ├── __pycache__
│   └── system_metrics.py
└── setup.md



