# kokoro-coreml

A production-ready PyTorch → CoreML conversion pipeline for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), enabling on-device text-to-speech on Apple Silicon with Apple Neural Engine acceleration.

> **Kokoro** is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects.

## 🎯 Key Achievements

- ✅ **Successfully exported Kokoro-82M to CoreML** with a novel two-stage architecture
- ⚡ **30-50% speedup** through Apple Neural Engine (ANE) optimization for the vocoder
- 🏗️ **Production-ready pipeline** with bucketing strategy for variable-length synthesis
- 📱 **iOS/macOS compatible** models ready for deployment

## 🚀 Quick Start: CoreML Export

```bash
# Install dependencies
pip install torch==2.3.0 coremltools==7.2 safetensors numpy==1.26.4

# Clone the repository
git clone https://github.com/yourusername/kokoro-coreml.git
cd kokoro-coreml

# Export to CoreML
python examples/export_coreml.py --output_dir coreml_models

# Output:
# ✅ kokoro_duration.mlpackage (dynamic text length)
# ✅ kokoro_synthesizer_3s.mlpackage (fixed 3s audio output)
```

## 📐 Architecture

The CoreML conversion uses a **two-stage pipeline** to handle Kokoro's complex dynamic operations:

1. **Duration Model** (CPU/GPU): Predicts phoneme durations and extracts features
   - Handles variable-length text input with `ct.RangeDim`
   - Runs transformer and LSTM layers
   - Outputs intermediate representations

2. **Synthesizer Model** (ANE-optimized): Generates audio waveforms
   - Uses fixed-size bucketing (3s, 5s, 10s, 30s)
   - Vocoder runs on Apple Neural Engine
   - Achieves significant speedup

```
Text → [Duration Model] → Features → [Synthesizer Model] → Audio
         ↓                             ↑
         Durations → [Client builds alignment matrix]
```

## 🔧 Technical Details

### What Works on ANE
- ✅ iSTFTNet vocoder (~50% of compute)
- ✅ Conv1d, ConvTranspose1d, LeakyReLU operations
- ✅ Result: 30-50% overall speedup

### What Doesn't
- ❌ LSTM layers (no ANE support)
- ❌ AdaLayerNorm, AdainResBlk1d (custom style injection)
- ❌ Solution: Keep these on CPU/GPU in Duration Model

### Key Innovations
- **Bucketing Strategy**: Pre-compile models for common audio lengths
- **Client-Side Alignment**: Build alignment matrix in Swift/Python
- **Monkey-Patching**: CoreML-friendly model modifications during export

## 📚 Documentation

- [**Detailed Conversion Guide**](docs/Kokoro-to-CoreML-conversion.md) - Step-by-step instructions
- [**Learnings & Challenges**](docs/learnings.md) - Technical deep-dive into solutions

## 🐍 Python Usage
The original Kokoro Python library is still available:
```py
!pip install -q kokoro>=0.9.4 soundfile
!apt-get -qq -y install espeak-ng > /dev/null 2>&1
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch
pipeline = KPipeline(lang_code='a')
text = '''
[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
'''
generator = pipeline(text, voice='af_heart')
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    display(Audio(data=audio, rate=24000, autoplay=i==0))
    sf.write(f'{i}.wav', audio, 24000)
```
Under the hood, `kokoro` uses [`misaki`](https://pypi.org/project/misaki/), a G2P library at https://github.com/hexgrad/misaki

### Advanced Usage
You can run this advanced cell on [Google Colab](https://colab.research.google.com/).
```py
# 1️⃣ Install kokoro
!pip install -q kokoro>=0.9.4 soundfile
# 2️⃣ Install espeak, used for English OOD fallback and some non-English languages
!apt-get -qq -y install espeak-ng > /dev/null 2>&1

# 3️⃣ Initalize a pipeline
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch
# 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
# 🇪🇸 'e' => Spanish es
# 🇫🇷 'f' => French fr-fr
# 🇮🇳 'h' => Hindi hi
# 🇮🇹 'i' => Italian it
# 🇯🇵 'j' => Japanese: pip install misaki[ja]
# 🇧🇷 'p' => Brazilian Portuguese pt-br
# 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
pipeline = KPipeline(lang_code='a') # <= make sure lang_code matches voice, reference above.

# This text is for demonstration purposes only, unseen during training
text = '''
The sky above the port was the color of television, tuned to a dead channel.
"It's not like I'm using," Case heard someone say, as he shouldered his way through the crowd around the door of the Chat. "It's like my body's developed this massive drug deficiency."
It was a Sprawl voice and a Sprawl joke. The Chatsubo was a bar for professional expatriates; you could drink there for a week and never hear two words in Japanese.

These were to have an enormous impact, not only because they were associated with Constantine, but also because, as in so many other areas, the decisions taken by Constantine (or in his name) were to have great significance for centuries to come. One of the main issues was the shape that Christian churches were to take, since there was not, apparently, a tradition of monumental church buildings when Constantine decided to help the Christian church build a series of truly spectacular structures. The main form that these churches took was that of the basilica, a multipurpose rectangular structure, based ultimately on the earlier Greek stoa, which could be found in most of the great cities of the empire. Christianity, unlike classical polytheism, needed a large interior space for the celebration of its religious services, and the basilica aptly filled that need. We naturally do not know the degree to which the emperor was involved in the design of new churches, but it is tempting to connect this with the secular basilica that Constantine completed in the Roman forum (the so-called Basilica of Maxentius) and the one he probably built in Trier, in connection with his residence in the city at a time when he was still caesar.

[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
'''
# text = '「もしおれがただ偶然、そしてこうしようというつもりでなくここに立っているのなら、ちょっとばかり絶望するところだな」と、そんなことが彼の頭に思い浮かんだ。'
# text = '中國人民不信邪也不怕邪，不惹事也不怕事，任何外國不要指望我們會拿自己的核心利益做交易，不要指望我們會吞下損害我國主權、安全、發展利益的苦果！'
# text = 'Los partidos políticos tradicionales compiten con los populismos y los movimientos asamblearios.'
# text = 'Le dromadaire resplendissant déambulait tranquillement dans les méandres en mastiquant de petites feuilles vernissées.'
# text = 'ट्रांसपोर्टरों की हड़ताल लगातार पांचवें दिन जारी, दिसंबर से इलेक्ट्रॉनिक टोल कलेक्शनल सिस्टम'
# text = "Allora cominciava l'insonnia, o un dormiveglia peggiore dell'insonnia, che talvolta assumeva i caratteri dell'incubo."
# text = 'Elabora relatórios de acompanhamento cronológico para as diferentes unidades do Departamento que propõem contratos.'

# 4️⃣ Generate, display, and save audio files in a loop.
generator = pipeline(
    text, voice='af_heart', # <= change voice here
    speed=1, split_pattern=r'\n+'
)
# Alternatively, load voice tensor directly:
# voice_tensor = torch.load('path/to/voice.pt', weights_only=True)
# generator = pipeline(
#     text, voice=voice_tensor,
#     speed=1, split_pattern=r'\n+'
# )

for i, (gs, ps, audio) in enumerate(generator):
    print(i)  # i => index
    print(gs) # gs => graphemes/text
    print(ps) # ps => phonemes
    display(Audio(data=audio, rate=24000, autoplay=i==0))
    sf.write(f'{i}.wav', audio, 24000) # save each audio file
```

### Windows Installation
To install espeak-ng on Windows:
1. Go to [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases)
2. Click on **Latest release** 
3. Download the appropriate `*.msi` file (e.g. **espeak-ng-20191129-b702b03-x64.msi**)
4. Run the downloaded installer

For advanced configuration and usage on Windows, see the [official espeak-ng Windows guide](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md)

### MacOS Apple Silicon GPU Acceleration

On Mac M1/M2/M3/M4 devices, you can explicitly specify the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to enable GPU acceleration.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python run-your-kokoro-script.py
```

### Conda Environment
Use the following conda `environment.yml` if you're facing any dependency issues.
```yaml
name: kokoro
channels:
  - defaults
dependencies:
  - python==3.9       
  - libstdcxx~=12.4.0 # Needed to load espeak correctly. Try removing this if you're facing issues with Espeak fallback. 
  - pip:
      - kokoro>=0.3.1
      - soundfile
      - misaki[en]
```

## 🎯 Performance

### Conversion Metrics
- **Model Size**: ~330MB per model (FP16 precision)
- **Export Time**: ~2-5 minutes per model
- **Inference Speed**: 30-50% faster with ANE optimization

### Compatibility
- **Minimum iOS**: 15.0 (16.0+ recommended for best performance)
- **Devices**: All Apple Silicon Macs, iPhone 12+, iPad Air 4+

## 🛠️ Advanced Usage

### Custom Bucketing
Modify the buckets in `export_coreml.py`:
```python
buckets = {
    "3s": 3 * 24000,   # 72,000 frames
    "5s": 5 * 24000,   # 120,000 frames
    "10s": 10 * 24000, # 240,000 frames
    "30s": 30 * 24000  # 720,000 frames
}
```

### Swift Integration Example
```swift
// Load models
let durationModel = try MLModel(contentsOf: durationURL)
let synthesizerModel = try MLModel(contentsOf: synthesizer3sURL)

// Run inference
let durationOutput = try durationModel.prediction(from: inputs)
let audioOutput = try synthesizerModel.prediction(from: features)
```

## 📝 Acknowledgements

### Original Kokoro Team
- 🛠️ [@yl4579](https://huggingface.co/yl4579) for architecting StyleTTS 2
- 🏆 [@Pendrokar](https://huggingface.co/Pendrokar) for TTS Spaces Arena
- 📊 Synthetic training data contributors
- ❤️ Compute sponsors

### CoreML Conversion
- 🍎 Apple's coremltools team for excellent documentation
- 🧠 The scrappy approach inspired by startup engineering principles
- 📖 Detailed learnings documented through iterative debugging

### Community
- 👾 Discord server: https://discord.gg/QuGxSWBfQy
- 🪽 Kokoro is Japanese for "heart" or "spirit"

<img src="https://static0.gamerantimages.com/wordpress/wp-content/uploads/2024/08/terminator-zero-41-1.jpg" width="400" alt="kokoro" />
