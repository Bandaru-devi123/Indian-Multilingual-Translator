import gradio as gr
import assemblyai as aai
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import uuid
from deep_translator import GoogleTranslator as DTGoogleTranslator

# Setup API keys
ASSEMBLYAI_API_KEY = "08462ccdabeb4b9293eeef620c728e4d"
ELEVENLABS_API_KEY = "sk_f7935dc89d772f37376990511ab6682933f8ceeeb3189fdd"
VOICE_ID = "m5qndnI7u4OAdXhH0Mr5"

# Supported languages for translation
LANGUAGES = {
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali"
}

SUPPORTED_TRANSCRIPTION_LANGUAGES = ["en", "hi"]

# Step 1: Transcribe audio
def audio_transcription(audio_file, input_lang="en"):
    try:
        aai.settings.api_key = ASSEMBLYAI_API_KEY
        transcriber = aai.Transcriber()
        lang_code = input_lang if input_lang in SUPPORTED_TRANSCRIPTION_LANGUAGES else "en"
        config = aai.TranscriptionConfig(language_code=lang_code)
        transcription = transcriber.transcribe(audio_file, config=config)
        if transcription.status == aai.TranscriptStatus.error:
            raise Exception(f"Transcription failed: {transcription.error}")
        return transcription.text
    except Exception as e:
        return f"Error in transcription: {str(e)}"

# Step 2: Read text file
def read_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Step 3: Translate text using deep-translator
def translate_text(text, input_lang="en"):
    translations = {}
    for code, name in LANGUAGES.items():
        try:
            translated = DTGoogleTranslator(source=input_lang, target=code).translate(text)
            translations[name] = translated
        except Exception as e:
            translations[name] = f"Translation error: {str(e)}"
    return translations

# Step 4: Convert to speech
def text_to_speech(text, lang_name):
    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        response = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=text[:1000],  # ElevenLabs limit per request
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.8,
                style=0.5,
                use_speaker_boost=True,
            ),
        )
        file_path = f"{uuid.uuid4()}.mp3"
        with open(file_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)
        return file_path
    except Exception as e:
        return f"Error generating audio for {lang_name}: {str(e)}"

# Step 5: Main handler
def process_input(audio_input, text_file, manual_text, input_language):
    if not audio_input and not text_file and not manual_text.strip():
        return "Please provide audio, text file, or type some text.", *[None]*6

    lang_map = {name: code for code, name in LANGUAGES.items()}
    lang_map["English"] = "en"
    input_lang_code = lang_map.get(input_language, "en")

    # Get original text
    try:
        if manual_text.strip():
            original_text = manual_text.strip()
        elif audio_input:
            original_text = audio_transcription(audio_input, input_lang_code)
        else:
            original_text = read_text_file(text_file)
        
        if "Error" in original_text:
            return original_text, *[None]*6
    except Exception as e:
        return f"Error processing input: {str(e)}", *[None]*6

    # Translate and generate audio
    translations = translate_text(original_text, input_lang_code)

    audio_outputs = []
    text_outputs = []
    for lang, translated_text in translations.items():
        if "error" not in translated_text.lower():
            audio_path = text_to_speech(translated_text, lang)
            audio_outputs.append((lang, audio_path))
            text_outputs.append(f"{lang}: {translated_text}")
        else:
            audio_outputs.append((lang, None))
            text_outputs.append(f"{lang}: {translated_text}")

    audio_dict = {lang: path for lang, path in audio_outputs}
    text_block = "\n\n".join(text_outputs)

    return (
        text_block,
        audio_dict.get("Hindi"),
        audio_dict.get("Telugu"),
        audio_dict.get("Tamil"),
        audio_dict.get("Kannada"),
        audio_dict.get("Malayalam"),
        audio_dict.get("Bengali")
    )

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🌐 Indian Language Translator: Audio/Text to Voice")
    gr.Markdown(
        "Upload audio (Hindi/English), upload a .txt file, or type your text. "
        "Translates into 6 Indian languages and generates voice output."
    )

    with gr.Row():
        input_language = gr.Dropdown(
            choices=["English", "Hindi"],
            label="Select Input Language",
            value="English"
        )

    with gr.Row():
        audio_input = gr.Audio(label="🎤 Audio Input", type="filepath")
        text_file = gr.File(label="📄 Text File Input (.txt)", file_types=[".txt"])

    manual_text = gr.Textbox(
        label="🖊 Or Type Text Directly",
        placeholder="Type or paste text here...",
        lines=5
    )

    translate_button = gr.Button("🌍 Translate and Convert to Speech")

    output_text = gr.Textbox(label="📝 Translated Texts", lines=15)

    with gr.Row():
        hindi = gr.Audio(label="Hindi")
        telugu = gr.Audio(label="Telugu")
        tamil = gr.Audio(label="Tamil")

    with gr.Row():
        kannada = gr.Audio(label="Kannada")
        malayalam = gr.Audio(label="Malayalam")
        bengali = gr.Audio(label="Bengali")

    translate_button.click(
        fn=process_input,
        inputs=[audio_input, text_file, manual_text, input_language],
        outputs=[output_text, hindi, telugu, tamil, kannada, malayalam, bengali]
    )

if __name__ == "_main_":
    demo.launch()