import os
from pathlib import Path
from gtts import gTTS
from pydub import AudioSegment

# Create output folder inside local_testing (relative to script location)
script_dir = Path(__file__).parent
output_dir = script_dir / "output_audios"
output_dir.mkdir(exist_ok=True)

print("Generating audio files... This may take a moment.")

def generate_tts_segment(text, lang):
    """Generates a temporary mp3 for a specific text segment."""
    tts = gTTS(text=text, lang=lang, slow=False)
    filename = "temp_segment.mp3"
    tts.save(filename)
    # Load as AudioSegment
    audio = AudioSegment.from_mp3(filename)
    os.remove(filename) # clean up
    return audio

def change_pitch(audio, semitones):
    """Changes pitch to simulate a different speaker."""
    new_sample_rate = int(audio.frame_rate * (2.0 ** (semitones / 12.0)))
    shifted = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
    return shifted.set_frame_rate(44100)

# --- SCENARIO 1 & 2: MONO SINGLE SPEAKER (Food Review) ---

scripts_single = [
    {
        "filename": "1_mono_hindi_food.mp3",
        "lang": "hi",
        "text": "नमस्ते दोस्तों! आज हम चांदनी चौक में हैं और मेरे सामने है यह गरमा-गरम आलू टिक्की। इसकी खुशबू लाजवाब है। ऊपर से डाली गई ये मीठी चटनी और दही का कॉम्बिनेशन इसे बहुत ही बेहतरीन बना रहा है। बाहर से क्रिस्पी और अंदर से सॉफ्ट, यह सच में दिल्ली का बेस्ट स्ट्रीट फूड है।"
    },
    {
        "filename": "2_mono_bengali_food.mp3",
        "lang": "bn",
        "text": "নমস্কার! আজ আমি ট্রাই করছি কলকাতার ফিশ ফ্রাই। বাইরের কোটিংটা বেশ মুচমুচে, আর ভেতরের মাছটা একদম তাজা। এর সাথে এই কাসুন্দিটা জাস্ট জমে গেছে। আপনারা যারা সি-ফুড ভালোবাসেন, তাদের জন্য এটা মাস্ট ট্রাই। দাম হিসেবে কোয়ান্টিটিও বেশ ভালো।"
    }
]

for item in scripts_single:
    print(f"Processing: {item['filename']}")
    audio = generate_tts_segment(item['text'], item['lang'])
    audio = audio.set_channels(1) # Ensure Mono
    audio.export(str(output_dir / item['filename']), format="mp3")


# --- SCENARIO 3 & 4: MONO TWO SPEAKERS (Fashion) ---
# We combine segments sequentially into one mono track.

scripts_dialogue_mono = [
    {
        "filename": "3_mono_hindi_fashion.mp3",
        "lang": "hi",
        "dialogue": [
            ("A", "अरे रिया, तुमने देखा आजकल पेस्टल कलर्स का कितना ट्रेंड चल रहा है शादियों में?"),
            ("B", "हाँ बिलकुल! मुझे लगता है डार्क कलर्स अब आउट ऑफ फैशन हो रहे हैं। सब को अब लाइट पिंक या आइवरी चाहिए।"),
            ("A", "सही कहा, और ज्वैलरी भी अब लोग हैवी नहीं, बल्कि मिनिमल ही पसंद कर रहे हैं।")
        ]
    },
    {
        "filename": "4_mono_bengali_fashion.mp3",
        "lang": "bn",
        "dialogue": [
            ("A", "এই পুজোয় তুই কি শাড়ি কিনলি? সিল্ক নাকি তাঁত?"),
            ("B", "আমি এবার একটা জামদানি নিয়েছি। গরমে ওটাই সবথেকে আরামদায়ক। তুই কি কিনলি?"),
            ("A", "আমি ভাবছি একটা হ্যান্ডলুম কুর্তা কিনবো। পুজোর সকালে ওটা পরতে বেশ ভালো লাগে।")
        ]
    }
]

for item in scripts_dialogue_mono:
    print(f"Processing: {item['filename']}")
    full_audio = AudioSegment.empty()
    
    for speaker, text in item['dialogue']:
        segment = generate_tts_segment(text, item['lang'])
        
        # If Speaker B, we pitch shift down slightly to sound like a different person
        if speaker == "B":
            segment = change_pitch(segment, -2.5)
            
        full_audio += segment
        full_audio += AudioSegment.silent(duration=500) # 0.5s pause between speakers
        
    full_audio = full_audio.set_channels(1) # Ensure Mono
    full_audio.export(str(output_dir / item['filename']), format="mp3")


# --- SCENARIO 5 & 6: STEREO TWO SPEAKERS (Gardening) ---
# Speaker A on LEFT channel, Speaker B on RIGHT channel.

scripts_dialogue_stereo = [
    {
        "filename": "5_stereo_hindi_gardening.mp3",
        "lang": "hi",
        "dialogue": [
            ("A", "माली काका, यह तुलसी का पौधा सूख क्यों रहा है? मैंने पानी तो रोज दिया था।"),
            ("B", "बेटा, आपने शायद ज्यादा पानी दे दिया। सर्दियों में इसे कम पानी की जरूरत होती है। मिट्टी चेक करके ही पानी डालना चाहिए।"),
            ("A", "अच्छा, तो क्या अब मुझे इसमें खाद डालनी चाहिए?")
        ]
    },
    {
        "filename": "6_stereo_bengali_gardening.mp3",
        "lang": "bn",
        "dialogue": [
            ("A", "আচ্ছা, গোলাপ গাছে পোকা ধরলে কি স্প্রে করা উচিত?"),
            ("B", "তুমি নিম তেল আর জলের মিশ্রণ স্প্রে করতে পারো। ওটা গাছের জন্য খুব ভালো আর প্রাকৃতিক।"),
            ("A", "বাহ! এটা তো জানতাম না। এটা কি সব গাছে ব্যবহার করা যাবে?")
        ]
    }
]

for item in scripts_dialogue_stereo:
    print(f"Processing: {item['filename']}")
    
    left_channel = AudioSegment.empty()
    right_channel = AudioSegment.empty()
    
    for speaker, text in item['dialogue']:
        segment = generate_tts_segment(text, item['lang'])
        
        if speaker == "A":
            # Shift pitch for Speaker A (optional, or keep original)
            left_segment = segment
            right_segment = AudioSegment.silent(duration=len(left_segment))
        else:
            # Shift pitch for Speaker B
            right_segment = change_pitch(segment, -2.5)
            left_segment = AudioSegment.silent(duration=len(right_segment))
        
        # Ensure both segments are the same length
        max_length = max(len(left_segment), len(right_segment))
        left_segment = left_segment[:max_length] if len(left_segment) < max_length else left_segment
        right_segment = right_segment[:max_length] if len(right_segment) < max_length else right_segment
        
        left_channel += left_segment
        right_channel += right_segment
            
        # Add small pause between turns
        pause = AudioSegment.silent(duration=300)
        left_channel += pause
        right_channel += pause

    # Ensure both channels are the same length before combining
    max_channel_length = max(len(left_channel), len(right_channel))
    # Use set_frame_rate to ensure same frame rate, then pad to exact length
    left_channel = left_channel.set_frame_rate(44100)
    right_channel = right_channel.set_frame_rate(44100)
    
    if len(left_channel) < max_channel_length:
        padding = AudioSegment.silent(duration=max_channel_length - len(left_channel))
        left_channel = left_channel + padding
    if len(right_channel) < max_channel_length:
        padding = AudioSegment.silent(duration=max_channel_length - len(right_channel))
        right_channel = right_channel + padding
    
    # Trim to exact same length (in case of slight differences)
    min_length = min(len(left_channel), len(right_channel))
    left_channel = left_channel[:min_length]
    right_channel = right_channel[:min_length]
    
    # Combine into stereo
    stereo_sound = AudioSegment.from_mono_audiosegments(left_channel, right_channel)
    stereo_sound.export(str(output_dir / item['filename']), format="mp3")

print("\nSuccess! All 6 files are in the 'output_audios' folder.")