"""
tts_service.py
Cockatiel TTS Module Entrypoint & Service Integration
"""

import os
import sys
from pathlib import Path

# 1. Automatically locate the Cockatiel root and register the lib-cockatiel folder
current_dir = Path(__file__).resolve().parent
cockatiel_root = current_dir.parent.parent  # Resolves to /Users/insert/Cockatiel/

# The python client lives in the engine's shared lib directory.
lib_path = cockatiel_root / "cockatiel_engine-rs" / "cockatiel_lib" / "python"
if str(lib_path) not in sys.path:
    sys.path.insert(0, str(lib_path))

# 2. Dynamically hunt down the .proto file ANYWHERE inside the Cockatiel root
proto_file = None
for file_path in cockatiel_root.rglob("cockatiel_protobuf.proto"):
    proto_file = file_path
    break  # Grab the first match we find

if proto_file and proto_file.exists():
    os.environ["COCKATIEL_PROTO_PATH"] = str(proto_file)
else:
    raise FileNotFoundError(
        f"CRITICAL: Could not find 'cockatiel_protobuf.proto' anywhere in {cockatiel_root}!"
    )

# Now import safely
from lib_cockatiel import CockatielClient, pb

import argparse
import asyncio
import json
import logging
import math

from worker_manager import WorkerManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("tts_service")

CONFIG_PATH = Path("config.json")
CLIPS_DIR = Path("clips")


def get_safe_filename(text: str, max_length: int = 50) -> str:
    """Replaces spaces with underscores, removes invalid chars, and trims length."""
    sanitized = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in text.replace(" ", "_"))
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")[:max_length]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Cockatiel TTS Module")
    parser.add_argument("--ip", type=str, help="Engine IP address")
    parser.add_argument("-p", "--port", type=int, help="Engine WebSocket port")
    parser.add_argument("--model", type=str, help="Default TTS worker model name")
    parser.add_argument("--test", type=str, help="Test TTS synthesis locally with a given message without connecting to Cockatiel")
    parser.add_argument("-n", "--new", action="store_true", help="Reset configuration and regenerate defaults on next start")
    parser.add_argument("--pin", type=str, help="Engine pairing PIN (accepted for CLI compatibility; config file takes precedence)")
    parser.add_argument("--name", type=str, help="Module name override")
    return parser.parse_args()


def load_env_file(path: str):
    """Load a KEY=VALUE `.env` file into the environment (real env wins)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"')
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


def load_or_setup_config(args) -> dict:
    if args.new and CONFIG_PATH.exists():
        logger.info("--new flag detected. Removing old config.json...")
        try:
            CONFIG_PATH.unlink()
        except OSError as e:
            logger.error("Failed to delete config.json: %s", e)

    config = {}
    if CONFIG_PATH.exists():
        try:
            config = json.loads(CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            config = {}

    if not config:
        logger.warning(
            "No config.json found. Writing default config "
            "(engine 127.0.0.1:9734). Address is overridden by --ip/--port/--pin."
        )
        config = {
            "engine_ip": "127.0.0.1",
            "engine_port": 9734,
            "model": "mms",
        }
        CONFIG_PATH.write_text(json.dumps(config, indent=2))

    # Settings only — the pairing PIN is a secret and lives in the
    # environment (COCKATIEL_PIN / --pin), never in config.json.
    if args.ip:
        config["engine_ip"] = args.ip
    if args.port:
        config["engine_port"] = args.port
    if args.model:
        config["model"] = args.model

    config.setdefault("volume", 0.4)
    config.setdefault("play_locally", False)

    return config


def play_audio(path: str, volume: float = 1.0):
    """Play a rendered audio file at the given volume (0.0–1.0) using pydub + simpleaudio."""
    try:
        from pydub import AudioSegment
        import simpleaudio as sa

        segment = AudioSegment.from_file(path)
        if volume < 0.0:
            volume = 0.0
        if volume > 1.0:
            volume = 1.0
        # Apply volume as a dB change relative to full scale.
        db = 20.0 * math.log10(volume) if volume > 0.0 else -120.0
        segment = segment.apply_gain(db)
        play_obj = sa.play_buffer(
            segment.raw_data,
            num_channels=segment.channels,
            bytes_per_sample=segment.sample_width,
            sample_rate=segment.frame_rate,
        )
        play_obj.wait_done()
    except ImportError as e:
        logger.warning("Playback disabled (missing pydub/simpleaudio): %s", e)
    except Exception as e:
        logger.error("Playback failed: %s", e)


async def main():
    args = parse_arguments()
    config = load_or_setup_config(args)

    CLIPS_DIR.mkdir(exist_ok=True)

    manager = WorkerManager(workers_dir="workers")
    available = manager.available_workers()
    logger.info("Discovered local TTS workers: %s", available)

    if not available:
        logger.error("No workers found in 'workers/' folder!")
        return

    active_model = config.get("model", "mms")
    if active_model not in available:
        logger.warning("Configured model '%s' not found. Falling back to '%s'", active_model, available[0])
        active_model = available[0]

    # Handle local testing mode via --test flag
    if args.test:
        logger.info("Running in TEST mode using model '%s'", active_model)
        logger.info("Test message: '%s'", args.test)
        
        safe_name = get_safe_filename(args.test)
        output_path = CLIPS_DIR / f"{safe_name}.mp3"

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                manager.synthesize,
                active_model,
                args.test,
                str(output_path)
            )
            logger.info("Test synthesis success! Audio saved to: %s", output_path.resolve())
        except Exception as e:
            logger.error("Test synthesis failed: %s", e)
        return

    # Normal Cockatiel Engine connection loop
    logger.info("TTS Service active using model worker: '%s'", active_model)

    # Modern config pattern: settings in config.json, secrets in .env / env.
    load_env_file(".env")
    engine_ip = config.get("engine_ip", "127.0.0.1")
    engine_port = int(config.get("engine_port", 9734))
    pairing_pin = int(os.environ.get("COCKATIEL_PIN") or args.pin or 0)
    module_name = args.name or "tts-service"

    client = await (
        CockatielClient.connect(module_name)
        .endpoint(engine_ip, engine_port)
        .pin(pairing_pin)
        .position("postprocess")
        .connect()
    )
    logger.info("TTS Service connected as '%s' (postprocess).", module_name)

    # Serialize synthesis: the local model isn't safe for concurrent inference,
    # and handlers now run as background tasks so probes are never blocked.
    synthesis_lock = asyncio.Semaphore(1)

    @client.on("message_post_process")
    async def handle_post_process(msg, container):
        text_to_speak = msg.processed_message or ""
        if not text_to_speak.strip():
            if msg.raw_message is not None and msg.raw_message.raw_message:
                text_to_speak = msg.raw_message.raw_message
        if not text_to_speak.strip():
            return

        logger.info("[TTS Engine] Rendering speech for message: '%s'", text_to_speak)
        safe_name = get_safe_filename(text_to_speak)
        output_path = CLIPS_DIR / f"{safe_name}_{msg.message_uuid7[:8]}.mp3"

        try:
            loop = asyncio.get_running_loop()
            async with synthesis_lock:
                await loop.run_in_executor(
                    None,
                    manager.synthesize,
                    active_model,
                    text_to_speak,
                    str(output_path),
                )
            audio_bytes = output_path.read_bytes()
            logger.info(
                "[TTS Engine] Success! Rendered %d bytes for %s",
                len(audio_bytes),
                msg.message_uuid7,
            )

            # Return the audio on the same message so it's persisted to the
            # timeline (displays retrieve + play it). This also acks the stage.
            reply = pb.MessagePostProcess(
                message_uuid7=msg.message_uuid7,
                processed_message=text_to_speak,
            )
            reply.audio = audio_bytes
            reply.audio_type = "audio/mpeg"
            await client.send("message_post_process", reply)

            # Optional local playback for standalone/no-display setups.
            if config.get("play_locally", False):
                volume = float(config.get("volume", 0.4))
                await loop.run_in_executor(None, play_audio, str(output_path), volume)
        except Exception as e:
            logger.error("[TTS Engine] Failed to synthesize speech: %s", e)

    logger.info("Listening for incoming Cockatiel engine stream payloads...")
    await client.listen()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("TTS Service stopped by user.")
