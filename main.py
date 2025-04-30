import asyncio
import json
import datetime
import logging
import os
import time
import traceback
from typing import AsyncGenerator, Optional, Callable, Awaitable, Annotated
from dotenv import load_dotenv
import aiohttp
from functools import partial

from openai import AsyncOpenAI as DirectAsyncOpenAI

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
    llm,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import openai, silero, groq
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.rtc import Room, DataPacketKind, LocalParticipant, RemoteParticipant, DataPacket, ConnectionState
from livekit.agents.voice import SpeechHandle, SpeechCreatedEvent 

from mem0 import MemoryClient
from num2words import num2words
import re


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('livekit').setLevel(logging.WARNING)
logging.getLogger('websockets').setLevel(logging.WARNING)
logging.getLogger('mem0').setLevel(logging.INFO)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('assistant-api').setLevel(logging.INFO)
logging.getLogger('aiohttp').setLevel(logging.WARNING)

load_dotenv()
logger.info("Environment variables loaded.")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MEM0_API_KEY = os.getenv("MEM0_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
    logger.error("FATAL: LIVEKIT_URL, LIVEKIT_API_KEY, dan LIVEKIT_API_SECRET harus diatur.")
    exit(1)
if not OPENAI_API_KEY:
    logger.error("FATAL: OPENAI_API_KEY not found.")
    exit(1)
if not PERPLEXITY_API_KEY:
    logger.warning("PERPLEXITY_API_KEY not found, internet search will fail.")
if not MEM0_API_KEY:
    logger.warning("MEM0_API_KEY not found, memory functions will be disabled.")

SEMANTIC_QUERY_GENERAL_STARTUP = (
    "Key points, facts, preferences, user's name, user's recent mood, "
    "and user's recent concerns shared in previous conversations"
)
SEMANTIC_QUERY_NAME_RECALL = "What is the user's name?"
MEM0_SEARCH_TIMEOUT = 15.0
MEM0_API_TIMEOUT = 10.0
DEVICE_ACTION_TIMEOUT = 15.0
INTERNET_SEARCH_TIMEOUT = 25.0
MEMORY_TOPICS = ["personal info", "preferences", "concerns", "goals", "life events", "relationships", "user name", "user age", "past advice", "feedback", "meeting schedule", "important dates"]

async def search_mem0_with_timeout(client: Optional[MemoryClient], user_id: str, query: str, limit: int = 5, timeout: float = MEM0_SEARCH_TIMEOUT):
    if not client:
        logger.warning(f"Mem0 search skipped for user '{user_id}': client not available.")
        return None
    if not user_id:
        logger.error("Mem0 search skipped: user_id is missing.")
        return None
    try:
        start_time = time.time()
        logger.debug(f"Starting Mem0 search for user '{user_id}' (query: '{query[:50]}...', limit: {limit})")
        search_coro = asyncio.to_thread(client.search, user_id=user_id, query=query, limit=limit)
        result = await asyncio.wait_for(search_coro, timeout=timeout)
        logger.debug(f"Finished Mem0 search in {time.time() - start_time:.2f}s")
        return result
    except asyncio.TimeoutError:
        logger.warning(f"Mem0 search timed out after {timeout}s for user '{user_id}'")
        return None
    except Exception as e:
        logger.error(f"Error during Mem0 search for user '{user_id}': {e}", exc_info=True)
        return None

async def generate_summary_with_llm(openai_client: DirectAsyncOpenAI, transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "Error: Transkrip kosong."
    if not openai_client:
        return "Error: Klien OpenAI tidak tersedia untuk ringkasan."

    MODEL_TO_USE = "gpt-4o"
    summary_prompt = (
        "Anda adalah asisten AI yang bertugas merangkum transkrip berikut dalam Bahasa Indonesia. "
        "Sebutkan topik utama yang dibahas. Jika ada keputusan atau item tindakan yang jelas, sebutkan juga. "
        "Jika tidak ada, cukup rangkum poin utamanya saja secara singkat.\n\n"
        f"Transkrip:\n{transcript}\n\n---\nRingkasan:"
    )
    try:
        logger.info(f"Requesting summary via DIRECT OpenAI API call (Model: {MODEL_TO_USE})...")
        response = await openai_client.chat.completions.create(
            model=MODEL_TO_USE,
            messages=[{"role": "user", "content": summary_prompt}],
            stream=False
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            summary = response.choices[0].message.content.strip()
            logger.info(f"Direct API summary generated. Length: {len(summary)}")
            return summary or "Model AI tidak dapat menghasilkan ringkasan (via direct call)."
        else:
            logger.error(f"Unexpected response structure from direct OpenAI call: {response}")
            return "Error: Struktur respons tidak terduga dari API OpenAI."
    except Exception as e:
        logger.error(f"Error during DIRECT OpenAI API summarization: {e}", exc_info=True)
        return f"Error saat membuat ringkasan (direct call): {type(e).__name__}"
    
def preprocess_text_for_tts(text: str) -> str:
    # Fungsi ini mencari angka dan mengubahnya menjadi kata (Bahasa Indonesia)
    def number_to_words_id(match):
        try:
            number = int(match.group(0))
            # Gunakan lang='id' untuk Bahasa Indonesia
            return num2words(number, lang='id')
        except ValueError:
            return match.group(0) # Kembalikan seperti semula jika bukan integer valid

    # Gunakan regex untuk menemukan urutan angka
    processed_text = re.sub(r'\d+', number_to_words_id, text)
    return processed_text

class AntyAgent(Agent):
    def __init__(
        self,
        mem0_client: Optional[MemoryClient],
        user_id: Optional[str],
        direct_openai_client: DirectAsyncOpenAI,
        local_participant: LocalParticipant,
    ) -> None:
        try:
            import locale
            # Coba set locale ke Indonesian. Jika gagal, format default akan digunakan.
            # Locale yang mungkin: 'id_ID', 'id_ID.UTF-8', 'Indonesian_indonesia.1252'
            # Periksa locale yang tersedia di sistem Anda dengan `locale -a`
            available_locales = locale.locale_alias.keys()
            target_locale = None
            for loc in ['id_ID.utf8', 'id_ID.UTF-8', 'id_ID', 'Indonesian_Indonesia']:
                 if loc.lower() in available_locales:
                     target_locale = loc
                     break
            if target_locale:
                locale.setlocale(locale.LC_TIME, target_locale)
                logger.info(f"Locale set to {target_locale} for date formatting.")
            else:
                 logger.warning("Indonesian locale not found, using system default for date.")
        except ImportError:
            logger.warning("Locale module not available, using default date format.")
        except Exception as e:
            logger.warning(f"Failed to set locale to Indonesian: {e}")

        # Format tanggal dalam Bahasa Indonesia (Nama hari, tanggal bulan tahun)
        today_date_str = datetime.date.today().strftime('%A, %d %B %Y')

        # 2. Masukkan tanggal ke dalam instruksi menggunakan f-string
        agent_instructions = (
            f"Anda adalah 'Anty', asisten suara yang ramah dan empatik dalam Bahasa Indonesia. "
            f"Kepribadian Anda suportif, membantu, dan sedikit informal namun selalu sopan. "
            f"Hari ini adalah {today_date_str}. Gunakan informasi tanggal ini untuk membantu memahami permintaan terkait waktu, seperti 'besok' atau 'Selasa depan'. " # <-- Tanggal ditambahkan di sini
            f"Jaga agar respons tetap ringkas dan percakapan dalam Bahasa Indonesia. "
            f"Gunakan fungsi yang tersedia jika diperlukan untuk mengingat informasi, mencari di internet, atau mengatur alarm (terutama untuk tanggal relatif)."
            "Pedoman:\n"
            "- Gunakan respons singkat dan ringkas, hindari penggunaan tanda baca yang sulit diucapkan.\n"
            "- Jaga agar respons tetap ringkas dan percakapan dalam Bahasa Indonesia.\n"
            "- Bersikaplah empatik dan suportif secara alami.\n"
            "- Gunakan fungsi memori untuk mempersonalisasi percakapan.\n"
            "- Jangan berikan karakter karakter yang susah untuk disebutkan dalam TTS.\n"
            "- Jangan berikan respons yang berupa angka, tapi buat angka itu menjadi kata kata yang bisa dibaca, misal jika response nya adalah 2025 maka ubah menjadi dua ribu dua puluh lima.\n"
            "- Respons yang anda berikan nanti akan diproses oleh sistem TTS jadi berikan respons yang dapat dibacakan dan dipahami konteks nya ketika dibacakan\n"
            
        )

        super().__init__(instructions=agent_instructions)
        self._mem0_client = mem0_client
        self._user_id = user_id
        self._direct_openai_client = direct_openai_client
        self._local_participant_ref = local_participant
        self._user_name: Optional[str] = None

        logger.info(f"AntyAgent initialized for user: {self._user_id}. Mem0 enabled: {self._mem0_client is not None}")

    async def _on_data_received(self, data: bytes, participant_identity: Optional[str]):
        # --- TAMBAHKAN LOG INI ---
        logger.info(f"Entering _on_data_received from {participant_identity}. Data length: {len(data)}")
        # --- AKHIR TAMBAHAN LOG ---
        try:
            payload_str = data.decode('utf-8')
            logger.info(f"Agent received data from {participant_identity}: {payload_str[:150]}...") # Log ini sudah ada
            payload = json.loads(payload_str)
            msg_type = payload.get("type")

            if msg_type == "summarize_meeting":
                transcript = payload.get("transcript")
                if transcript:
                    logger.info("Summarization request received. Generating summary...")
                    summary_text = await generate_summary_with_llm(self._direct_openai_client, transcript)
                    logger.info(f"Summary generated: '{summary_text[:100]}...'")
                    response_payload_dict = { # Buat dictionary dulu
                        "type": "meeting_summary_result",
                        "summary": summary_text,
                        "original_transcript": transcript
                    }
                    response_payload_str = json.dumps(response_payload_dict)
                    response_payload_bytes = response_payload_str.encode('utf-8')

                    # --- PERBAIKI PENGIRIMAN DATA ---
                    logger.info(f"Attempting to send summary result via local_participant_ref: {response_payload_str[:100]}...")
                    if not hasattr(self, '_local_participant_ref') or not self._local_participant_ref:
                        logger.error("Cannot send summary result: Explicit local_participant reference not found.")
                    else:
                        try:
                            await asyncio.wait_for(
                                self._local_participant_ref.publish_data(payload=response_payload_bytes), # Gunakan referensi eksplisit
                                timeout=DEVICE_ACTION_TIMEOUT # Atau timeout lain yang sesuai
                            )
                            logger.info("Successfully sent summary result via data channel.")

                            # Pindahkan 'say' ke sini agar hanya diucapkan jika pengiriman berhasil
                            logger.info("Attempting to speak the summary...")
                            await self.session.say(text=preprocess_text_for_tts(f"Berikut ringkasan meetingnya: {summary_text}"), allow_interruptions=False)
                            logger.info("Summary spoken.")

                        except Exception as send_e:
                            logger.error(f"Error sending summary result via data channel or speaking it: {send_e}", exc_info=True)
                    # --- AKHIR PERBAIKAN PENGIRIMAN DATA ---

                else:
                    logger.warning("Summarize request received without transcript.")
            else:
                logger.debug(f"Received unhandled data message type: {msg_type}")

        except UnicodeDecodeError as ude:
            logger.error(f"Failed to decode data from {participant_identity} as UTF-8: {ude}")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON data from {participant_identity}")
        except Exception as e:
            logger.error(f"Error processing data from {participant_identity}: {e}", exc_info=True)

    @function_tool(description="Remember the user's name when they explicitly state it (e.g., 'My name is John').")
    async def remember_name(
        self,
        name: Annotated[str, "The user's name as stated by them."]
    ):
        if not name or not name.strip():
             logger.warning("LLM called remember_name with empty name.")
             return "Maaf, sepertinya Anda belum menyebutkan nama."
        if not self._user_id:
             logger.error("Cannot remember name: User ID not set in agent.")
             return "Maaf, terjadi masalah internal, saya tidak bisa mengingat nama saat ini."

        logger.info(f"LLM identified user's name: {name} for user {self._user_id}")
        self._user_name = name.strip().capitalize()

        if self._mem0_client:
            try:
                memory_to_store = f"The user stated their name is {self._user_name}."
                asyncio.create_task(asyncio.to_thread(
                    self._mem0_client.add,
                    memory_to_store,
                    user_id=self._user_id,
                    metadata={'category': 'personal_details', 'type': 'name', 'value': self._user_name}
                ))
                logger.info(f"Scheduled storage of user name memory for user {self._user_id}")
                return f"Baik, {self._user_name}. Senang mengetahui nama Anda. Saya akan mengingatnya."
            except Exception as e:
                logger.error(f"Failed to schedule store name in Mem0 for user {self._user_id}: {e}", exc_info=True)
                return f"Baik, {self._user_name}. Saya akan coba mengingatnya, tapi ada sedikit masalah dengan sistem memori jangka panjang saya."
        else:
            logger.warning(f"Cannot store name for user {self._user_id}: Mem0 client not available.")
            return f"Baik, {self._user_name}. Senang mengetahui nama Anda."

    @function_tool(description="Store important information, preferences, facts, goals, or concerns shared by the user.")
    async def remember_important_info(
        self,
        memory_topic: Annotated[str, f"A concise category for the information (e.g., {', '.join(MEMORY_TOPICS)}). Choose the most relevant category."],
        content: Annotated[str, "The specific piece of information, preference, or fact to remember, phrased clearly."],
    ):
        if not content or not content.strip():
            logger.warning("LLM called remember_important_info with empty content.")
            return "Maaf, sepertinya tidak ada informasi spesifik yang perlu diingat."
        if not memory_topic or not memory_topic.strip():
             logger.warning("LLM called remember_important_info with empty topic.")
             memory_topic = "general info"
        if not self._user_id:
             logger.error("Cannot remember info: User ID not set in agent.")
             return "Maaf, terjadi masalah internal, saya tidak bisa mengingat info saat ini."

        logger.info(f"LLM wants to remember for user {self._user_id}: Topic='{memory_topic}', Content='{content[:100]}...'")

        if not self._mem0_client:
            logger.warning(f"Cannot store info for user {self._user_id}: Mem0 client not available.")
            return "Saya akan coba mengingatnya untuk percakapan ini, tapi sistem memori jangka panjang saya sedang tidak aktif."
        try:
            data_to_store = f"User shared information related to '{memory_topic}': {content.strip()}"
            asyncio.create_task(asyncio.to_thread(
                self._mem0_client.add,
                data_to_store,
                user_id=self._user_id,
                metadata={'category': memory_topic.lower().replace(" ", "_"), 'value': content.strip()}
            ))
            logger.info(f"Scheduled storage of info in Mem0 for user {self._user_id}: Topic='{memory_topic}'")
            return f"Oke, saya sudah catat informasi tentang {memory_topic} itu."
        except Exception as e:
            logger.error(f"Failed to schedule store info in Mem0 for user {self._user_id}: {e}", exc_info=True)
            return "Maaf, terjadi masalah saat mencoba menyimpan informasi itu ke memori jangka panjang."

    @function_tool(description="Recall relevant past information based on a specific topic, keyword, or question about previous conversations. Optionally specify the maximum number of memories.")
    async def recall_memories(
        self,
        topic_query: Annotated[str, "A specific topic, keyword, or question about past information. Examples: 'my job concerns', 'what did we discuss about project X?', 'user goals', 'user name', 'details about my last vacation'. Be specific."],
        # Ubah 'limit' menjadi Optional dan hapus nilai default '= 3' dari signature
        limit: Annotated[Optional[int], "Optional. Maximum number of relevant memories to recall. Defaults to 3 if not specified."] = None
    ):
        if not topic_query or not topic_query.strip():
            logger.warning("LLM called recall_memories with empty query.")
            return "Untuk mengingat sesuatu, tolong beritahu topik atau kata kuncinya."
        if not self._user_id:
            logger.error("Cannot recall memories: User ID not set in agent.")
            return "Maaf, terjadi masalah internal, saya tidak bisa mengakses memori saat ini."
        if not self._mem0_client:
            logger.warning(f"Cannot recall memories for user {self._user_id}: Mem0 client not available.")
            return "Sistem memori jangka panjang saya tidak dapat diakses saat ini."

        # Tangani nilai default untuk 'limit' di dalam fungsi
        actual_limit = limit if limit is not None else 3
        # Pastikan limit berada dalam batas wajar (misalnya 1 hingga 5)
        safe_limit = max(1, min(actual_limit, 5))

        search_query = topic_query.strip()
        # Gunakan safe_limit untuk logging dan pencarian
        logger.info(f"Recalling memories for user {self._user_id} with query: '{search_query}' (limit: {safe_limit})")

        search_results = await search_mem0_with_timeout(
            self._mem0_client, self._user_id, search_query, safe_limit, timeout=MEM0_API_TIMEOUT
        )

        if search_results is None:
            return "Maaf, saya mengalami kesulitan saat mencoba mengakses memori jangka panjang."

        memories_content = []
        if isinstance(search_results, list):
            memories_content = [
                item.get('memory', '').strip() for item in search_results
                if isinstance(item, dict) and item.get('memory') and item.get('memory').strip()
            ]

        if not memories_content:
            logger.info(f"No relevant memories found for query: '{search_query}' for user {self._user_id}")
            return f"Saya sudah mencari, tapi tidak menemukan catatan spesifik tentang '{search_query}'."

        memory_text_formatted = "\n".join([f"- {mem}" for mem in memories_content])
        logger.info(f"Found {len(memories_content)} memories for query: '{search_query}' for user {self._user_id}")
        return f"Mengenai '{search_query}', ini beberapa hal yang saya ingat dari percakapan kita sebelumnya:\n{memory_text_formatted}"

    @function_tool(description="Sets an alarm on the user's connected device. "
                             "Requires the exact hour (0-23), minute (0-59), date (in YYYY-MM-DD format), and a descriptive message/label for the alarm. "
                             "Before calling this function, you MUST confirm all details (hour, minute, YYYY-MM-DD date, message) with the user. "
                             "Resolve relative dates like 'tomorrow' or 'next Tuesday' to the specific YYYY-MM-DD format based on the current date. "
                             "If any detail is missing, ask the user for it first instead of calling this function.")
    async def set_device_alarm(
        self,
        hour: Annotated[int, "The hour for the alarm (24-hour format, 0-23)."],
        minute: Annotated[int, "The minute for the alarm (0-59)."],
        date: Annotated[str, "The exact date for the alarm in YYYY-MM-DD format."],
        message: Annotated[str, "The descriptive message or label for the alarm (e.g., 'Meeting kantor bulanan', 'Jemput anak sekolah')."]
    ):
        logger.info(f"LLM requests to set alarm for user {self._user_id}: Date='{date}', Time={hour:02d}:{minute:02d}, Message='{message}'")

        # ... (Validasi input tetap sama) ...
        if not isinstance(hour, int) or not (0 <= hour <= 23):
            logger.error(f"Invalid hour received from LLM: {hour}")
            return "Maaf, jam alarm tidak valid (harus antara 0 dan 23)."
        if not isinstance(minute, int) or not (0 <= minute <= 59):
            logger.error(f"Invalid minute received from LLM: {minute}")
            return "Maaf, menit alarm tidak valid (harus antara 0 dan 59)."
        if not message or not message.strip():
            logger.error("Empty alarm message received from LLM.")
            return "Maaf, pesan untuk alarm tidak boleh kosong."
        try:
            datetime.datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid date format received from LLM: {date}")
            return f"Maaf, format tanggal ('{date}') sepertinya tidak valid. Gunakan format YYYY-MM-DD."
        if not self._user_id:
            logger.error("User ID not set in agent. Cannot send alarm command.")
            return "Maaf, saya tidak yakin harus mengirim perintah alarm ke siapa. Terjadi masalah internal."

        payload = {
            "type": "set_alarm",
            "hour": hour,
            "minute": minute,
            "date": date,
            "message": message.strip()
        }
        
        payload_str = json.dumps(payload)
        payload_bytes = payload_str.encode('utf-8')

        try:
            # --- PERUBAHAN DI SINI ---
            # Gunakan referensi local_participant yang disimpan secara eksplisit
            if not hasattr(self, '_local_participant_ref') or not self._local_participant_ref:
                logger.error("Cannot send data: Explicit local_participant reference not found or invalid in agent.")
                return "Maaf, terjadi masalah internal saat mencoba mengirim perintah alarm (participant reference missing)."

            logger.info(f"Sending 'set_alarm' command to user {self._user_id} via data channel using explicit local_participant reference: {payload_str}")
            # Gunakan self._local_participant_ref.publish_data
            await asyncio.wait_for(
                self._local_participant_ref.publish_data(payload=payload_bytes),
                timeout=DEVICE_ACTION_TIMEOUT
            )
            # --- AKHIR PERUBAHAN ---

            logger.info(f"Successfully sent 'set_alarm' command for user {self._user_id}.")
            return f"Oke, permintaan untuk menyetel alarm '{message}' pada {date} jam {hour:02d}:{minute:02d} sudah dikirim ke perangkat Anda."

        except AttributeError as ae:
            # Tangkap AttributeError jika _local_participant_ref tidak punya publish_data (seharusnya punya)
            logger.error(f"Failed to send 'set_alarm' command (AttributeError on local_participant?): {ae}", exc_info=True)
            return "Maaf, terjadi masalah internal saat mencoba mengirim perintah alarm (participant attribute error)."
        except asyncio.TimeoutError:
            logger.error(f"Timeout sending 'set_alarm' data for user {self._user_id}.")
            return "Maaf, butuh waktu terlalu lama untuk mengirim perintah alarm ke perangkat Anda. Silakan coba lagi."
        except ConnectionError as e:
            logger.error(f"Connection error sending 'set_alarm' command for user {self._user_id}: {e}")
            return "Maaf, sepertinya ada masalah koneksi saat mengirim perintah alarm ke perangkat Anda."
        except Exception as e:
            logger.error(f"Failed to send 'set_alarm' command via data channel for user {self._user_id}: {e}", exc_info=True)
            return "Maaf, terjadi kesalahan teknis saat mencoba mengirim perintah alarm."

    @function_tool(description="Search the internet for up-to-date information, current events, facts, or topics outside of your training data.")
    async def search_internet(
            self,
            query: Annotated[str, "The specific search query or question to look up online."]
    ):
        if not query or not query.strip():
            logger.warning("LLM called search_internet with empty query.")
            return preprocess_text_for_tts("Tolong berikan topik atau pertanyaan spesifik yang ingin Anda cari informasinya.") # Preprocess error message
        if not PERPLEXITY_API_KEY:
            logger.error("Cannot search internet: PERPLEXITY_API_KEY not configured.")
            return preprocess_text_for_tts("Maaf, saya tidak dapat melakukan pencarian internet saat ini karena konfigurasi API Key belum diatur.") # Preprocess error message

        logger.info(f"LLM requests internet search for user {self._user_id} with query: '{query}'")

        # --- LANGKAH 1: Ucapkan Filler ---
        filler_message = f"Baik, mohon tunggu sebentar ya selagi saya mencari informasi terbaru mengenai {query}... di internet untuk Anda.."
        logger.info(f"Speaking filler message: '{filler_message}'")
        try:
            # Preprocess filler message juga untuk konsistensi angka jika ada
            await self.session.say(text=preprocess_text_for_tts(filler_message), allow_interruptions=False)
        except Exception as say_err:
            logger.error(f"Error speaking filler message during search: {say_err}", exc_info=True)
            # Tidak perlu menghentikan musik di sini karena belum dimulai

        # --- LANGKAH 2: Kirim Perintah Mulai Musik Tunggu ---
        logger.info(f"Sending 'play_waiting_music' command to user {self._user_id}")
        start_payload = {"type": "play_waiting_music"}
        start_payload_bytes = json.dumps(start_payload).encode('utf-8')
        try:
            if not hasattr(self, '_local_participant_ref') or not self._local_participant_ref:
                 logger.error("Cannot send play_waiting_music: Explicit local_participant reference not found.")
            else:
                await asyncio.wait_for(
                    self._local_participant_ref.publish_data(payload=start_payload_bytes),
                    timeout=DEVICE_ACTION_TIMEOUT # Gunakan timeout yang sesuai
                )
                logger.info("Successfully sent 'play_waiting_music' command.")
        except Exception as e:
            logger.error(f"Failed to send 'play_waiting_music' command: {e}", exc_info=True)
            # Lanjutkan pencarian meskipun gagal mengirim perintah musik

        # --- LANGKAH 3: Lakukan Pencarian Internet ---
        search_content = None
        search_error_message = None
        try:
            headers = { # ... (headers seperti sebelumnya) ...
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            data = { # ... (data seperti sebelumnya) ...
                 "model": "sonar",
                 "messages": [
                     {"role": "system", "content": "You are an AI assistant that searches the internet to provide accurate, concise, and up-to-date answers in Indonesian based on the user's query. Cite sources if possible."},
                     {"role": "user", "content": query}
                 ]
            }

            logger.debug(f"Making request to Perplexity API with data: {json.dumps(data)}")
            async with aiohttp.ClientSession() as http_session:
                async with http_session.post(
                        "https://api.perplexity.ai/chat/completions",
                        headers=headers, json=data, timeout=INTERNET_SEARCH_TIMEOUT
                ) as response:
                    logger.debug(f"Perplexity API response status: {response.status}")
                    if response.status == 200:
                        result = await response.json()
                        logger.debug(f"Successfully received response from Perplexity API: {json.dumps(result)[:200]}...")
                        if "choices" in result and len(result["choices"]) > 0 and \
                           "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                            search_content = result["choices"][0]["message"]["content"] # Simpan ke variabel sementara
                            logger.info(f"Internet search successful for query: '{query}'. Result length: {len(search_content)}")
                        else:
                            logger.error(f"Unexpected response structure from Perplexity: {json.dumps(result)}")
                            search_error_message = "Maaf, saya menerima format respons yang tidak terduga dari layanan pencarian."
                    else:
                        error_text = await response.text()
                        logger.error(f"Error from Perplexity API (Status {response.status}): {error_text}")
                        if response.status == 401: search_error_message = "Maaf, terjadi masalah otentikasi dengan layanan pencarian."
                        elif response.status == 429: search_error_message = "Maaf, batas penggunaan layanan pencarian telah tercapai. Coba lagi nanti."
                        elif response.status >= 500: search_error_message = "Maaf, layanan pencarian sedang mengalami masalah internal."
                        else: search_error_message = f"Maaf, terjadi kesalahan saat mencari informasi (Kode: {response.status})."

        except asyncio.TimeoutError:
             logger.error(f"Internet search timed out after {INTERNET_SEARCH_TIMEOUT}s for query: '{query}'")
             search_error_message = "Maaf, pencarian informasi memakan waktu terlalu lama. Silakan coba lagi."
        except aiohttp.ClientError as e:
             logger.error(f"Network error during internet search: {e}", exc_info=True)
             search_error_message = "Maaf, terjadi masalah jaringan saat mencoba mencari informasi."
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Exception in search_internet: {str(e)}\n{error_details}")
            search_error_message = f"Maaf, terjadi kesalahan tak terduga saat mencoba melakukan pencarian: {str(e)}"

        # --- LANGKAH 4: Kirim Perintah Hentikan Musik Tunggu ---
        logger.info(f"Sending 'stop_waiting_music' command to user {self._user_id}")
        stop_payload = {"type": "stop_waiting_music"}
        stop_payload_bytes = json.dumps(stop_payload).encode('utf-8')
        try:
             if not hasattr(self, '_local_participant_ref') or not self._local_participant_ref:
                 logger.error("Cannot send stop_waiting_music: Explicit local_participant reference not found.")
             else:
                await asyncio.wait_for(
                    self._local_participant_ref.publish_data(payload=stop_payload_bytes),
                    timeout=DEVICE_ACTION_TIMEOUT # Gunakan timeout yang sesuai
                )
                logger.info("Successfully sent 'stop_waiting_music' command.")
                # Beri jeda SANGAT SINGKAT agar klien sempat memproses stop sebelum TTS mulai
                await asyncio.sleep(0.1) # 100ms delay
        except Exception as e:
            logger.error(f"Failed to send 'stop_waiting_music' command: {e}", exc_info=True)
            # Tetap lanjutkan untuk memberikan hasil/error

        # --- LANGKAH 5: Kembalikan Hasil atau Pesan Error (Akan di-TTS) ---
        if search_content:
             # Preprocess hasil sebelum dikembalikan
             return preprocess_text_for_tts(search_content)
        else:
             # Preprocess pesan error sebelum dikembalikan
             return preprocess_text_for_tts(search_error_message or "Maaf, terjadi kesalahan yang tidak diketahui saat mencari.")

def prewarm(proc: JobProcess):
    logger.info("Prewarming VAD model...")
    try:
        # Pastikan silero.VAD.load() adalah metode yang benar
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("VAD model prewarmed successfully.")
    except Exception as e:
        logger.error(f"Failed to prewarm VAD model: {e}", exc_info=True)
        proc.userdata["vad"] = None

# --- Fungsi Helper untuk TTS Completion Callback (Tetap sama) ---
async def publish_speech_end_flag(participant: Optional[LocalParticipant]):
    """Coroutine to send the agent_speech_end flag via data channel."""
    if not participant:
        logger.error("Cannot send speech end flag: local participant is None.")
        return

    payload_dict = {
        "type": "agent_speech_end",
        # Pertimbangkan time.time() jika perlu konsistensi dengan timestamp lain
        "timestamp": asyncio.get_event_loop().time()
    }
    payload_bytes = json.dumps(payload_dict).encode("utf-8")
    try:
        # --- MODIFIKASI PESAN LOG ---
        # logger.info(f"Sending agent_speech_end flag to room {participant.room.name}.") # <-- Baris Lama Penyebab Error
        logger.info(f"Sending agent_speech_end flag for participant {participant.identity}.") # <-- Log yang Disederhanakan

        # Mengirim data tetap sama
        await participant.publish_data(payload=payload_bytes, reliable=True, topic="agent_state")
        logger.info("agent_speech_end flag sent successfully.")
    except Exception as e:
        # Tambahkan exc_info untuk detail error yang lebih lengkap
        logger.error(f"Failed to publish agent_speech_end flag: {e}", exc_info=True)

def speech_completion_callback(handle: SpeechHandle, participant: Optional[LocalParticipant]):
    """
    Callback executed when the SpeechHandle completes (naturally or via interruption).
    Schedules the flag sending regardless of interruption status.
    """
    if participant is None:
        logger.error("Speech completion callback invoked with None participant.")
        return

    try:
      
        interruption_status = "interrupted" if handle.interrupted else "completed naturally"
        logger.info(f"Speech ended ({interruption_status}), scheduling agent_speech_end flag send.")
        asyncio.create_task(publish_speech_end_flag(participant))
     
    except asyncio.CancelledError:
        logger.info("Speech task was cancelled.") # Tetap tangani pembatalan eksplisit jika ada
    except Exception as e:
        # Tangkap error lain dalam logika callback
        logger.error(f"Error in speech completion callback logic: {e}", exc_info=True)

async def entrypoint(ctx: JobContext):
    start_entrypoint_time = time.time()
    # Dapatkan nama room dari context job SEBELUM connect
    ephemeral_room_name = ctx.job.room.name
    job_id = ctx.job.id
    logger.info(f"Initializing agent job {job_id} for room: {ephemeral_room_name}")

    persistent_user_id: Optional[str] = None
    local_mem0_client: Optional[MemoryClient] = None
    direct_openai_client: Optional[DirectAsyncOpenAI] = None
    extracted_user_name: Optional[str] = None
    session: Optional[AgentSession] = None # Inisialisasi session
    agent: Optional[AntyAgent] = None # Inisialisasi agent
    local_participant: Optional[LocalParticipant] = None # Inisialisasi local_participant
    room_data_handler_sync_defined = False # Flag untuk handler data
    room_data_handler_sync = None # Placeholder untuk fungsi handler

    try:
        # --- Ekstraksi User ID (Menggunakan Logika dari Kode Lama Anda) ---
        try:
            parts = ephemeral_room_name.split('-')
            # GUNAKAN LOGIKA YANG BENAR DARI KODE LAMA ANDA
            if len(parts) == 3 and parts[0] == "usession":
                persistent_user_id = parts[1]
                logger.info(f"Job {job_id}: Extracted persistent user_id: {persistent_user_id}")
            else:
                # Konsisten dengan kode lama, log error dan raise
                logger.error(f"Job {job_id}: Could not parse user_id from room name format: {ephemeral_room_name}")
                raise ValueError("Invalid room name format for user_id extraction")
        except Exception as e:
            logger.error(f"Job {job_id}: Error processing room name for user_id: {e}", exc_info=True)
            # Re-raise agar ditangkap oleh blok except utama entrypoint
            raise ValueError(f"Failed to determine persistent user_id from room name: {ephemeral_room_name}") from e

        # Pengecekan user ID setelah blok try-except parsing
        if not persistent_user_id:
             # Ini seharusnya tidak tercapai jika raise ValueError di atas bekerja
             logger.critical(f"FATAL: Job {job_id}: persistent_user_id is None after parsing attempt.")
             raise SystemExit("Could not obtain user_id")

        # Set log context setelah user_id pasti ada
        ctx.log_context_fields = {
            "room": ephemeral_room_name, # Gunakan nama room awal
            "user_id": persistent_user_id,
            "job_id": job_id,
        }
        logger.info(f"Log context updated with user_id: {persistent_user_id}")

        # --- Inisialisasi Mem0 (Tetap Sama) ---
        if MEM0_API_KEY:
            logger.info(f"Job {job_id}: Initializing Mem0 Client...")
            try:
                local_mem0_client = MemoryClient(api_key=MEM0_API_KEY) # Asumsi perlu API key
                logger.info(f"Job {job_id}: Mem0 Client initialized.")
                # ... (logika pencarian memori awal tetap sama) ...
                logger.info(f"Retrieving initial context from Mem0 for user '{persistent_user_id}'...")
                general_memories = await search_mem0_with_timeout(
                    local_mem0_client, persistent_user_id, SEMANTIC_QUERY_GENERAL_STARTUP, limit=5
                )
                if isinstance(general_memories, list) and general_memories:
                     retrieved_general_memory_texts = [ mem.get('memory') for mem in general_memories if isinstance(mem, dict) and mem.get('memory') ]
                     logger.info(f"Retrieved {len(retrieved_general_memory_texts)} general context memories.")
                     for mem_text in retrieved_general_memory_texts:
                          if "name is" in mem_text.lower():
                              try:
                                  match = re.search(r"name is\s+([a-zA-Z]+)", mem_text, re.IGNORECASE)
                                  if match:
                                       potential_name = match.group(1).strip().capitalize()
                                       if potential_name:
                                            extracted_user_name = potential_name
                                            logger.info(f"Tentatively extracted user name from memory: {extracted_user_name}")
                                            break
                              except Exception as e:
                                   logger.warning(f"Error extracting name from memory '{mem_text}': {e}")
                elif isinstance(general_memories, list):
                      logger.info(f"No general context memories found in Mem0 for user {persistent_user_id}.")
                else:
                      logger.warning(f"Failed to retrieve general context from Mem0 for user {persistent_user_id}.")
            except Exception as e:
                 logger.error(f"Job {job_id}: Failed to initialize Mem0 Client or retrieve context: {e}", exc_info=True)
                 local_mem0_client = None
        else:
             logger.warning(f"Job {job_id}: MEM0_API_KEY not found, Mem0 features disabled.")

        # --- Inisialisasi Direct OpenAI Client (Tetap Sama) ---
        if OPENAI_API_KEY:
             logger.info(f"Job {job_id}: Initializing Direct OpenAI Client...")
             try:
                 direct_openai_client = DirectAsyncOpenAI(api_key=OPENAI_API_KEY)
                 logger.info(f"Job {job_id}: Direct OpenAI Client initialized.")
             except Exception as e:
                  logger.error(f"Job {job_id}: Failed to initialize Direct OpenAI Client: {e}", exc_info=True)
                  direct_openai_client = None # Tetap None jika gagal
        else:
              logger.error(f"Job {job_id}: Cannot initialize Direct OpenAI Client - OPENAI_API_KEY missing.")

        # --- Koneksi ke Room ---
        logger.info(f"Job {job_id}: Connecting to LiveKit room...")
        await ctx.connect()
        # Setelah connect, ctx.room dan ctx.room.local_participant tersedia
        logger.info(f"Job {job_id}: Agent connected to room: {ctx.room.name}")

        # --- Dapatkan Local Participant ---
        if not ctx.room or not ctx.room.local_participant:
             logger.critical(f"Job {job_id}: Room or LocalParticipant not available after connect. Cannot proceed.")
             raise ConnectionError("Failed to get local participant after connecting.")
        local_participant = ctx.room.local_participant # Simpan untuk digunakan nanti
        logger.info(f"Job {job_id}: Obtained local participant: {local_participant.identity}")

        # --- Dapatkan VAD Plugin ---
        vad_plugin = ctx.proc.userdata.get("vad")
        if not vad_plugin:
             logger.error(f"Job {job_id}: VAD plugin not loaded during prewarm. Agent might not function correctly.")
             # Anda bisa fallback ke default VAD jika perlu:
             # logger.warning("Falling back to default Silero VAD.")
             # vad_plugin = silero.VAD.load() # Atau cara inisialisasi default VAD

        # --- Buat AgentSession ---
        logger.info(f"Job {job_id}: Creating AgentSession...")
        try:
            session = AgentSession(
                vad=vad_plugin,
                stt=groq.STT(model="whisper-large-v3-turbo", language="id"),
                llm=openai.LLM(model="gpt-4o-mini"),
                tts=openai.TTS(voice="nova"),
                turn_detection=MultilingualModel(),
            )
            logger.info(f"Job {job_id}: AgentSession created.")
        except Exception as e:
             logger.error(f"Job {job_id}: Failed to create AgentSession: {e}", exc_info=True)
             raise RuntimeError("Failed to create AgentSession") from e
        
        if session and local_participant:
            callback_with_context = partial(speech_completion_callback, participant=local_participant)

            @session.on("speech_created")
            def on_speech_created(event: SpeechCreatedEvent): # noqa: F841 <-- Opsional
                """Listener yang dipanggil setiap kali agent mulai berbicara."""
                logger.info(f"Agent speech created, attaching completion callback.") # Log disederhanakan

                # --- MODIFIKASI AKSES ATRIBUT DI SINI ---
                # handle: SpeechHandle = event.handle # <-- Baris Lama
                handle: SpeechHandle = event.speech_handle # <-- Atribut yang Benar ([1])

                # Daftarkan callback ke handle yang didapatkan
                handle.add_done_callback(callback_with_context)

            logger.info(f"Job {job_id}: Registered listener for 'speech_created' event.")
        else:
            logger.error(f"Job {job_id}: Cannot register speech_created listener...")

        # --- Setup Metrics Collector (Tetap Sama) ---
        usage_collector = metrics.UsageCollector()
        @session.on("metrics_collected")
        def _on_metrics_collected(ev: MetricsCollectedEvent):
            usage_collector.collect(ev.metrics)

        async def log_final_usage():
            summary = usage_collector.get_summary()
            logger.info(f"Job {job_id}: Final Usage Summary: {summary}")

        ctx.add_shutdown_callback(log_final_usage)
        logger.info(f"Job {job_id}: Metrics collector and shutdown logger registered.")

        # --- Buat Instance Agent ---
        logger.info(f"Job {job_id}: Instantiating AntyAgent...")
        try:
            agent = AntyAgent(
                mem0_client=local_mem0_client,
                user_id=persistent_user_id,
                direct_openai_client=direct_openai_client,
                local_participant=local_participant, # Berikan local_participant yang valid
            )
            agent._user_name = extracted_user_name
            logger.info(f"Job {job_id}: AntyAgent instantiated.")
        except Exception as e:
             logger.error(f"Job {job_id}: Failed to instantiate AntyAgent: {e}", exc_info=True)
             raise RuntimeError("Failed to instantiate AntyAgent") from e

        # --- Definisikan dan Daftarkan Data Handler (Tetap Sama) ---
        # Pastikan agent sudah ada
        if agent:
            def room_data_handler_sync_local(data_packet: DataPacket, participant: Optional[RemoteParticipant] = None):
                participant_identity: Optional[str] = None
                if participant is not None:
                    participant_identity = participant.identity
                    logger.debug(f"Job {job_id}: Room sync handler received data from {participant_identity}, scheduling async handler.")
                else:
                    logger.debug(f"Job {job_id}: Room sync handler received data, but participant is None. Data Kind: {data_packet.kind}, Data: {data_packet.data[:50]}...")

                # Panggil _on_data_received dari instance agent
                asyncio.create_task(agent._on_data_received(data_packet.data, participant_identity))

            room_data_handler_sync = room_data_handler_sync_local # Assign ke variabel di scope luar
            ctx.room.on("data_received", room_data_handler_sync)
            room_data_handler_sync_defined = True
            logger.info(f"Job {job_id}: Registered synchronous data received handler directly on Room object.")
        else:
             logger.error(f"Job {job_id}: Agent not instantiated, cannot define or register room data handler.")


        # --- Tunggu Partisipan dan Mulai Session ---
        logger.info(f"Job {job_id}: Starting AgentSession process...")
        logger.info(f"Job {job_id}: Waiting for participant to join...")
        await ctx.wait_for_participant() # Tunggu partisipan non-agent
        logger.info(f"Job {job_id}: Participant joined. Starting session processing.")

        # Mulai session HANYA jika agent dan session sudah dibuat
        if agent and session:
            await session.start(
                agent=agent,
                room=ctx.room,
                room_input_options=RoomInputOptions(),
                room_output_options=RoomOutputOptions(transcription_enabled=True),
            )
            logger.info(f"Job {job_id}: AgentSession started successfully.")
        else:
             logger.critical(f"Job {job_id}: Cannot start AgentSession because agent or session is None.")
             raise RuntimeError("Agent or Session not initialized correctly.")

        # --- Logika Salam Pembuka dengan Callback (Tetap Sama) ---
        logger.info(f"Agent initial greeting logic started for user {persistent_user_id}")
        start_greeting_time = time.time()
        greeting_text = f"Halo{' ' + agent._user_name if agent._user_name else ''}, Ada yang bisa saya bantu hari ini?"
        processed_greeting = preprocess_text_for_tts(greeting_text)
        logger.info(f"Speaking initial greeting: '{processed_greeting}'")

        if session:
            try:
                # Cukup panggil say() atau generate_reply(). Listener event akan menangani callback.
                await session.say(
                    text=processed_greeting,
                    allow_interruptions=True
                )
                logger.info("Initial greeting speech initiated.")
            except Exception as e:
                logger.error(f"Error during initial greeting speech: {e}", exc_info=True)
        else:
            logger.error("Cannot initiate greeting: AgentSession not available.")


        logger.info(f"Agent initial greeting logic finished in {time.time() - start_greeting_time:.2f}s")
        # --- Akhir Logika Salam Pembuka ---

        total_setup_time = time.time() - start_entrypoint_time
        logger.info(f"Job {job_id}: Agent setup complete. Total time: {total_setup_time:.2f} seconds.")

        # --- Loop Utama Agent ---
        logger.info(f"Job {job_id}: Agent running...")
        await asyncio.Future() # Keep the entrypoint alive

    except ValueError as e:
        # Tangkap error spesifik dari ekstraksi user ID atau validasi lain
        logger.error(f"CRITICAL: Job {job_id}: Configuration or setup error. Agent cannot proceed. Error: {e}", exc_info=True)
    except ConnectionError as e:
        # Tangkap error koneksi
        logger.error(f"CRITICAL: Job {job_id}: Connection error. Agent cannot proceed. Error: {e}", exc_info=True)
    except SystemExit as e:
        # Tangkap SystemExit jika user ID tidak ditemukan
        logger.critical(f"CRITICAL: Job {job_id}: SystemExit due to missing user ID or critical setup failure: {e}")
        # Tidak perlu raise lagi, biarkan finally berjalan
    except RuntimeError as e:
         # Tangkap error runtime dari inisialisasi komponen
         logger.error(f"CRITICAL: Job {job_id}: Runtime error during initialization. Error: {e}", exc_info=True)
    except asyncio.CancelledError:
        logger.info(f"Agent job {job_id} cancelled.")
    except Exception as e:
        # Tangkap semua error lain yang tidak terduga
        logger.error(f"CRITICAL: Unhandled exception in agent entrypoint for Job {job_id}: {e}", exc_info=True)
    finally:
        logger.info(f"Starting graceful shutdown sequence for Job {job_id}...")
        shutdown_start_time = time.time()

        # --- Logika Shutdown yang Lebih Aman ---
        # 1. Unregister event handlers
        try:
            if 'ctx' in locals() and ctx.room and hasattr(ctx.room, 'off') and room_data_handler_sync_defined and room_data_handler_sync:
                ctx.room.off("data_received", room_data_handler_sync)
                logger.info(f"Job {job_id}: Unregistered room data received handler.")
            elif room_data_handler_sync_defined:
                 logger.warning(f"Job {job_id}: Could not unregister room data handler during shutdown (context/room missing but handler was defined).")
            else:
                 logger.info(f"Job {job_id}: Room data handler was not registered, skipping unregistration.")
        except Exception as e:
            logger.warning(f"Job {job_id}: Error unregistering room data handler during shutdown: {e}")

        # 2. Hentikan AgentSession
        if session and hasattr(session, 'aclose'):
            try:
                await session.aclose()
                logger.info(f"Job {job_id}: AgentSession closed successfully.")
            # --- PERUBAHAN DI SINI ---
            except RuntimeError as e:
                # Tangkap RuntimeError secara spesifik
                if "no running event loop" in str(e):
                    # Jika itu error yang kita harapkan saat shutdown, log sebagai warning
                    logger.warning(f"Job {job_id}: Encountered expected 'no running event loop' during session.aclose() in shutdown. Error: {e}")
                else:
                    # Jika RuntimeError lain, tetap log sebagai error
                    logger.error(f"Job {job_id}: Unexpected RuntimeError closing AgentSession: {e}", exc_info=True)
            # --- AKHIR PERUBAHAN ---
            except Exception as e:
                # Tangkap error umum lainnya
                logger.error(f"Job {job_id}: Generic error closing AgentSession: {e}", exc_info=True)
        else:
            logger.info(f"Job {job_id}: AgentSession not available or already closed.")

        # 3. Tutup koneksi lain (misal OpenAI Client jika perlu)
        # Biasanya tidak perlu ditutup manual, library menangani
        if direct_openai_client and hasattr(direct_openai_client, 'close'):
              logger.info(f"Job {job_id}: (Skipping explicit close for Direct OpenAI Client)")
              pass

        # 4. Disconnect dari Room
        try:
            if 'ctx' in locals() and hasattr(ctx, 'disconnect') and ctx.room and ctx.room.connection_state == ConnectionState.CONN_CONNECTED:
                 await ctx.disconnect()
                 logger.info(f"Job {job_id}: Agent disconnected from room.")
            elif 'ctx' in locals() and ctx.room:
                 logger.info(f"Job {job_id}: Agent already disconnected or context/room not available.")
            else:
                 logger.info(f"Job {job_id}: Context not available, skipping disconnect call.")
        except Exception as e:
             logger.error(f"Job {job_id}: Error disconnecting from room during shutdown: {e}", exc_info=True)


        logger.info(f"Agent shutdown sequence for Job {job_id} completed in {time.time() - shutdown_start_time:.2f}s")

if __name__ == "__main__":
    logger.info("Starting LiveKit Agent worker...")

    worker_options = WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
    )

    use_ssl = os.getenv('USE_SSL', 'false').lower() == 'true'
    cert_path = 'certs/cert.pem'
    key_path = 'certs/key.pem'
    if use_ssl and os.path.exists(cert_path) and os.path.exists(key_path):
        worker_options.ssl_certfile = cert_path
        worker_options.ssl_keyfile = key_path
        logger.info("SSL certificates found and will be used.")
    else:
        logger.info("Running without SSL.")

    try:
        cli.run_app(worker_options)
    except KeyboardInterrupt:
        logger.info("Worker stopped by user (KeyboardInterrupt).")
    except Exception as e:
        logger.error(f"Worker failed unexpectedly: {e}", exc_info=True)
    finally:
        logger.info("LiveKit Agent worker finished.")