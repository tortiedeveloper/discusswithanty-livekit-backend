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
from livekit.rtc import Room, DataPacketKind, LocalParticipant, RemoteParticipant, DataPacket

from mem0 import MemoryClient

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

class AntyAgent(Agent):
    def __init__(
        self,
        mem0_client: Optional[MemoryClient],
        user_id: Optional[str],
        direct_openai_client: DirectAsyncOpenAI,
        local_participant: LocalParticipant,
    ) -> None:
        super().__init__(
            instructions=(
                "Anda adalah 'Anty', asisten suara yang ramah dan empatik dalam Bahasa Indonesia. "
                "Kepribadian Anda suportif, membantu, dan sedikit informal namun selalu sopan. "
                "Jaga agar respons tetap ringkas dan percakapan dalam Bahasa Indonesia. "
                "Gunakan fungsi yang tersedia jika diperlukan untuk mengingat informasi, mencari di internet, atau mengatur alarm."
            )
        )
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
                            await self.session.say(text=f"Berikut ringkasan meetingnya: {summary_text}", allow_interruptions=True)
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
            return "Tolong berikan topik atau pertanyaan spesifik yang ingin Anda cari informasinya."
        if not PERPLEXITY_API_KEY:
            logger.error("Cannot search internet: PERPLEXITY_API_KEY not configured.")
            return "Maaf, saya tidak dapat melakukan pencarian internet saat ini karena konfigurasi API Key belum diatur."

        logger.info(f"LLM requests internet search for user {self._user_id} with query: '{query}'")

        filler_message = f"Oke, saya coba cari informasi terbaru tentang '{query}...' ya."
        logger.info(f"Speaking filler message: '{filler_message}'")
        try:
            await self.session.say(text=filler_message, allow_interruptions=True)
        except Exception as say_err:
            logger.error(f"Error speaking filler message during search: {say_err}", exc_info=True)

        try:
            headers = {
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            data = {
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
                        headers=headers,
                        json=data,
                        timeout=INTERNET_SEARCH_TIMEOUT
                ) as response:
                    logger.debug(f"Perplexity API response status: {response.status}")
                    if response.status == 200:
                        result = await response.json()
                        logger.debug(f"Successfully received response from Perplexity API: {json.dumps(result)[:200]}...")
                        if "choices" in result and len(result["choices"]) > 0 and \
                           "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                            content = result["choices"][0]["message"]["content"]
                            logger.info(f"Internet search successful for query: '{query}'. Result length: {len(content)}")
                            return content
                        else:
                            logger.error(f"Unexpected response structure from Perplexity: {json.dumps(result)}")
                            return "Maaf, saya menerima format respons yang tidak terduga dari layanan pencarian."
                    else:
                        error_text = await response.text()
                        logger.error(f"Error from Perplexity API (Status {response.status}): {error_text}")
                        if response.status == 401: return "Maaf, terjadi masalah otentikasi dengan layanan pencarian."
                        elif response.status == 429: return "Maaf, batas penggunaan layanan pencarian telah tercapai. Coba lagi nanti."
                        elif response.status >= 500: return "Maaf, layanan pencarian sedang mengalami masalah internal."
                        else: return f"Maaf, terjadi kesalahan saat mencari informasi (Kode: {response.status})."

        except asyncio.TimeoutError:
             logger.error(f"Internet search timed out after {INTERNET_SEARCH_TIMEOUT}s for query: '{query}'")
             return "Maaf, pencarian informasi memakan waktu terlalu lama. Silakan coba lagi."
        except aiohttp.ClientError as e:
             logger.error(f"Network error during internet search: {e}", exc_info=True)
             return "Maaf, terjadi masalah jaringan saat mencoba mencari informasi."
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Exception in search_internet: {str(e)}\n{error_details}")
            return f"Maaf, terjadi kesalahan tak terduga saat mencoba melakukan pencarian: {str(e)}"

def prewarm(proc: JobProcess):
    logger.info("Prewarming VAD model...")
    try:
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("VAD model prewarmed successfully.")
    except Exception as e:
        logger.error(f"Failed to prewarm VAD model: {e}", exc_info=True)
        proc.userdata["vad"] = None

async def entrypoint(ctx: JobContext):
    start_entrypoint_time = time.time()
    ephemeral_room_name = ctx.room.name
    job_id = ctx.job.id
    logger.info(f"Initializing agent job {job_id} for room: {ephemeral_room_name}")

    persistent_user_id: Optional[str] = None
    local_mem0_client: Optional[MemoryClient] = None
    direct_openai_client: Optional[DirectAsyncOpenAI] = None
    extracted_user_name: Optional[str] = None

    try:
        try:
            parts = ephemeral_room_name.split('-')
            if len(parts) == 3 and parts[0] == "usession":
                persistent_user_id = parts[1]
                logger.info(f"Job {job_id}: Extracted persistent user_id: {persistent_user_id}")
            else:
                logger.error(f"Job {job_id}: Could not parse user_id from room name format: {ephemeral_room_name}")
                raise ValueError("Invalid room name format for user_id extraction")
        except Exception as e:
            logger.error(f"Job {job_id}: Error extracting user_id: {e}", exc_info=True)
            raise ValueError("Failed to determine persistent user_id from room name") from e

        if not persistent_user_id:
             logger.critical(f"FATAL: Job {job_id}: persistent_user_id is None.")
             raise SystemExit("Could not obtain user_id")

        ctx.log_context_fields = {
            "room": ctx.room.name,
            "user_id": persistent_user_id,
            "job_id": job_id,
        }
        logger.info(f"Log context updated with user_id: {persistent_user_id}")

        if MEM0_API_KEY:
            logger.info(f"Job {job_id}: Initializing Mem0 Client...")
            try:
                local_mem0_client = MemoryClient()
                logger.info(f"Job {job_id}: Mem0 Client initialized.")

                logger.info(f"Retrieving initial context from Mem0 for user '{persistent_user_id}'...")
                general_memories = await search_mem0_with_timeout(
                    local_mem0_client, persistent_user_id, SEMANTIC_QUERY_GENERAL_STARTUP, limit=5
                )
                if isinstance(general_memories, list):
                    retrieved_general_memory_texts = [
                        mem.get('memory') for mem in general_memories
                        if isinstance(mem, dict) and mem.get('memory')
                    ]
                    logger.info(f"Retrieved {len(retrieved_general_memory_texts)} general context memories.")
                    for mem_text in retrieved_general_memory_texts:
                        if "name is" in mem_text.lower():
                            try:
                                parts = mem_text.lower().split("name is", 1)
                                if len(parts) > 1:
                                    potential_name = parts[1].strip().split()[0].rstrip('.?!,').capitalize()
                                    if potential_name:
                                        extracted_user_name = potential_name
                                        logger.info(f"Tentatively extracted user name from memory: {extracted_user_name}")
                                        break
                            except Exception as e:
                                logger.warning(f"Error extracting name from memory '{mem_text}': {e}")
                else:
                    logger.warning(f"Failed to retrieve general context or none found for user {persistent_user_id}.")

            except Exception as e:
                logger.error(f"Job {job_id}: Failed to initialize Mem0 Client or retrieve context: {e}", exc_info=True)
                local_mem0_client = None
        else:
            logger.warning(f"Job {job_id}: MEM0_API_KEY not found, Mem0 features disabled.")

        if OPENAI_API_KEY:
             logger.info(f"Job {job_id}: Initializing Direct OpenAI Client...")
             direct_openai_client = DirectAsyncOpenAI(api_key=OPENAI_API_KEY)
             logger.info(f"Job {job_id}: Direct OpenAI Client initialized.")
        else:
             logger.error(f"Job {job_id}: Cannot initialize Direct OpenAI Client - OPENAI_API_KEY missing.")

        logger.info(f"Job {job_id}: Connecting to LiveKit room...")
        await ctx.connect()
        logger.info(f"Job {job_id}: Agent connected to room: {ephemeral_room_name}")

        vad_plugin = ctx.proc.userdata.get("vad")
        if not vad_plugin:
             logger.error(f"Job {job_id}: VAD plugin not loaded during prewarm. Agent might not function correctly.")

        logger.info(f"Job {job_id}: Creating AgentSession...")
        session = AgentSession(
            vad=vad_plugin,
            stt=groq.STT(model="whisper-large-v3-turbo", language="id"),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=openai.TTS(voice="nova"),
            turn_detection=MultilingualModel(),
        )
        logger.info(f"Job {job_id}: AgentSession created.")

        usage_collector = metrics.UsageCollector()
        @session.on("metrics_collected")
        def _on_metrics_collected(ev: MetricsCollectedEvent):
            usage_collector.collect(ev.metrics)

        async def log_final_usage():
            summary = usage_collector.get_summary()
            logger.info(f"Job {job_id}: Final Usage Summary: {summary}")

        ctx.add_shutdown_callback(log_final_usage)
        logger.info(f"Job {job_id}: Metrics collector and shutdown logger registered.")

        if not ctx.room or not ctx.room.local_participant:
            logger.critical(f"Job {job_id}: Room or LocalParticipant not available after connect. Cannot proceed.")
            # Anda mungkin ingin menangani error ini dengan lebih baik, misal raise Exception
            raise ConnectionError("Failed to get local participant after connecting.")
        
        logger.info(f"Job {job_id}: Instantiating AntyAgent...")
        agent = AntyAgent(
            mem0_client=local_mem0_client,
            user_id=persistent_user_id,
            direct_openai_client=direct_openai_client,
            local_participant=ctx.room.local_participant, # <-- Teruskan local_participant
        )
        agent._user_name = extracted_user_name

        def room_data_handler_sync(data_packet: DataPacket, participant: Optional[RemoteParticipant] = None): # Tambahkan '= None'
            # Handler untuk Room.on("data_received")
            # Sekarang participant bisa None jika tidak disediakan oleh emitter
            participant_identity: Optional[str] = None # Default ke None
            if participant is not None:
                participant_identity = participant.identity # Dapatkan identity jika participant ada
                logger.debug(f"Job {job_id}: Room sync handler received data from {participant_identity}, scheduling async handler.")
            else:
                # Log jika participant None
                logger.debug(f"Job {job_id}: Room sync handler received data, but participant is None. Data: {data_packet.data[:50]}...")

            # Tetap panggil _on_data_received, teruskan participant_identity (bisa None)
            # Pastikan _on_data_received bisa menangani participant_identity == None jika perlu
            asyncio.create_task(agent._on_data_received(data_packet.data, participant_identity))

        # Pendaftaran handler tetap sama:
        ctx.room.on("data_received", room_data_handler_sync)
        logger.info(f"Job {job_id}: Registered synchronous data received handler directly on Room object.")
        

        logger.info(f"Job {job_id}: Starting AgentSession...")
        logger.info(f"Job {job_id}: Waiting for participant to join...")
        await ctx.wait_for_participant()
        logger.info(f"Job {job_id}: Participant joined. Starting session processing.")

        await session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(),
            room_output_options=RoomOutputOptions(transcription_enabled=True),
        )
        logger.info(f"Job {job_id}: AgentSession started successfully.")

        # --- Logika Inisialisasi yang Dipindahkan (sebelumnya di AntyAgent.setup) ---
        logger.info(f"Agent initial greeting logic started for user {persistent_user_id}")
        start_greeting_time = time.time()

        greeting_text = f"Halo{' ' + agent._user_name if agent._user_name else ''}, Ada yang bisa saya bantu hari ini?"

        logger.info(f"Speaking initial greeting: '{greeting_text}'")
        try:
             session.generate_reply(instructions=greeting_text)
             logger.info("Initial greeting generation requested from LLM.")
        except Exception as e:
            logger.error(f"Error during initial greeting: {e}", exc_info=True)

        logger.info(f"Agent initial greeting logic finished in {time.time() - start_greeting_time:.2f}s")
        # --- Akhir Logika Inisialisasi yang Dipindahkan ---

        total_setup_time = time.time() - start_entrypoint_time
        logger.info(f"Job {job_id}: Agent setup complete. Total time: {total_setup_time:.2f} seconds.")

        await asyncio.Future()

    except ValueError as e:
        logger.error(f"CRITICAL: Job {job_id}: Could not obtain persistent user_id. Agent cannot proceed. Error: {e}")
    except asyncio.CancelledError:
        logger.info(f"Agent job {job_id} cancelled.")
    except Exception as e:
        logger.error(f"Unhandled error in agent entrypoint for Job {job_id}: {e}", exc_info=True)
    finally:
        logger.info(f"Starting shutdown sequence for Job {job_id}...")
        shutdown_start_time = time.time()

        try:
            if 'ctx' in locals() and ctx.room and hasattr(ctx.room, 'off') and 'room_data_handler_sync' in locals():
                 ctx.room.off("data_received", room_data_handler_sync)
                 logger.info(f"Job {job_id}: Unregistered room data received handler.")
            else:
                 logger.warning(f"Job {job_id}: Could not unregister room data handler during shutdown (context/room/handler missing).")
        except Exception as e:
            logger.warning(f"Job {job_id}: Error unregistering room data handler during shutdown: {e}")

        if direct_openai_client and hasattr(direct_openai_client, 'close'):
             logger.info(f"Job {job_id}: (Skipping explicit close for Direct OpenAI Client)")
             pass

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