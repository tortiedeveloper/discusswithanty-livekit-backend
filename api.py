import asyncio
import json
import os
import aiohttp
import traceback
from typing import Annotated, List, Optional, Callable, Awaitable
from livekit.agents import llm
import logging
import random
import time
from datetime import datetime

logger = logging.getLogger("assistant-api")
logger.setLevel(logging.INFO)

MEMORY_TOPICS = ["personal info", "preferences", "concerns", "goals", "life events", "relationships", "user name", "user age", "past advice", "feedback", "meeting schedule", "important dates"]
SEMANTIC_QUERY_RECALL_DEFAULT = "Information relevant to the user's current topic or question"
SEMANTIC_QUERY_NAME_RECALL = "What is the user's name?"
MEM0_API_TIMEOUT = 10.0
DEVICE_ACTION_TIMEOUT = 15.0
INTERNET_SEARCH_TIMEOUT = 25.0

SendDataCallback = Callable[[str], Awaitable[None]]
AssistantSayCallback = Callable[[str, bool], Awaitable[None]]

class AssistantFnc(llm.FunctionContext):
    def __init__(self,
                 client,
                 send_data_callback: Optional[SendDataCallback] = None
                 ) -> None:
        super().__init__()
        self._mem0_client = client
        self._send_data_callback = send_data_callback
        self._assistant_say_callback: Optional[AssistantSayCallback] = None
        self._current_user_id: Optional[str] = None
        self._user_name: Optional[str] = None
        logger.info(f"AssistantFnc initialized. Mem0: {client is not None}, SendData: {send_data_callback is not None}")

    def set_assistant_say_callback(self, say_callback: AssistantSayCallback):
        self._assistant_say_callback = say_callback
        logger.info("Assistant 'say' callback has been set in AssistantFnc.")

    async def set_user_id(self, user_id: str):
        if not user_id:
             logger.error("Attempted to set an empty user_id.")
             return
        self._current_user_id = user_id
        logger.info(f"Set current user ID for API context: {user_id}")

        if not self._mem0_client:
            logger.warning("Mem0 Client not available in API context. Cannot recall name on set_user_id.")
            return
        try:
            start_time = time.time()
            logger.debug(f"Attempting initial name recall (in thread) for {user_id} in API context.")
            search_coro = asyncio.to_thread(
                self._mem0_client.search,
                query=SEMANTIC_QUERY_NAME_RECALL,
                user_id=self._current_user_id,
                limit=1
            )
            name_memories = await asyncio.wait_for(search_coro, timeout=MEM0_API_TIMEOUT)
            logger.debug(f"Initial name recall search took {time.time() - start_time:.2f}s")

            if isinstance(name_memories, list) and name_memories:
                memory_text = name_memories[0].get("memory", "")
                if memory_text and "name is" in memory_text.lower():
                    try:
                        potential_name = memory_text.lower().split("name is", 1)[1].strip().rstrip('.?!').capitalize()
                        if potential_name:
                            self._user_name = potential_name
                            logger.info(f"Tentatively cached user name from initial recall: {self._user_name}")
                    except IndexError:
                        logger.warning(f"Could not parse name from memory: '{memory_text}'")
                    except Exception as e:
                        logger.error(f"Error parsing name from memory: {e}", exc_info=True)
        except asyncio.TimeoutError:
             logger.warning(f"Initial name recall timed out after {MEM0_API_TIMEOUT}s.")
        except AttributeError:
             logger.warning("Mem0 client instance does not have a 'search' method or call failed.")
        except Exception as e:
            logger.error(f"Error during initial name recall in API context: {e}", exc_info=True)

    @llm.ai_callable(description="Remember the user's name when they explicitly state it (e.g., 'My name is John').")
    def remember_name(
        self,
        name: Annotated[str, llm.TypeInfo(description="The user's name as stated by them.")]
    ):
        if not name or not name.strip():
             logger.warning("LLM called remember_name with empty name.")
             return "Maaf, sepertinya Anda belum menyebutkan nama."

        logger.info(f"LLM identified user's name: {name}")
        self._user_name = name.strip().capitalize()

        if self._current_user_id and self._mem0_client:
            try:
                memory_to_store = f"The user stated their name is {self._user_name}."
                self._mem0_client.add(
                    memory_to_store,
                    user_id=self._current_user_id,
                    metadata={'category': 'personal_details', 'type': 'name', 'value': self._user_name}
                )
                logger.info(f"Stored user name memory for user {self._current_user_id}")
                return f"Baik, {self._user_name}. Senang mengetahui nama Anda. Saya akan mengingatnya."
            except Exception as e:
                logger.error(f"Failed to store name in Mem0 for user {self._current_user_id}: {e}", exc_info=True)
                return f"Baik, {self._user_name}. Saya akan coba mengingatnya, tapi ada sedikit masalah dengan sistem memori jangka panjang saya."
        else:
            logger.warning("Cannot store name: User ID or Mem0 client not available.")
            return f"Baik, {self._user_name}. Senang mengetahui nama Anda."

    @llm.ai_callable(description="Store important information, preferences, facts, goals, or concerns shared by the user.")
    def remember_important_info(
        self,
        memory_topic: Annotated[str, llm.TypeInfo(description=f"A concise category for the information (e.g., {', '.join(MEMORY_TOPICS)}). Choose the most relevant category.")],
        content: Annotated[str, llm.TypeInfo(description="The specific piece of information, preference, or fact to remember, phrased clearly.")],
    ):
        if not content or not content.strip():
            logger.warning("LLM called remember_important_info with empty content.")
            return "Maaf, sepertinya tidak ada informasi spesifik yang perlu diingat."
        if not memory_topic or not memory_topic.strip():
             logger.warning("LLM called remember_important_info with empty topic.")
             memory_topic = "general info"

        logger.info(f"LLM wants to remember: Topic='{memory_topic}', Content='{content[:100]}...'")

        if not self._current_user_id or not self._mem0_client:
            logger.warning("Cannot store info: User ID or Mem0 client not available.")
            return "Saya akan coba mengingatnya untuk percakapan ini, tapi sistem memori jangka panjang saya sedang tidak aktif."
        try:
            data_to_store = f"User shared information related to '{memory_topic}': {content.strip()}"
            self._mem0_client.add(
                data_to_store,
                user_id=self._current_user_id,
                metadata={'category': memory_topic.lower().replace(" ", "_"), 'value': content.strip()}
            )
            logger.info(f"Stored info in Mem0 for user {self._current_user_id}: Topic='{memory_topic}'")
            return f"Oke, saya sudah catat informasi tentang {memory_topic} itu."
        except Exception as e:
            logger.error(f"Failed to store info in Mem0 for user {self._current_user_id}: {e}", exc_info=True)
            return "Maaf, terjadi masalah saat mencoba menyimpan informasi itu ke memori jangka panjang."

    @llm.ai_callable(description="Recall relevant past information based on a specific topic, keyword, or question about previous conversations.")
    async def recall_memories(
        self,
        topic_query: Annotated[str, llm.TypeInfo(
            description="A specific topic, keyword, or question about past information. "
                        f"Examples: 'my job concerns', 'what did we discuss about project X?', 'user goals', 'user name', 'details about my last vacation'. Be specific."
        )],
        limit: Annotated[int, llm.TypeInfo(description="Maximum number of relevant memories to recall (default 3).")] = 3
    ):
        if not topic_query or not topic_query.strip():
            logger.warning("LLM called recall_memories with empty query.")
            return "Untuk mengingat sesuatu, tolong beritahu topik atau kata kuncinya."

        if not self._current_user_id or not self._mem0_client:
            logger.warning("Cannot recall memories: User ID or Mem0 client not available.")
            return "Sistem memori jangka panjang saya tidak dapat diakses saat ini."

        safe_limit = max(1, min(limit, 5))

        try:
            start_time = time.time()
            search_query = topic_query.strip()
            logger.info(f"Recalling memories (in thread) for user {self._current_user_id} with query: '{search_query}' (limit: {safe_limit})")

            search_coro = asyncio.to_thread(
                self._mem0_client.search,
                query=search_query,
                user_id=self._current_user_id,
                limit=safe_limit
            )
            search_results = await asyncio.wait_for(search_coro, timeout=MEM0_API_TIMEOUT)
            logger.debug(f"Memory recall search took {time.time() - start_time:.2f}s")

            memories_content = []
            if isinstance(search_results, list):
                memories_content = [
                    item.get('memory', '').strip() for item in search_results
                    if isinstance(item, dict) and item.get('memory') and item.get('memory').strip()
                ]

            if not memories_content:
                logger.info(f"No relevant memories found for query: '{search_query}'")
                return f"Saya sudah mencari, tapi tidak menemukan catatan spesifik tentang '{search_query}'."

            memory_text_formatted = "\n".join([f"- {mem}" for mem in memories_content])
            logger.info(f"Found {len(memories_content)} memories for query: '{search_query}'")
            return f"Mengenai '{search_query}', ini beberapa hal yang saya ingat dari percakapan kita sebelumnya:\n{memory_text_formatted}"

        except asyncio.TimeoutError:
             logger.warning(f"Memory recall timed out after {MEM0_API_TIMEOUT}s for query: '{search_query}'.")
             return "Maaf, saya butuh waktu terlalu lama untuk mencoba mengingat itu."
        except AttributeError:
             logger.warning("Mem0 client instance does not have a 'search' method or call failed.")
             return "Maaf, saya tidak bisa mengakses memori jangka panjang saat ini."
        except Exception as e:
            logger.error(f"Failed to recall memories from Mem0 for user {self._current_user_id}: {e}", exc_info=True)
            return "Terjadi masalah saat mencoba mengakses memori jangka panjang."

    @llm.ai_callable(description="Sets an alarm on the user's connected device. "
                                 "Requires the exact hour (0-23), minute (0-59), date (in YYYY-MM-DD format), and a descriptive message/label for the alarm. "
                                 "Before calling this function, you MUST confirm all details (hour, minute, YYYY-MM-DD date, message) with the user. "
                                 "Resolve relative dates like 'tomorrow' or 'next Tuesday' to the specific YYYY-MM-DD format based on the current date. "
                                 "If any detail is missing, ask the user for it first instead of calling this function.")
    async def set_device_alarm(
        self,
        hour: Annotated[int, llm.TypeInfo(description="The hour for the alarm (24-hour format, 0-23).")],
        minute: Annotated[int, llm.TypeInfo(description="The minute for the alarm (0-59).")],
        date: Annotated[str, llm.TypeInfo(description="The exact date for the alarm in YYYY-MM-DD format.")],
        message: Annotated[str, llm.TypeInfo(description="The descriptive message or label for the alarm (e.g., 'Meeting kantor bulanan', 'Jemput anak sekolah').")]
    ):
        logger.info(f"LLM requests to set alarm: Date='{date}', Time={hour:02d}:{minute:02d}, Message='{message}'")

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
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid date format received from LLM: {date}")
            return f"Maaf, format tanggal ('{date}') sepertinya tidak valid. Gunakan format YYYY-MM-DD."

        if not self._send_data_callback:
            logger.error("Send data callback is not configured. Cannot send alarm command.")
            return "Maaf, saya tidak dapat mengirim perintah alarm ke perangkat Anda saat ini karena masalah koneksi internal."

        if not self._current_user_id:
             logger.error("User ID not set. Cannot determine target for alarm command.")
             return "Maaf, saya tidak yakin harus mengirim perintah alarm ke siapa. Terjadi masalah internal."

        payload = {
            "type": "set_alarm",
            "hour": hour,
            "minute": minute,
            "date": date,
            "message": message.strip()
        }
        payload_str = json.dumps(payload)

        try:
            logger.info(f"Sending 'set_alarm' command to user {self._current_user_id}: {payload_str}")
            await asyncio.wait_for(
                self._send_data_callback(payload_str),
                timeout=DEVICE_ACTION_TIMEOUT
            )
            logger.info(f"Successfully sent 'set_alarm' command for user {self._current_user_id}.")
            return f"Oke, permintaan untuk menyetel alarm '{message}' pada {date} jam {hour:02d}:{minute:02d} sudah dikirim ke perangkat Anda."

        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for send_data_callback to complete for 'set_alarm'.")
            return "Maaf, butuh waktu terlalu lama untuk mengirim perintah alarm ke perangkat Anda. Silakan coba lagi."
        except ConnectionError as e:
             logger.error(f"Connection error sending 'set_alarm' command: {e}")
             return "Maaf, sepertinya ada masalah koneksi saat mengirim perintah alarm ke perangkat Anda."
        except Exception as e:
            logger.error(f"Failed to send 'set_alarm' command via callback for user {self._current_user_id}: {e}", exc_info=True)
            return "Maaf, terjadi kesalahan teknis saat mencoba mengirim perintah alarm."

    @llm.ai_callable(description="Search the internet for up-to-date information...")
    async def search_internet(
            self,
            query: Annotated[
                str, llm.TypeInfo(description="The specific search query...")
            ]
    ):
        if not query or not query.strip():
            logger.warning("LLM called search_internet with empty query.")
            return "Tolong berikan topik atau pertanyaan spesifik yang ingin Anda cari informasinya."

        logger.info(f"LLM requests internet search with query: '{query}'")

        # --- MEMBUAT TASK 'SAY' TEPAT SEBELUM REQUEST JARINGAN ---
        say_task = None
        if self._assistant_say_callback:
            filler_message = f"Oke, saya coba cari informasi terbaru tentang '{query[:30]}...' ya."
            logger.info(f"Creating background task to speak filler message: '{filler_message}'")
            try:
                # Buat task, jangan di-await. Berikan argumen allow_interruptions=True
                say_task = asyncio.create_task(self._assistant_say_callback(filler_message))
            except Exception as say_err:
                logger.error(f"Error creating background task for speaking: {say_err}", exc_info=True)
                # Tetap lanjutkan pencarian meskipun 'say' gagal dibuat tasknya
        else:
            logger.warning("Assistant 'say' callback not available, cannot speak filler message concurrently.")
        # --- AKHIR PEMBUATAN TASK 'SAY' ---

        try:
            perplexity_api_key = os.environ.get('PERPLEXITY_API_KEY')
            if not perplexity_api_key:
                logger.error("PERPLEXITY_API_KEY not found in environment variables.")
                # Jika task 'say' dibuat, kita mungkin ingin membatalkannya jika terjadi error awal
                if say_task and not say_task.done():
                    say_task.cancel()
                return "Maaf, saya tidak dapat melakukan pencarian internet saat ini karena konfigurasi API Key belum diatur."

            headers = {
                "Authorization": f"Bearer {perplexity_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            data = {
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": "You are an AI assistant that searches the internet to provide accurate, concise, and up-to-date answers based on the user's query. Cite sources if possible."},
                    {"role": "user", "content": query}
                ]
            }

            logger.debug(f"Making request to Perplexity API (sonar-medium-online) with data: {json.dumps(data)}")

            # --- REQUEST JARINGAN BERJALAN, TASK 'SAY' BERJALAN DI BACKGROUND ---
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        "https://api.perplexity.ai/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=INTERNET_SEARCH_TIMEOUT
                ) as response:
                    logger.debug(f"Perplexity API response status: {response.status}")

                    # --- PROSES RESPONSE SETELAH REQUEST SELESAI ---
                    if response.status == 200:
                        result = await response.json()
                        logger.debug(f"Successfully received response from Perplexity API: {json.dumps(result)[:200]}...")

                        if "choices" in result and len(result["choices"]) > 0 and \
                           "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                            content = result["choices"][0]["message"]["content"]
                            logger.info(f"Internet search successful for query: '{query}'. Result length: {len(content)}")
                            # Task 'say' mungkin masih berjalan atau sudah selesai, tidak masalah.
                            return content
                        else:
                            logger.error(f"Unexpected response structure from Perplexity: {json.dumps(result)}")
                            return "Maaf, saya menerima format respons yang tidak terduga dari layanan pencarian."
                    else:
                        error_text = await response.text()
                        logger.error(f"Error from Perplexity API (Status {response.status}): {error_text}")
                        if response.status == 401:
                             return "Maaf, terjadi masalah otentikasi dengan layanan pencarian."
                        elif response.status == 429:
                             return "Maaf, batas penggunaan layanan pencarian telah tercapai. Coba lagi nanti."
                        else:
                             return f"Maaf, terjadi kesalahan saat mencari informasi (Kode: {response.status})."

        except asyncio.TimeoutError:
             logger.error(f"Internet search timed out after {INTERNET_SEARCH_TIMEOUT}s for query: '{query}'")
             # Batalkan task 'say' jika masih berjalan saat timeout
             if say_task and not say_task.done():
                 say_task.cancel()
             return "Maaf, pencarian informasi memakan waktu terlalu lama. Silakan coba lagi."
        except aiohttp.ClientError as e:
             logger.error(f"Network error during internet search: {e}", exc_info=True)
             if say_task and not say_task.done():
                 say_task.cancel()
             return "Maaf, terjadi masalah jaringan saat mencoba mencari informasi."
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Exception in search_internet: {str(e)}\n{error_details}")
            if say_task and not say_task.done():
                 say_task.cancel()
            return f"Maaf, terjadi kesalahan tak terduga saat mencoba melakukan pencarian: {str(e)}"