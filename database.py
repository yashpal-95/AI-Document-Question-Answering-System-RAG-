# ============================================================
#  database.py  –  MongoDB async manager
# ============================================================

import motor.motor_asyncio
from datetime import datetime
from typing import Optional
import uuid

MONGO_DETAILS = "mongodb://localhost:27017/"


class MongoDBManager:
    def __init__(self):
        self.client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self.db               = None
        self.chat_collection  = None
        self.files_collection = None
        self.users_collection = None          # ← NEW: user registry

    # ── Connection ───────────────────────────────────────────

    def connect(self):
        try:
            self.client           = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
            self.db               = self.client["rag_database"]
            self.chat_collection  = self.db.get_collection("chat_history")
            self.files_collection = self.db.get_collection("user_files")
            self.users_collection = self.db.get_collection("users")   # ← NEW
            print("✅ Connected to MongoDB.")
        except Exception as e:
            print(f"❌ Could not connect to MongoDB: {e}")

    def disconnect(self):
        if self.client:
            self.client.close()
            print("🛑 Disconnected from MongoDB.")

    def _normalize(self, user_id: str) -> str:
        return user_id.strip().lower()

    # ── User Registry ─────────────────────────────────────────

    async def create_user(self, display_name: str) -> dict:
        """
        Create a new user with an auto-generated unique ID like user_01.
        Returns the created user document.
        """
        # Count existing users to generate next ID
        count = await self.users_collection.count_documents({})
        new_id = f"user_{str(count + 1).zfill(2)}"

        # Make sure ID doesn't already exist (handle gaps)
        while await self.users_collection.find_one({"user_id": new_id}):
            count += 1
            new_id = f"user_{str(count + 1).zfill(2)}"

        user_doc = {
            "user_id":      new_id,
            "display_name": display_name.strip(),
            "created_at":   datetime.utcnow(),
            "last_login":   datetime.utcnow(),
        }
        await self.users_collection.insert_one(user_doc)
        print(f"✅ Created user: {new_id} ({display_name})")
        return {"user_id": new_id, "display_name": display_name.strip()}

    async def get_user(self, user_id: str) -> Optional[dict]:
        """Return user doc if exists, else None."""
        uid = self._normalize(user_id)
        doc = await self.users_collection.find_one({"user_id": uid}, {"_id": 0})
        return doc

    async def user_exists(self, user_id: str) -> bool:
        uid = self._normalize(user_id)
        return await self.users_collection.find_one({"user_id": uid}) is not None

    async def login_user(self, user_id: str) -> Optional[dict]:
        """Update last_login and return user. Returns None if not found."""
        uid = self._normalize(user_id)
        doc = await self.users_collection.find_one_and_update(
            {"user_id": uid},
            {"$set": {"last_login": datetime.utcnow()}},
            {"_id": 0},
            return_document=True,
        )
        return doc

    async def list_all_users(self) -> list[dict]:
        """Return all registered users sorted by creation date."""
        cursor = self.users_collection.find({}, {"_id": 0}).sort("created_at", 1)
        return await cursor.to_list(length=200)

    async def delete_user_account(self, user_id: str):
        """Remove user from the users registry."""
        uid = self._normalize(user_id)
        await self.users_collection.delete_one({"user_id": uid})

    # ── File Metadata ────────────────────────────────────────

    async def save_file_metadata(self, user_id: str, filename: str):
        """
        Store file metadata properly in MongoDB user_files:
        {
            user_id:      "user_01",
            display_name: "Alice",
            files: [
                { filename: "Resume.pdf", uploaded_at: <datetime> },
                ...
            ],
            last_updated: <datetime>
        }
        """
        uid = self._normalize(user_id)
        try:
            # Fetch display_name from users collection
            user_doc     = await self.users_collection.find_one({"user_id": uid})
            display_name = user_doc.get("display_name", uid) if user_doc else uid

            file_entry = {
                "filename":    filename,
                "uploaded_at": datetime.utcnow(),
            }

            # $addToSet won't dedup dicts — use $pull then $push to avoid duplicates
            await self.files_collection.update_one(
                {"user_id": uid},
                {"$pull": {"files": {"filename": filename}}},  # remove old entry if exists
            )
            await self.files_collection.update_one(
                {"user_id": uid},
                {
                    "$push": {"files": file_entry},
                    "$set":  {
                        "display_name": display_name,
                        "last_updated": datetime.utcnow(),
                    },
                },
                upsert=True,
            )
            print(f"✅ MongoDB user_files: user={uid} ({display_name}), file={filename}")
        except Exception as e:
            print(f"MongoDB Metadata Save Error: {e}")

    async def delete_file_metadata(self, user_id: str, filename: str):
        uid = self._normalize(user_id)
        try:
            await self.files_collection.update_one(
                {"user_id": uid},
                {
                    "$pull": {"files": filename},
                    "$set":  {"last_updated": datetime.utcnow()},
                },
            )
        except Exception as e:
            print(f"MongoDB Metadata Delete Error: {e}")

    async def get_user_files(self, user_id: str) -> list[str]:
        uid = self._normalize(user_id)
        doc = await self.files_collection.find_one({"user_id": uid})
        return doc.get("files", []) if doc else []

    async def get_user_context(self, user_id: str):
        uid = self._normalize(user_id)
        return await self.files_collection.find_one({"user_id": uid})

    # ── Chat History ─────────────────────────────────────────

    async def save_chat_log(self, user_id: str, question: str, answer: str):
        uid = self._normalize(user_id)
        if isinstance(answer, list):
            answer = ", ".join(str(a) for a in answer)
        elif not isinstance(answer, str):
            answer = str(answer)
        try:
            await self.chat_collection.insert_one({
                "log_id":    str(uuid.uuid4()),
                "user_id":   uid,
                "question":  question,
                "answer":    answer,
                "timestamp": datetime.utcnow(),
            })
        except Exception as e:
            print(f"MongoDB Chat Log Save Error: {e}")

    async def get_chat_history(self, user_id: str, limit: int = 50) -> list:
        uid = self._normalize(user_id)
        cursor = self.chat_collection.find(
            {"user_id": uid}, {"_id": 0}
        ).sort("timestamp", -1).limit(limit)
        return await cursor.to_list(length=limit)

    async def delete_chat_history(self, user_id: str) -> int:
        uid = self._normalize(user_id)
        try:
            result = await self.chat_collection.delete_many({"user_id": uid})
            print(f"🗑️ Deleted {result.deleted_count} chat log(s) for {uid}")
            return result.deleted_count
        except Exception as e:
            print(f"MongoDB Chat Delete Error: {e}")
            return 0

    # ── Delete Everything ────────────────────────────────────

    async def delete_all_user_data(self, user_id: str):
        """
        Full wipe: chat history + file metadata + user account entry.
        Disk and ChromaDB are handled separately in routes.py.
        """
        uid = self._normalize(user_id)
        try:
            chat_res = await self.chat_collection.delete_many({"user_id": uid})
            await self.files_collection.delete_one({"user_id": uid})
            await self.users_collection.delete_one({"user_id": uid})
            print(f"🗑️ Full wipe for {uid}: {chat_res.deleted_count} chats deleted")
            return chat_res.deleted_count
        except Exception as e:
            print(f"MongoDB Full Delete Error: {e}")
            return 0


db_manager = MongoDBManager()
