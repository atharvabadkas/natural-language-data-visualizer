from typing import List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path
import logging

class Message:
    def __init__(self, content: str, role: str, timestamp: Optional[datetime] = None):
        self.content = content
        self.role = role
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "role": self.role,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        return cls(
            content=data["content"],
            role=data["role"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )

class ConversationMemory:
    def __init__(self, session_id: str, memory_dir: str = "memory"):
        self.session_id = session_id
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.messages: List[Message] = []
        self.context: Dict = {}
        self._setup_logging()
    
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_message(self, content: str, role: str) -> None:
        """Add a new message to the conversation history."""
        message = Message(content=content, role=role)
        self.messages.append(message)
    
    def get_recent_messages(self, limit: int = 10) -> List[Dict]:
        """Get the most recent messages."""
        return [msg.to_dict() for msg in self.messages[-limit:]]
    
    def set_context(self, key: str, value: any) -> None:
        """Set a context value."""
        self.context[key] = value
    
    def get_context(self, key: str) -> Optional[any]:
        """Get a context value."""
        return self.context.get(key)
    
    def save_session(self) -> bool:
        """Save the current session to disk."""
        try:
            session_data = {
                "session_id": self.session_id,
                "messages": [msg.to_dict() for msg in self.messages],
                "context": self.context,
                "last_updated": datetime.now().isoformat()
            }
            
            filepath = self.memory_dir / f"session_{self.session_id}.json"
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            self.logger.info(f"Session saved successfully: {self.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving session: {str(e)}")
            return False
    
    def load_session(self) -> bool:
        """Load a session from disk."""
        try:
            filepath = self.memory_dir / f"session_{self.session_id}.json"
            if not filepath.exists():
                self.logger.warning(f"Session file not found: {self.session_id}")
                return False
            
            with open(filepath, 'r') as f:
                session_data = json.load(f)
            
            self.messages = [Message.from_dict(msg_data) for msg_data in session_data["messages"]]
            self.context = session_data["context"]
            
            self.logger.info(f"Session loaded successfully: {self.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading session: {str(e)}")
            return False
    
    def clear_session(self) -> None:
        """Clear the current session data."""
        self.messages = []
        self.context = {}
    
    def get_context_summary(self) -> Dict:
        """Get a summary of the current context."""
        return {
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "context_keys": list(self.context.keys()),
            "last_message": self.messages[-1].to_dict() if self.messages else None
        } 