import pytest
from pydantic import ValidationError

from workflowai.core.domain.completion import AudioContent, DocumentContent, ImageContent, Message, TextContent


class TestMessage:
    def test_basic_text(self):
        # Test basic text message validation
        json_str = '{"role": "user", "content": "Hello, world!"}'
        message = Message.model_validate_json(json_str)
        assert message.role == "user"
        assert message.content == "Hello, world!"

    def test_with_text_content(self):
        # Test message with TextContent
        json_str = """
        {
            "role": "assistant",
            "content": {
                "type": "text",
                "text": "This is a test message"
            }
        }
        """
        message = Message.model_validate_json(json_str)
        assert message.role == "assistant"
        assert isinstance(message.content, TextContent)
        assert message.content.text == "This is a test message"

    def test_with_document_content(self):
        # Test message with DocumentContent
        json_str = """
        {
            "role": "user",
            "content": {
                "type": "document_url",
                "source": {
                    "url": "https://example.com/doc.pdf"
                }
            }
        }
        """
        message = Message.model_validate_json(json_str)
        assert message.role == "user"
        assert isinstance(message.content, DocumentContent)
        assert message.content.source.url == "https://example.com/doc.pdf"

    def test_with_image_content(self):
        # Test message with ImageContent
        json_str = """
        {
            "role": "user",
            "content": {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg"
                }
            }
        }
        """
        message = Message.model_validate_json(json_str)
        assert message.role == "user"
        assert isinstance(message.content, ImageContent)
        assert message.content.image_url.url == "https://example.com/image.jpg"

    def test_with_audio_content(self):
        # Test message with AudioContent
        json_str = """
        {
            "role": "user",
            "content": {
                "type": "audio_url",
                "audio_url": {
                    "url": "https://example.com/audio.mp3"
                }
            }
        }
        """
        message = Message.model_validate_json(json_str)
        assert message.role == "user"
        assert isinstance(message.content, AudioContent)
        assert message.content.audio_url.url == "https://example.com/audio.mp3"

    def test_empty_role(self):
        # Test message with empty role
        json_str = '{"role": "", "content": "Test message"}'
        message = Message.model_validate_json(json_str)
        assert message.role == ""
        assert message.content == "Test message"

    def test_missing_role(self):
        # Test message with missing role
        json_str = '{"content": "Test message"}'
        message = Message.model_validate_json(json_str)
        assert message.role == ""  # Default value
        assert message.content == "Test message"

    def test_invalid_content_type(self):
        # Test message with invalid content type
        json_str = """
        {
            "role": "user",
            "content": {
                "type": "invalid_type",
                "text": "This should fail"
            }
        }
        """
        with pytest.raises(ValidationError):
            Message.model_validate_json(json_str)

    def test_missing_content(self):
        # Test message with missing content
        json_str = '{"role": "user"}'
        message = Message.model_validate_json(json_str)
        assert message.role == "user"
        assert message.content == ""  # Default value

    def test_invalid_json(self):
        # Test with invalid JSON string
        json_str = '{"role": "user", "content": "Test message"'  # Missing closing brace
        with pytest.raises(ValidationError):
            Message.model_validate_json(json_str)
