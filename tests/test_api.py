
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Проверяем, что API запущен."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "RAG Assistant работает"


def test_ask_with_relevant_question():
    """Проверяем ответ на вопрос по документам."""
    response = client.post(
        "/ask",
        json={"question": "Что такое RAG и как он работает?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] != "Не нашёл информации по вашему вопросу"
    assert len(data["sources"]) > 0
    assert data["model_version"] == "intfloat/multilingual-e5-small"

    # Проверяем структуру источников.
    source = data["sources"][0]
    assert "source" in source
    assert "chunk_id" in source
    assert "score" in source
    assert "distance" in source


def test_ask_without_relevant_information():
    """Проверяем отказ отвечать на вопрос вне корпуса."""
    response = client.post(
        "/ask",
        json={"question": "Как приготовить борщ?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "Не нашёл информации по вашему вопросу"
    assert data["sources"] == []
    assert data["model_version"] == "intfloat/multilingual-e5-small"


def test_ask_with_empty_question():
    """Проверяем валидацию пустого вопроса."""
    response = client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422

