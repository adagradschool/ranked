import json
import logging
from openai import OpenAI


SYSTEM_PROMPT = """You are a search query generation assistant for SEO optimization.

Given content about a business (restaurant, healthcare clinic, etc.), generate 10 realistic search queries that real users would type into Google or ask an AI assistant to find this business.

These queries should:
- Sound natural, like how real people search (e.g., "best pizza restaurants in manhattan", "urgent care clinic open late near me")
- Include location-specific terms when relevant (neighborhood, city, area)
- Cover different search intents: discovery ("best X in Y"), specific needs ("X that accepts insurance"), comparisons ("top rated X near me")
- Help this business appear in search results and AI recommendations
- Be diverse - mix short queries ("nyc sushi") with longer natural questions ("where can I find good sushi in the east village")

Return JSON in this exact format:
{
  "questions": [
    {"query": "best pizza in manhattan", "intent": "discovery"},
    {"query": "pizza restaurants open late nyc", "intent": "specific_need"},
    {"query": "where to get authentic italian pizza near times square", "intent": "natural_question"}
  ]
}

Generate exactly 10 search queries based on the business content provided."""


def generate_questions(
    client: OpenAI,
    text: str,
    business_name: str = "",
    category: str = "",
    model: str = "gpt-4o",
    num_queries: int = 10,
) -> list[dict]:
    if not text:
        return []

    context = f"Business: {business_name}\nCategory: {category}\n\n" if business_name else ""
    user_prompt = f"{context}Generate {num_queries} search queries based on this content:\n\n{text[:8000]}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        data = json.loads(content)
        return data.get("questions", [])
    except json.JSONDecodeError as e:
        logging.error("Failed to parse OpenAI response: %s", e)
        return []
    except Exception as e:
        logging.error("OpenAI API error: %s", e)
        return []
