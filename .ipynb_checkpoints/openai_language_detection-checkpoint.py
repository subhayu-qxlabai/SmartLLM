from helpers.call_openai import call_openai_api


def language_detection(self, que: dict) -> str:
        question=que.get("question")
        response = call_openai_api(
            messages=[
                {"role": "system", "content": "You are a great language detector. You should only return the identified language in string. If you are not able to identify the language then just return 'Language Invalid'."},
                {"role": "user", "content": "बिना सोचे समझे"},
                {"role": "assistant", "content": "Hindi"},
                {"role": "user", "content": "¡Qué bien te ves!"},
                {"role": "assistant", "content": "Spanish"},
                {"role": "user", "content": "నేను భారతదేశంలో నివసిస్తున్నాను"},
                {"role": "assistant", "content": "Telugu"},
                {"role": "user", "content": "Chi è il presidente degli Stati Uniti?"},
                {"role": "assistant", "content": "Italian"},
                {"role": "user", "content": "What are the latest developments in quantum computing as of 2023?"},
                {"role": "assistant", "content": "English"},
                {"role": "user", "content": f"{question}"}
            ]
        )
        if not hasattr(response, "choices"):
            return response
        return response.choices[0].message.content
    
    
question={"question":"అమెరికా అధ్యక్షుడు ఎవరు?"}
language_detection(question)