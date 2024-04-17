from helpers.call_openai import call_openai_api


def language_detection(question: str) -> str:
        # question=que.get("question")
        response = call_openai_api(
            messages=[
                {"role": "system", "content": "You are a great language detector. You will be given a text or a question and your task is to only identify the language of that text or question and return it in string. Please note that you have to only return the language and not to answer that text if that text is a question."},
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
                {"role": "user", "content": "If 1=3 2=3 3=5 4=4 5=4.Then, 6=?"},
                {"role": "assistant", "content": "English"},
                {"role": "user", "content": "Add 8.563 and 4.8292."},
                {"role": "assistant", "content": "English"},
                {"role": "user", "content": f"{question}"}
            ]
        )
        if not hasattr(response, "choices"):
            return response
        return response.choices[0].message.content
    
    
# question="అమెరికా అధ్యక్షుడు ఎవరు(2023)?"
# print(language_detection(question))