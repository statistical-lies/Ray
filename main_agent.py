# main_agent.py
import httpx
from langchain_ollama import ChatOllama

SERVER_URL = "http://127.0.0.1:8000/analyze"

def run_terminal_demo():
    print("\n Raster processing terminal")

    payload = {
        "red_path": r"C:\Users\User\all data\band4.tif",
        "nir_path": r"C:\Users\User\all data\band5.tif"
    }

    # Call Ray Backend
    try:
        print("Sending task to Ray...")
        with httpx.Client(timeout=None) as client:
            response = client.post(SERVER_URL, json=payload)
            response.raise_for_status()
            data = response.json()

        mean_ndvi = data.get("average_ndvi")
        print(f"Ray Success! Mean NDVI = {mean_ndvi:.4f}")

    except Exception as e:
        print(f" Ray failed: {e}")
        return

    # use Ollama for nice summary (with fallback)
    try:
        print("Asking Ollama to write summary after NDVI is calculated...")
        llm = ChatOllama(
            model="gemma2:2b",
            temperature=0.0,
            base_url="http://127.0.0.1:11434"
        )
        report = llm.invoke(
            f"NDVI is {mean_ndvi:.4f}. "
            "Write a short, professional 2-3 sentence summary of the forest/vegetation health."
        )
        print(f"\n REPORT:\n{report.content}")

    except Exception as e:
        print(f"Ollama failed: {e}")
        print("\n SIMPLE REPORT (Ollama unavailable):")
        if mean_ndvi < 0.1:
            print("Vegetation health is very low. The area appears mostly bare soil or non-vegetated.")
        elif mean_ndvi < 0.3:
            print("Vegetation health is low. Sparse or stressed vegetation detected.")
        elif mean_ndvi < 0.5:
            print("Moderate vegetation health. Some healthy green areas present.")
        else:
            print("Good vegetation health. Area shows significant healthy vegetation.")

if __name__ == "__main__":
    run_terminal_demo()