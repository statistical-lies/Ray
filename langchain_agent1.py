# langchain_agent.py

import httpx
from langchain_ollama import ChatOllama
from langchain.tools import tool


# CONFIG

SERVER_URL = "http://127.0.0.1:8000/analyze"

RED_PATH = r"C:\Users\User\all data\band4.tif"
NIR_PATH = r"C:\Users\User\all data\band5.tif"


# TOOL

@tool
def calculate_ndvi(red_path: str, nir_path: str) -> str:
    """Calculate NDVI from Red and NIR bands."""
    try:
        with httpx.Client(timeout=None) as client:
            response = client.post(SERVER_URL, json={
                "red_path": red_path,
                "nir_path": nir_path
            })
            response.raise_for_status()
            data = response.json()

        mean_ndvi = data.get("average_ndvi", 0.0)

        if mean_ndvi < 0.1:
            health = "very low (mostly bare soil)"
        elif mean_ndvi < 0.3:
            health = "low"
        elif mean_ndvi < 0.5:
            health = "moderate"
        else:
            health = "good"

        return f"NDVI = {mean_ndvi:.4f}. Vegetation health: {health}."

    except Exception as e:
        return f"Error: {str(e)}"


# MAIN

def main():
    print("NDVI Analysis Agent\n")

    # Initialize LLM
    llm = ChatOllama(
        model="gemma2:2b",         
        temperature=0.0,
        base_url="http://127.0.0.1:11434"
    )

    # Direct tool call
    print("Calculating NDVI...")
    result = calculate_ndvi.invoke({"red_path": RED_PATH, "nir_path": NIR_PATH})

    print(f"\nTool Result: {result}")

    # summary
    print("\nGenerating summary...")
    summary = llm.invoke(
        f"NDVI Analysis: {result}\n\n"
        "Write a short, professional 2-3 sentence summary of the vegetation health."
    )

    print("\n REPORT ")
    print(summary.content.strip())


if __name__ == "__main__":
    main()