import os
import subprocess
from typing import Optional


def main(saved_file_path: Optional[str] = None) -> None:
    print("Launching Dashboard...")
    # Streamlit command: streamlit run smds/dashboard.py
    visualizer_path = os.path.join(os.path.dirname(__file__), "dashboard.py")

    if saved_file_path:
        subprocess.run(["streamlit", "run", visualizer_path, "--", saved_file_path])
    else:
        subprocess.run(["streamlit", "run", visualizer_path])


if __name__ == "__main__":
    main()
