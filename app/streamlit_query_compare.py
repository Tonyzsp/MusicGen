"""
New entrypoint for multi-mode retrieval compare UI.

Use this file name because the app supports both text and uploaded-audio query modes.
"""

from app.streamlit_text_compare import main


if __name__ == "__main__":
    main()
