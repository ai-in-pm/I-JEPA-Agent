import os
import sys
import argparse
import subprocess
import traceback

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    try:
        parser = argparse.ArgumentParser(description="Run I-JEPA Demonstration")
        parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on")
        parser.add_argument("--debug", action="store_true", help="Run in debug mode with extra logging")
        args = parser.parse_args()

        # Set up paths relative to the script directory
        app_path = os.path.join(script_dir, "app.py")

        if args.debug:
            print(f"App path: {app_path}")

        # Verify app.py exists
        if not os.path.exists(app_path):
            print(f"Error: App file not found at {app_path}")
            print("Files in script directory:")
            print(os.listdir(script_dir))
            sys.exit(1)

        # Run the Streamlit app
        print(f"Starting I-JEPA demonstration on port {args.port}...")
        print(f"Running: streamlit run {app_path} --server.port {args.port}")

        # Use full path to streamlit if needed
        streamlit_cmd = "streamlit"

        subprocess.run([
            streamlit_cmd, "run",
            app_path,
            "--server.port", str(args.port)
        ])
    except Exception as e:
        print(f"Error running demonstration: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
