from src.app import build_model
import os

def test_build_model():
    build_model()
    assert os.path.exists("saved_model/model.joblib")

import subprocess

def test_streamlit_app():
    # Run the Streamlit app as a separate process
    process = subprocess.Popen(["streamlit", "run", "src/app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Check if the process exited successfully
    assert process.returncode == 0, "Streamlit app failed to run"

    # Check if there are no error messages in the output
    assert not stderr, f"Streamlit app produced error output: {stderr.decode()}"

    # # Check if the expected output is present in the output
    # expected_output = "Your expected output"
    # assert expected_output in stdout.decode(), "Expected output not found in Streamlit app output"

if __name__ == "__main__":
    test_build_model()
    # test_streamlit_app()
    print("Done!")
