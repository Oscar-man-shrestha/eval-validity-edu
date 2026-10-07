"""Start the NeuroTrace-DAG web demo.

    python3 start_demo.py
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("serve.app:app", host="127.0.0.1", port=8020, reload=False)
