"""
Root-level server/app.py — required by OpenEnv validate.
Delegates to gridops.server.app for all functionality.
"""

from gridops.server.app import app  # noqa: F401


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run("gridops.server.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
