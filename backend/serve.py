"""Start the local Tralens HTTP API and its bundled diagnostic viewer."""
from tools.dev_frontend import main


if __name__ == "__main__":
    main(reader=True)
