"""Quick manual demo: python3 demo.py "your message here" """
import json
import sys
from src.agent import handle_message


def main():
    if len(sys.argv) > 1:
        msg = " ".join(sys.argv[1:])
    else:
        msg = "@AmazonHelp my order still hasn't shown up, its been 6 days, this is ridiculous"
    result = handle_message(msg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
