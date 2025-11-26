from datetime import datetime


def greet(name: str) -> str:
    """Return a friendly greeting that includes the current weekday."""
    today = datetime.now().strftime("%A")
    return f"Hello, {name}! Happy {today}."


def main() -> None:
    """Entry point for ad-hoc testing."""
    print(greet("Developer"))


if __name__ == "__main__":
    main()

