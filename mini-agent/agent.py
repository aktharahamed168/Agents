import ollama

from tools import AVAILABLE_FUNCTIONS, TOOLS


MODEL = "llama3.2"


def run_agent(user_message: str) -> str:
    """
    Send a message to Ollama and execute any requested tools.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. "
                "Use the calculator tool whenever mathematical "
                "calculation is required. "
                "After receiving a tool result, give the user a clear "
                "final answer."
            ),
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    while True:
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )

        # Add the assistant response to the conversation.
        messages.append(response.message)

        # If there are no tool calls, the model has finished.
        if not response.message.tool_calls:
            return response.message.content

        # Execute every tool requested by the model.
        for tool_call in response.message.tool_calls:
            function_name = tool_call.function.name
            arguments = tool_call.function.arguments

            function_to_call = AVAILABLE_FUNCTIONS.get(function_name)

            if function_to_call is None:
                result = f"Unknown tool: {function_name}"
            else:
                try:
                    result = function_to_call(**arguments)
                except Exception as error:
                    result = f"Tool error: {error}"

            print(f"[Tool] {function_name}({arguments})")
            print(f"[Tool result] {result}")

            # Send the tool result back to Ollama.
            messages.append(
                {
                    "role": "tool",
                    "tool_name": function_name,
                    "content": str(result),
                }
            )


def main():
    print("=" * 50)
    print("Mini Ollama Agent")
    print("=" * 50)
    print(f"Model: {MODEL}")
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:
        try:
            user_input = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            answer = run_agent(user_input)
            print(f"Agent: {answer}")
            print()

        except Exception as error:
            print(f"Error: {error}")
            print(
                "Make sure Ollama is running and the model "
                f"'{MODEL}' is installed."
            )
            print()


if __name__ == "__main__":
    main()
