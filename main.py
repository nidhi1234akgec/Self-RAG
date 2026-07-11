from rag.graph import ask


def main():

    while True:

        question = input("\nAsk a Question (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        result = ask(question)

        print("\nAnswer:\n")

        print(result["answer"])


if __name__ == "__main__":
    main()