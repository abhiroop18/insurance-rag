from dotenv import load_dotenv

from insurance_rag.workflow.rag_workflow import (
    create_rag_graph,
)


load_dotenv()


def main():

    graph = create_rag_graph()

    question = (
        "What is the grace period if I pay "
        "my premium monthly?"
    )

    result = graph.invoke(
        {
            "question": question,
            "contexts": [],
            "answer": "",
        }
    )

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)

    print(question)

    print("\n" + "=" * 80)
    print("RETRIEVED CONTEXTS")
    print("=" * 80)

    for i, context in enumerate(
        result["contexts"],
        start=1,
    ):

        print(f"\n--- Context {i} ---")
        print(context)

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(result["answer"])


if __name__ == "__main__":

    main()