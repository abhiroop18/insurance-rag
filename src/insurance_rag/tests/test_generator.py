from insurance_rag.generation.generator import InsuranceGenerator


def main():

    generator = InsuranceGenerator()

    question = "What is the grace period if I pay my premium monthly?"

    contexts = [
        """
        The grace period for payment of premium shall be fifteen days
        where premium payment mode is monthly and thirty days in all
        other cases. Coverage will be available during the grace period
        if the premium is paid in instalments during the policy period.
        """
    ]

    answer = generator.generate(
        question=question,
        contexts=contexts,
    )

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()