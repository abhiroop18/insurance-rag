from insurance_rag.guardrails.output_guardrail import (
    OutputGuardrail,
)


# =============================================================================
# TEST HELPER
# =============================================================================

def run_test(
    test_name: str,
    question: str,
    answer: str,
    contexts: list[str],
    expected_passed: bool,
):
    print("\n")
    print("=" * 80)
    print(test_name)
    print("=" * 80)

    print("\nQuestion:")
    print(question)

    print("\nRetrieved context:")

    for i, context in enumerate(contexts, start=1):

        print(f"\nChunk {i}:")
        print(context)

    print("\nGenerated answer:")
    print(answer)

    # -------------------------------------------------------------------------
    # Create guardrail
    # -------------------------------------------------------------------------

    guardrail = OutputGuardrail()

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    result = guardrail.validate(
        question=question,
        answer=answer,
        contexts=contexts,
    )

    # -------------------------------------------------------------------------
    # Print result
    # -------------------------------------------------------------------------

    print("\nGuardrail result:")

    print(
        f"Passed: "
        f"{result['passed']}"
    )

    print(
        f"Reason: "
        f"{result['reason']}"
    )

    print(
        f"Answer relevant: "
        f"{result['answer_relevant']}"
    )

    print(
        f"Answer grounded: "
        f"{result['answer_grounded']}"
    )

    print(
        f"Unsupported claims: "
        f"{result['unsupported_claims']}"
    )

    # -------------------------------------------------------------------------
    # Assertion
    # -------------------------------------------------------------------------

    assert (
        result["passed"]
        == expected_passed
    ), (
        f"Expected passed="
        f"{expected_passed}, "
        f"got passed="
        f"{result['passed']}"
    )

    print("\nASSERTIONS: PASSED")


# =============================================================================
# TEST 1
# CORRECT + GROUNDED ANSWER
# =============================================================================

run_test(

    test_name="TEST 1: CORRECT AND GROUNDED ANSWER",

    question=(
        "What is the waiting period "
        "for cataract surgery?"
    ),

    contexts=[
        (
            "Cataract surgery is subject to "
            "a waiting period of 24 months "
            "from the date of inception of "
            "the first policy with the insurer."
        ),

        (
            "The waiting period does not apply "
            "when the claim arises due to an accident."
        ),
    ],

    answer=(
        "The waiting period for cataract surgery "
        "is 24 months from the date of inception "
        "of the first policy with the insurer. "
        "The waiting period does not apply if "
        "the claim arises due to an accident."
    ),

    expected_passed=True,
)


# =============================================================================
# TEST 2
# IRRELEVANT ANSWER
# =============================================================================

run_test(

    test_name="TEST 2: IRRELEVANT ANSWER",

    question=(
        "What is the waiting period "
        "for cataract surgery?"
    ),

    contexts=[
        (
            "Cataract surgery is subject to "
            "a waiting period of 24 months."
        ),
    ],

    answer=(
        "The policy provides access to "
        "health assistance services and "
        "appointment scheduling."
    ),

    expected_passed=False,
)


# =============================================================================
# TEST 3
# HALLUCINATED / UNSUPPORTED ANSWER
# =============================================================================

run_test(

    test_name="TEST 3: HALLUCINATED ANSWER",

    question=(
        "What is the waiting period "
        "for cataract surgery?"
    ),

    contexts=[
        (
            "Cataract surgery is subject to "
            "a waiting period of 24 months "
            "from the date of inception of "
            "the first policy with the insurer."
        ),
    ],

    answer=(
        "The waiting period for cataract surgery "
        "is 12 months. After 12 months, the insurer "
        "will pay the full cost of the surgery."
    ),

    expected_passed=False,
)


# =============================================================================
# TEST 4
# PARTIALLY UNSUPPORTED ANSWER
# =============================================================================

run_test(

    test_name="TEST 4: PARTIALLY UNSUPPORTED ANSWER",

    question=(
        "What is the waiting period "
        "for cataract surgery?"
    ),

    contexts=[
        (
            "Cataract surgery is subject to "
            "a waiting period of 24 months "
            "from the date of inception of "
            "the first policy with the insurer."
        ),
    ],

    answer=(
        "The waiting period for cataract surgery "
        "is 24 months. The insurer will also "
        "reimburse medicines and hospital room "
        "charges without any limit."
    ),

    expected_passed=False,
)


# =============================================================================
# TEST 5
# CORRECT POLICY ANSWER
# =============================================================================

run_test(

    test_name="TEST 5: CORRECT HOSPITALIZATION ANSWER",

    question=(
        "Does the policy cover "
        "hospitalization expenses?"
    ),

    contexts=[
        (
            "Hospitalization expenses are covered "
            "subject to the terms, conditions, "
            "limits and exclusions specified "
            "in the policy."
        ),

        (
            "The insured must be hospitalized for "
            "at least 24 hours unless the treatment "
            "qualifies as a day-care procedure."
        ),
    ],

    answer=(
        "Yes. Hospitalization expenses are covered "
        "subject to the terms, conditions, limits "
        "and exclusions specified in the policy."
    ),

    expected_passed=True,
)


# =============================================================================
# TEST 6
# EMPTY ANSWER
# =============================================================================

run_test(

    test_name="TEST 6: EMPTY ANSWER",

    question=(
        "What is the waiting period "
        "for cataract surgery?"
    ),

    contexts=[
        (
            "Cataract surgery is subject to "
            "a waiting period of 24 months."
        ),
    ],

    answer="",

    expected_passed=False,
)


print("\n")
print("=" * 80)
print("OUTPUT GUARDRAIL TEST COMPLETED")
print("=" * 80)