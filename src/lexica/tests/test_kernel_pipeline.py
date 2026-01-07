import os
from lexica.pipeline.orchestration import run_pipeline


def test_end_to_end_kernel_pipeline():
    """
    Golden end-to-end test:
    NL -> IR -> IRL -> CAD -> STEP
    """

    output_path = "mock_llm_output.step"

    prompt = (
        "Create a box of size 40 by 30 by 20. "
        "Fillet all edges with radius 2. "
        "Export it as a STEP file."
    )

    # Cleanup from previous runs
    if os.path.exists(output_path):
        os.remove(output_path)

    run_pipeline(prompt, debug=False)

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
