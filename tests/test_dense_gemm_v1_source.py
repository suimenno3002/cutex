import ast
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[1] / "cutex" / "kernels" / "dense_gemm_v1.py"
)


def test_dense_gemm_v1_is_implemented_without_tensor_core_calls():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert "NotImplementedError" not in names
    assert "tcgen05" not in names
    assert "gemm" not in attributes
    assert "sync_threads" in attributes


def test_dense_gemm_v1_preserves_the_five_pointer_launch_interface():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "dense_gemm_v1"
    )
    assert [argument.arg for argument in function.args.args] == [
        "a_ptr",
        "b_ptr",
        "sfa_ptr",
        "sfb_ptr",
        "c_ptr",
        "stream",
    ]

