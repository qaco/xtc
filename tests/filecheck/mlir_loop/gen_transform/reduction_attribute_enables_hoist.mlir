// RUN: mlir-loop --no-alias --print-source-ir %s 2>&1 | filecheck %s --check-prefix=WITH
// RUN: sed '/loop.reduction/d' %s > %t.mlir && mlir-loop --no-alias --print-source-ir %t.mlir 2>&1 | filecheck %s --check-prefix=WITHOUT
// UNSUPPORTED: mlir-target=nvgpu

func.func @matmul(
  %A: memref<4x8xf32>,
  %B: memref<8x16xf32>,
  %C: memref<4x16xf32>
) {
  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.reduction = ["k"],
      loop.schedule = {
        "i",
          "j",
            "k",
              "i#2" = {"unroll"},
                "j#16" = {"vectorize"}
      }
    }
    ins(%A, %B : memref<4x8xf32>, memref<8x16xf32>)
    outs(%C : memref<4x16xf32>)
  return
}

// With `loop.reduction`, the unroll is followed by the accumulator hoist.
// WITH:      transform.loop.unroll %{{.*}} {factor = 2 : i64}
// WITH-NEXT: %[[FN:.*]] = transform.get_parent_op %{{.*}} {isolated_from_above}
// WITH-NEXT: %[[RS:.*]] = transform.apply_registered_pass "resolve-ranked-shaped-type-result-dims" to %[[FN]]
// WITH-NEXT: transform.apply_cse to %[[RS]]
// WITH-NEXT: transform.structured.hoist_redundant_vector_transfers %[[RS]]

// Without it, the unroll is followed straight by the lowering patterns (the
// adjacency leaves no room for the hoist block), and the hoist is never emitted
// anywhere after.
// WITHOUT:      transform.loop.unroll %{{.*}} {factor = 2 : i64}
// WITHOUT-NEXT: %[[FN:.*]] = transform.get_parent_op %{{.*}} {isolated_from_above}
// WITHOUT-NEXT: transform.apply_patterns to %[[FN]] {
// WITHOUT-NOT:  transform.structured.hoist_redundant_vector_transfers
