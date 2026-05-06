describe("binarize_largest_gap", {
  it("creates a binarize_strategy object", {
    s <- binarize_largest_gap()
    expect_s3_class(s, "binarize_strategy")
    expect_equal(s$name, "largest_gap")
  })
})

describe("binarize_disabled", {
  it("creates a binarize_strategy object", {
    s <- binarize_disabled()
    expect_s3_class(s, "binarize_strategy")
    expect_equal(s$name, "disabled")
  })
})
