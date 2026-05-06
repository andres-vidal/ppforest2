describe("leaf_majority_vote", {
  it("creates a leaf_strategy object", {
    s <- leaf_majority_vote()
    expect_s3_class(s, "leaf_strategy")
    expect_equal(s$name, "majority_vote")
  })
})

describe("leaf_mean_response", {
  it("creates a leaf_strategy object", {
    s <- leaf_mean_response()
    expect_s3_class(s, "leaf_strategy")
    expect_equal(s$name, "mean_response")
  })
})
