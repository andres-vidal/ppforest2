describe("grouping_by_label", {
  it("creates a grouping_strategy object", {
    s <- grouping_by_label()
    expect_s3_class(s, "grouping_strategy")
    expect_equal(s$name, "by_label")
  })
})

describe("grouping_by_cutpoint", {
  it("creates a grouping_strategy object", {
    s <- grouping_by_cutpoint()
    expect_s3_class(s, "grouping_strategy")
    expect_equal(s$name, "by_cutpoint")
  })
})
