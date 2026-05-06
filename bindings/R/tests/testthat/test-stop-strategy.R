describe("stop_pure_node", {
  it("creates a stop_strategy object", {
    s <- stop_pure_node()
    expect_s3_class(s, "stop_strategy")
    expect_equal(s$name, "pure_node")
  })
})

describe("stop_min_size", {
  it("creates a stop_strategy object", {
    s <- stop_min_size(5L)
    expect_s3_class(s, "stop_strategy")
    expect_equal(s$name, "min_size")
    expect_equal(s$min_size, 5L)
  })

  it("rejects min_size < 2", {
    expect_error(stop_min_size(0L), ">= 2")
    expect_error(stop_min_size(1L), ">= 2")
  })
})

describe("stop_min_variance", {
  it("creates a stop_strategy object", {
    s <- stop_min_variance(0.01)
    expect_s3_class(s, "stop_strategy")
    expect_equal(s$name, "min_variance")
    expect_equal(s$threshold, 0.01)
  })

  it("rejects negative thresholds", {
    expect_error(stop_min_variance(-1), "non-negative")
  })
})

describe("stop_any", {
  it("combines rules", {
    s <- stop_any(stop_min_size(5L), stop_min_variance(0.01))
    expect_s3_class(s, "stop_strategy")
    expect_equal(s$name, "any")
    expect_length(s$rules, 2L)
  })

  it("rejects empty call", {
    expect_error(stop_any(), "at least one stop rule")
  })

  it("rejects non-stop_strategy arguments", {
    expect_error(stop_any("not a rule"), "stop_strategy")
  })
})
