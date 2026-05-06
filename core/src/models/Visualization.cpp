/**
 * @file Visualization.cpp
 * @brief Implementation of tree-analysis visitors and geometric
 *        utilities for visualization.
 *
 * @see Visualization.hpp for the public API and module overview.
 */

#include "models/Visualization.hpp"

#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace ppforest2::viz {
  /**
   * NodeDataVisitor
   *
   * Traverses the tree in pre-order, routing observations by the split
   * condition at each internal node.  At each condition node, every
   * reaching observation is projected onto the node's projector, and
   * the projected value is compared against the cutpoint to partition
   * observations into the lower and upper children.
   *
   * The result is a pre-order sequence of NodeData structs containing:
   *   - For internal nodes: projected values, group labels, projector,
   *     cutpoint (used for histogram rendering in the tree diagram).
   *   - For leaf nodes: group labels of reaching observations, and the
   *     predicted group (used for leaf labels and coloring).
   */
  NodeDataVisitor::NodeDataVisitor(types::FeatureMatrix const& x, types::OutcomeVector const& y)
      : x(x)
      , y(y)
      , depth(0) {
    // Initially all observations are routed through the root.
    indices.resize(static_cast<std::size_t>(x.rows()));

    for (int i = 0; i < x.rows(); ++i) {
      indices[static_cast<std::size_t>(i)] = i;
    }
  }

  void NodeDataVisitor::visit(TreeBranch const& node) {
    // Collect projected values for all observations reaching this node.
    NodeData nd;
    nd.is_leaf   = false;
    nd.depth     = depth;
    nd.projector = node.projector;
    nd.cutpoint  = node.cutpoint;
    nd.value     = 0;

    // Partition observations into lower/upper based on projected value.
    std::vector<int> lower_indices;
    std::vector<int> upper_indices;

    for (int idx : indices) {
      types::Feature proj_val = x.row(idx).dot(node.projector);
      nd.projected_values.push_back(proj_val);
      nd.groups.push_back(y(idx));

      if (proj_val < node.cutpoint) {
        lower_indices.push_back(idx);
      } else {
        upper_indices.push_back(idx);
      }
    }

    nodes.push_back(std::move(nd));

    // Save and swap indices for child traversal.
    auto saved_indices = std::move(indices);

    indices = std::move(lower_indices);
    depth++;
    node.lower->accept(*this);

    indices = std::move(upper_indices);
    node.upper->accept(*this);
    depth--;

    indices = std::move(saved_indices);
  }

  void NodeDataVisitor::visit(TreeLeaf const& node) {
    NodeData nd;
    nd.is_leaf  = true;
    nd.depth    = depth;
    nd.cutpoint = 0;
    nd.value    = node.value;

    for (int idx : indices) {
      nd.groups.push_back(y(idx));
    }

    nodes.push_back(std::move(nd));
  }

  /**
   * 2D Projection Helpers
   *
   * When visualizing a p-dimensional tree in 2D, we select two variables
   * (var_i, var_j) for the axes and hold all others at fixed values
   * (typically medians).  Each split's projector and cutpoint are then
   * reduced to 2D by extracting the two relevant components and
   * subtracting the fixed variables' contributions from the cutpoint.
   */
  types::FeatureVector project_2d(types::FeatureVector const& full_proj, int var_i, int var_j) {
    types::FeatureVector proj2d(2);
    proj2d(0) = full_proj(var_i);
    proj2d(1) = full_proj(var_j);
    return proj2d;
  }

  types::Feature adjust_cutpoint(
      types::FeatureVector const& full_proj,
      types::Feature thr,
      std::vector<std::pair<int, types::Feature>> const& fixed_vars
  ) {
    // t' = t - Σ projector_k * value_k  for each fixed variable k
    for (auto const& fv : fixed_vars) {
      thr -= full_proj(fv.first) * fv.second;
    }

    return thr;
  }

  HalfSpace project_halfspace_2d(
      HalfSpace const& hs, int var_i, int var_j, std::vector<std::pair<int, types::Feature>> const& fixed_vars
  ) {
    return {
        project_2d(hs.projector, var_i, var_j),
        adjust_cutpoint(hs.projector, hs.cutpoint, fixed_vars),
        hs.is_lower,
    };
  }

  /**
   * Parametric Line Clipping
   *
   * Decision boundaries in 2D are lines of the form  a^T x = t.  We
   * parametrize each line as  P(u) = P0 + u * D  where D is tangent to
   * the line (perpendicular to 'a') and P0 is any point on the line.
   *
   * Clipping to a rectangle and ancestor half-spaces reduces to
   * tightening the interval [u_min, u_max].  If u_min >= u_max after
   * all constraints, the line is fully outside the visible region.
   */
  bool clip_param_to_range(
      types::Feature origin,
      types::Feature direction,
      types::Feature range_min,
      types::Feature range_max,
      types::Feature& u_min,
      types::Feature& u_max
  ) {
    types::Feature const eps = static_cast<types::Feature>(1e-12);

    if (std::abs(direction) < eps) {
      // Line is parallel to this axis — check if origin is in range.
      return origin >= range_min && origin <= range_max;
    }

    types::Feature u1 = (range_min - origin) / direction;
    types::Feature u2 = (range_max - origin) / direction;

    if (direction < 0) {
      std::swap(u1, u2);
    }

    u_min = std::max(u_min, u1);
    u_max = std::min(u_max, u2);

    return u_min < u_max;
  }

  bool clip_boundary_2d(
      types::FeatureVector const& a,
      types::Feature cutpoint,
      std::vector<HalfSpace> const& constraints,
      types::Feature x_min,
      types::Feature x_max,
      types::Feature y_min,
      types::Feature y_max,
      BoundarySegment& segment,
      int depth
  ) {
    types::Feature const eps = static_cast<types::Feature>(1e-12);

    // Direction tangent to the boundary line a^T x = cutpoint.
    // Perpendicular to a = (a0, a1) is D = (-a1, a0).
    types::Feature dx = -a(1);
    types::Feature dy = a(0);

    // Find a point P0 on the line.
    types::Feature px = 0;
    types::Feature py = 0;

    if (std::abs(a(1)) > std::abs(a(0))) {
      py = cutpoint / a(1);
    } else {
      px = cutpoint / a(0);
    }

    // Start with a large parametric interval and clip.
    types::Feature u_min = static_cast<types::Feature>(-1e6);
    types::Feature u_max = static_cast<types::Feature>(1e6);

    // Clip to bounding box.
    if (!clip_param_to_range(px, dx, x_min, x_max, u_min, u_max)) {
      return false;
    }

    if (!clip_param_to_range(py, dy, y_min, y_max, u_min, u_max)) {
      return false;
    }

    // Clip against each ancestor half-space constraint.
    for (auto const& con : constraints) {
      types::Feature denom = con.projector(0) * dx + con.projector(1) * dy;

      if (std::abs(denom) < eps) {
        continue;
      }

      types::Feature u_intersect = (con.cutpoint - (con.projector(0) * px + con.projector(1) * py)) / denom;

      // is_lower keeps the side where projector^T x < cutpoint;
      // !is_lower keeps projector^T x >= cutpoint.  The sign of denom
      // determines which parametric bound to tighten.  Flipping
      // is_lower is equivalent to flipping the sign of denom.
      bool tighten_max = (denom > 0) == con.is_lower;

      if (tighten_max) {
        u_max = std::min(u_max, u_intersect);
      } else {
        u_min = std::max(u_min, u_intersect);
      }

      if (u_min >= u_max) {
        return false;
      }
    }

    // Compute clipped endpoints.
    segment.x_start = px + u_min * dx;
    segment.y_start = py + u_min * dy;
    segment.x_end   = px + u_max * dx;
    segment.y_end   = py + u_max * dy;
    segment.depth   = depth;
    return true;
  }

  /**
   * Polygon Clipping (Sutherland–Hodgman)
   *
   * Used by RegionVisitor to compute decision region polygons.  Each
   * leaf's region starts as the full bounding box rectangle, then is
   * progressively clipped by each ancestor's half-space constraint.
   * Since all constraints are linear half-spaces and the initial shape
   * is convex, the result is always a convex polygon (or empty).
   *
   * The key property for visualization: the union of all leaf polygons
   * tiles the entire bounding box.  Every point in the bbox is
   * classified by the tree into exactly one leaf, so the corresponding
   * leaf's polygon contains that point.  This guarantees complete
   * background coverage in the plot when the bbox matches the visible
   * coordinate range (i.e., no white gaps between regions).
   *
   * Complexity: O(k × n) per leaf, where k = tree depth (number of
   * ancestor constraints) and n = max polygon vertices at any step.
   * Since k and n are both bounded by tree depth, this is effectively
   * O(depth²) per leaf.
   */

  Polygon clip_polygon_halfspace(
      Polygon const& polygon, types::FeatureVector const& normal, types::Feature cutpoint, bool is_lower
  ) {
    if (polygon.empty()) {
      return {};
    }

    Polygon result;
    std::size_t n = polygon.size();

    for (std::size_t i = 0; i < n; ++i) {
      auto const& curr = polygon[i];
      auto const& next = polygon[(i + 1) % n];

      // Signed distance from the boundary: normal^T x - cutpoint
      types::Feature d_curr = normal(0) * curr.first + normal(1) * curr.second - cutpoint;
      types::Feature d_next = normal(0) * next.first + normal(1) * next.second - cutpoint;

      // Determine which side each vertex is on.
      //   is_lower (a^T x < t):  inside means d < 0
      //   !is_lower (a^T x >= t): inside means d >= 0
      bool curr_inside = is_lower ? (d_curr < 0) : (d_curr >= 0);
      bool next_inside = is_lower ? (d_next < 0) : (d_next >= 0);

      // Emit intersection when the edge crosses the boundary.
      if (curr_inside != next_inside) {
        types::Feature t = d_curr / (d_curr - d_next);
        result.emplace_back(curr.first + t * (next.first - curr.first), curr.second + t * (next.second - curr.second));
      }

      // Emit next vertex when it's inside.
      if (next_inside) {
        result.push_back(next);
      }
    }

    return result;
  }

  /**
   * SpatialVisitor
   *
   * Base class for visitors that project tree splits into a 2D feature
   * plane.  Implements the common traversal pattern: push the current
   * split as a constraint, recurse into both children (lower then
   * upper), and pop.  Subclasses override visit(TreeBranch) to add
   * per-node work and call the base for traversal.
   */

  SpatialVisitor::SpatialVisitor(
      int var_i,
      int var_j,
      std::vector<std::pair<int, types::Feature>> const& fixed_vars,
      types::Feature x_min,
      types::Feature x_max,
      types::Feature y_min,
      types::Feature y_max
  )
      : var_i(var_i)
      , var_j(var_j)
      , fixed_vars(fixed_vars)
      , x_min(x_min)
      , x_max(x_max)
      , y_min(y_min)
      , y_max(y_max)
      , depth(0) {}

  void SpatialVisitor::visit(TreeBranch const& node) {
    // Push this split as a constraint and recurse.
    constraints.emplace_back(node.projector, node.cutpoint, true);

    depth++;
    node.lower->accept(*this);
    depth--;

    // Flip to upper constraint, recurse right.
    constraints.back().is_lower = false;

    depth++;
    node.upper->accept(*this);
    depth--;

    constraints.pop_back();
  }

  void SpatialVisitor::visit(TreeLeaf const&) {}

  /**
   * BoundaryVisitor
   *
   * At each split node, projects the boundary and all ancestor
   * constraints to 2D, clips the boundary line, and stores visible
   * segments.  Then delegates to SpatialVisitor for constraint
   * management and child traversal.
   */

  void BoundaryVisitor::visit(TreeBranch const& node) {
    HalfSpace split2d = project_halfspace_2d({node.projector, node.cutpoint, true}, var_i, var_j, fixed_vars);

    // Project all ancestor constraints to 2D for clipping.
    std::vector<HalfSpace> constraints2d;
    constraints2d.reserve(constraints.size());

    for (auto const& con : constraints) {
      constraints2d.push_back(project_halfspace_2d(con, var_i, var_j, fixed_vars));
    }

    // Clip the boundary and store if visible.
    BoundarySegment seg;

    if (clip_boundary_2d(split2d.projector, split2d.cutpoint, constraints2d, x_min, x_max, y_min, y_max, seg, depth)) {
      segments.push_back(seg);
    }

    SpatialVisitor::visit(node);
  }

  /**
   * RegionVisitor
   *
   * At each leaf, computes the convex polygon for that leaf's decision
   * region by clipping the bounding box against all ancestor constraints
   * projected to 2D (Sutherland–Hodgman).  Traversal and constraint
   * management are handled by SpatialVisitor.
   */

  void RegionVisitor::visit(TreeLeaf const& node) {
    // Start with the bounding box as a rectangle (CCW winding).
    Polygon poly = {{x_min, y_min}, {x_max, y_min}, {x_max, y_max}, {x_min, y_max}};

    // Clip against each ancestor constraint in 2D.
    for (auto const& con : constraints) {
      HalfSpace hs2d = project_halfspace_2d(con, var_i, var_j, fixed_vars);
      poly           = clip_polygon_halfspace(poly, hs2d.projector, hs2d.cutpoint, hs2d.is_lower);

      if (poly.empty()) {
        return; // Region unreachable in this 2D slice.
      }
    }

    RegionPolygon region;
    region.vertices        = std::move(poly);
    region.predicted_group = node.value;
    regions.push_back(std::move(region));
  }

  /**
   * Tree Layout
   *
   * Computes (x, y) positions for rendering the tree as a diagram.
   * Uses a recursive bottom-up algorithm:
   *   1. Layout each subtree, tracking its width.
   *   2. Place sibling subtrees side-by-side with a gap.
   *   3. Position the parent above its left child (left-aligned style).
   *   4. y decreases with depth so the root is at the top.
   */

  namespace {
    /// Intermediate result for subtree layout computation.
    struct LayoutSubtree {
      types::Feature center_x; ///< x-coordinate of this subtree's root node.
      types::Feature width;    ///< Total horizontal span of this subtree.
    };

    /// Format a cutpoint value for edge labels (e.g. "1.50").
    std::string format_cutpoint(types::Feature t) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2) << t;
      return oss.str();
    }

    /**
     * @brief Visitor that recursively lays out tree nodes.
     *
     * Carries shared mutable state (depth, node_idx, x_offset) and
     * writes positioned nodes/edges into output vectors.  After
     * accepting a node, the result is available in @c result.
     */
    struct LayoutVisitor : TreeNode::Visitor {
      int depth;
      int& node_idx;
      types::Feature x_offset;
      LayoutParams const& params;
      std::vector<LayoutNode>& nodes;
      std::vector<LayoutEdge>& edges;

      LayoutSubtree result;

      LayoutVisitor(
          int depth,
          int& node_idx,
          types::Feature x_offset,
          LayoutParams const& params,
          std::vector<LayoutNode>& nodes,
          std::vector<LayoutEdge>& edges
      )
          : depth(depth)
          , node_idx(node_idx)
          , x_offset(x_offset)
          , params(params)
          , nodes(nodes)
          , edges(edges)
          , result{0, 0} {}

      void visit(TreeLeaf const& /* response */) override {
        types::Feature y_pos = -static_cast<types::Feature>(depth) * params.y_spacing;
        int my_idx           = node_idx;
        node_idx++;

        types::Feature cx = x_offset + params.leaf_w / 2;
        nodes.push_back({cx, y_pos, true, my_idx});

        result = {cx, params.leaf_w + 0.1f};
      }

      void visit(TreeBranch const& condition) override {
        types::Feature y_pos = -static_cast<types::Feature>(depth) * params.y_spacing;
        int my_idx           = node_idx;
        node_idx++;

        // Reserve slot (pre-order), x filled below.
        std::size_t my_slot = nodes.size();
        nodes.push_back({0, y_pos, false, my_idx});

        LayoutVisitor left_visitor(depth + 1, node_idx, x_offset, params, nodes, edges);
        condition.lower->accept(left_visitor);

        LayoutVisitor right_visitor(
            depth + 1, node_idx, x_offset + left_visitor.result.width + params.gap, params, nodes, edges
        );
        condition.upper->accept(right_visitor);

        // Left-aligned: parent centered above left child.
        types::Feature center_x = left_visitor.result.center_x;
        nodes[my_slot].x        = center_x;

        // Compute edge endpoints (parent bottom -> child top).
        types::Feature from_y  = y_pos - params.node_h / 2;
        types::Feature child_y = -static_cast<types::Feature>(depth + 1) * params.y_spacing;

        types::Feature left_child_h  = is_leaf(condition.lower) ? params.leaf_h : params.node_h;
        types::Feature right_child_h = is_leaf(condition.upper) ? params.leaf_h : params.node_h;

        std::string thr = format_cutpoint(condition.cutpoint);

        edges.push_back({center_x, from_y, left_visitor.result.center_x, child_y + left_child_h / 2, "< " + thr});

        edges.push_back({
            center_x,
            from_y,
            right_visitor.result.center_x,
            child_y + right_child_h / 2,
            "\xe2\x89\xa5 " + thr // UTF-8 for ≥
        });

        types::Feature total_width = left_visitor.result.width + params.gap + right_visitor.result.width;
        types::Feature min_width   = params.node_w + 0.1f;

        result = {center_x, std::max(total_width, min_width)};
      }
    };
  }

  TreeLayout compute_tree_layout(TreeNode const& root, LayoutParams const& params) {
    std::vector<LayoutNode> nodes;
    std::vector<LayoutEdge> edges;
    int node_idx = 0;

    LayoutVisitor visitor(0, node_idx, 0.0f, params, nodes, edges);
    root.accept(visitor);

    return {std::move(nodes), std::move(edges)};
  }
}
